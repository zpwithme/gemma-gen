#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batch image generation + evaluation driver.

Reads prompts from a file, generates images via the chosen pipeline
(Tuna-2 / Tuna-Gemma), saves them in a folder structure compatible with
standard T2I benchmarks (GenEval / DPG-Bench / FID / CLIP score), and
optionally runs the metric computation directly.

Output layout::

    {output_dir}/
        meta.jsonl                          # per-image prompt + path metadata
        images/
            0000_<prompt-slug>.png
            0001_<prompt-slug>.png
            ...

Usage
-----

::

    # 1. Generate images for a prompt list (resume-safe).
    python -m tuna.scripts.eval_image \\
        --model-config configs/model/tuna_2_pixel_gemma_12b.yaml \\
        --ckpt ./outputs/train_gemma/merged/step_50000.pt \\
        --prompts assets/prompts.txt \\
        --output-dir ./eval/s1_F_full \\
        --height 512 --width 512 --guidance 4.0 --steps 50

    # 2. Compute FID against a reference image dir (uses cleanfid if installed).
    python -m tuna.scripts.eval_image \\
        --output-dir ./eval/s1_F_full \\
        --skip-gen \\
        --fid-ref /data/mjhq30k/images

    # 3. Compute CLIP score (text-image alignment) — needs `clip_score`.
    python -m tuna.scripts.eval_image \\
        --output-dir ./eval/s1_F_full --skip-gen --clip-score

GenEval / DPG-Bench
-------------------
This script writes images + meta.jsonl in the layout those benchmarks
expect (one image per prompt, ordered by index). After running, point
the official benchmark tool at ``{output_dir}/images``::

    # GenEval (https://github.com/djghosh13/geneval)
    python evaluation/evaluate_images.py {output_dir}/images \\
        --metadata_file {output_dir}/meta.jsonl

    # DPG-Bench (https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench)
    python compute_dpg_bench.py --image_root {output_dir}/images \\
        --resolution 512 --pic_num 1

The repo's bundled ``lmms-eval/`` directory can also evaluate this folder
end-to-end with its tuna2 task definitions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


logger: logging.Logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------

def load_prompts(prompts_path: str) -> List[Dict[str, Any]]:
    """Read prompts from a .txt (one per line) or .jsonl (`{"prompt": ...}`).

    Returns a list of dicts, each at least ``{"index": int, "prompt": str}``.
    Extra fields in JSONL are preserved (useful for GenEval's metadata).
    """
    out: List[Dict[str, Any]] = []
    p = Path(prompts_path)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"{prompts_path}:{i+1}: bad JSON: {e}")
                    continue
                if "prompt" not in rec and "caption" in rec:
                    rec["prompt"] = rec["caption"]
                rec["index"] = i
                out.append(rec)
    else:
        # Plain text — one prompt per non-empty line.
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                out.append({"index": i, "prompt": line})

    logger.info(f"Loaded {len(out)} prompts from {prompts_path}")
    return out


def _slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9 _-]", "", s).strip().replace(" ", "_")
    return s[:max_len] if s else "prompt"


# -----------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------

def run_generation(args: argparse.Namespace) -> None:
    """Generate images for each prompt and save under args.output_dir."""
    from tuna.inference.runner import TunaInference

    prompts = load_prompts(args.prompts)
    images_dir = Path(args.output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.output_dir) / "meta.jsonl"

    # Resume-safe: skip prompts whose output already exists.
    done_indices: set[int] = set()
    if meta_path.exists() and not args.overwrite:
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "image_path" in rec and Path(rec["image_path"]).exists():
                        done_indices.add(int(rec.get("index", -1)))
                except Exception:
                    pass
        logger.info(f"Resume mode: skipping {len(done_indices)} already-generated prompts")

    # Build model + runner once.
    logger.info(f"Loading model config: {args.model_config}")
    cfg = OmegaConf.load(args.model_config)
    model = instantiate(cfg)

    inference_kwargs: Dict[str, Any] = {
        "inference_mode": "t2i",
        "pipe": args.pipe,
        "use_ckpt": True,
        "ckpt_path": args.ckpt,
        "use_chat_template": False,
        "add_aspect_ratio_embeds": getattr(cfg, "add_aspect_ratio_embeds", False),
        "generation_mode": "t2i_pixel",
        "weight_dtype": "bfloat16",
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "noise_scale": args.noise_scale,
        "shift": args.shift,
        "sampling_method": args.sampling,
        "num_images_per_prompt": 1,
        "negative_prompt": args.negative_prompt,
    }
    runner = TunaInference(model=model, **inference_kwargs)

    t_total = 0.0
    n_generated = 0
    meta_f = meta_path.open("a" if not args.overwrite else "w", encoding="utf-8")
    try:
        for rec in prompts:
            idx = rec["index"]
            if idx in done_indices:
                continue
            prompt = rec["prompt"]
            out_name = f"{idx:05d}_{_slug(prompt)}.png"
            out_path = images_dir / out_name

            data = {"text": [prompt]}
            t0 = time.time()
            with torch.no_grad():
                outputs = runner(data, seed=args.seed + idx)
            t_total += (time.time() - t0)

            # Save the first image (1 image per prompt by default).
            img = _extract_pil(outputs)
            img.save(out_path)

            # Record meta.
            entry = {
                "index": idx,
                "prompt": prompt,
                "image_path": str(out_path),
            }
            # Carry through extra fields useful for GenEval (e.g., category).
            for k in ("category", "tag", "include", "exclude"):
                if k in rec:
                    entry[k] = rec[k]
            meta_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            meta_f.flush()

            n_generated += 1
            if n_generated % 10 == 0:
                logger.info(
                    f"[{n_generated}/{len(prompts) - len(done_indices)}] "
                    f"avg {t_total / n_generated:.2f}s/img"
                )
    finally:
        meta_f.close()

    logger.info(
        f"Generation done: {n_generated} new images in {images_dir}. "
        f"Total time: {t_total:.1f}s (avg {t_total / max(n_generated, 1):.2f}s/img)."
    )


def _extract_pil(outputs: dict) -> Image.Image:
    """Pull the first PIL image out of the runner's heterogeneous output dict."""
    for key in ("images", "samples", "pil"):
        if key in outputs and outputs[key]:
            v = outputs[key]
            if isinstance(v, list) and isinstance(v[0], Image.Image):
                return v[0]
            if isinstance(v, list) and isinstance(v[0], torch.Tensor):
                t = v[0].detach().cpu()
                if t.dim() == 3:
                    arr = ((t.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
                    return Image.fromarray(arr)
    # Fallback: scan tensor outputs
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor) and v.dim() in (3, 4):
            t = v.detach().cpu()
            if t.dim() == 4:
                t = t[0]
            arr = ((t.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
            return Image.fromarray(arr)
    raise RuntimeError(f"Could not extract PIL image from outputs: {list(outputs.keys())}")


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_fid(args: argparse.Namespace) -> Optional[float]:
    """Compute FID between generated images and a reference directory.

    Uses ``cleanfid`` if available — gives the most reproducible numbers
    aligned with public benchmarks.
    """
    if args.fid_ref is None:
        return None
    try:
        from cleanfid import fid
    except ImportError:
        logger.error(
            "cleanfid not installed. Install with: pip install clean-fid"
        )
        return None

    images_dir = str(Path(args.output_dir) / "images")
    logger.info(f"Computing FID between {images_dir} and {args.fid_ref} ...")
    score = fid.compute_fid(images_dir, args.fid_ref, mode="clean", device="cuda")
    logger.info(f"FID = {score:.4f}")
    return float(score)


def compute_clip_score(args: argparse.Namespace) -> Optional[float]:
    """Compute average CLIP score between each image and its prompt.

    Uses ``torchmetrics.multimodal.clip_score`` (lightweight, no extra deps).
    """
    try:
        from torchmetrics.multimodal.clip_score import CLIPScore
    except ImportError:
        logger.error(
            "torchmetrics not installed with clip extras. "
            "Install: pip install 'torchmetrics[multimodal]'"
        )
        return None

    meta_path = Path(args.output_dir) / "meta.jsonl"
    if not meta_path.exists():
        logger.error(f"No meta.jsonl found at {meta_path}")
        return None

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
    scores: List[float] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            img = Image.open(rec["image_path"]).convert("RGB")
            t = torch.from_numpy(
                __import__("numpy").array(img)
            ).permute(2, 0, 1).unsqueeze(0).to("cuda")
            s = metric(t, [rec["prompt"]]).item()
            scores.append(s)
    avg = sum(scores) / max(len(scores), 1)
    logger.info(f"CLIP score (n={len(scores)}) = {avg:.4f}")
    return float(avg)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch T2I generation + evaluation driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--prompts", default="assets/prompts.txt",
                   help="Prompts file (.txt one-per-line or .jsonl with 'prompt'/'caption').")
    p.add_argument("--output-dir", required=True,
                   help="Directory for images/ and meta.jsonl.")
    p.add_argument("--overwrite", action="store_true",
                   help="Don't resume; regenerate all images from scratch.")

    # Model + pipeline
    p.add_argument("--model-config", default="configs/model/tuna_2_pixel_gemma_12b.yaml",
                   help="Hydra-instantiable model config (the model: section of a train config).")
    p.add_argument("--ckpt", default=None,
                   help="Path to merged single-file checkpoint (.pt or .safetensors).")
    p.add_argument("--pipe", default="Tuna2PixelPipeline",
                   choices=["Tuna2PixelPipeline", "Tuna2RPixelPipeline", "TunaPipeline"],
                   help="Inference pipeline class name.")

    # Generation
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--noise-scale", type=float, default=2.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--sampling", default="euler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt",
                   default="ugly, distorted, blurry, low quality, watermark")

    # Skip / metric phases
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip generation; only compute metrics on existing images/.")
    p.add_argument("--fid-ref", default=None,
                   help="Reference image directory for FID (uses cleanfid).")
    p.add_argument("--clip-score", action="store_true",
                   help="Compute CLIP score (text-image alignment).")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    results: Dict[str, Any] = {}
    if not args.skip_gen:
        run_generation(args)
    else:
        logger.info("--skip-gen: skipping image generation.")

    fid = compute_fid(args)
    if fid is not None:
        results["fid"] = fid

    if args.clip_score:
        clip = compute_clip_score(args)
        if clip is not None:
            results["clip_score"] = clip

    if results:
        results_path = Path(args.output_dir) / "metrics.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote metrics: {results}")
        logger.info(f"→ {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
