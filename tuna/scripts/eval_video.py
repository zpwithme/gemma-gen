#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Batch AR video generation + evaluation driver.

Reads prompts, generates videos via Tuna2PixelARVideoPipeline (chunk-wise
AR pixel rollout), saves them as MP4 in a folder structure compatible with
VBench / other public video benchmarks.

Output layout::

    {output_dir}/
        meta.jsonl                          # per-video prompt + path + duration
        videos/
            0000_<prompt-slug>.mp4
            0001_<prompt-slug>.mp4
            ...
        frames/                              # optional, --save-frames
            0000_<prompt-slug>/
                frame_000.png
                frame_001.png
                ...

Usage
-----

::

    # Generate videos for a prompt list (resume-safe).
    python -m tuna.scripts.eval_video \\
        --model-config configs/model/tuna_2_pixel_gemma_12b.yaml \\
        --ckpt ./outputs/video/V-S3/merged/step_30000.pt \\
        --prompts vbench_prompts.txt \\
        --output-dir ./eval/video_v_s3 \\
        --num-frames 16 --frames-per-chunk 4 \\
        --height 512 --width 512 --fps 8

    # Compute VBench metrics (requires the official VBench package).
    python -m tuna.scripts.eval_video \\
        --output-dir ./eval/video_v_s3 \\
        --skip-gen --vbench-dimensions subject_consistency motion_smoothness aesthetic_quality

VBench Integration
------------------
After generation, the official VBench tool can ingest ``{output_dir}/videos``::

    # https://github.com/Vchitect/VBench
    cd VBench
    python evaluate.py --dimension subject_consistency motion_smoothness \\
        --videos_path /path/to/output_dir/videos \\
        --prompt_file /path/to/output_dir/meta.jsonl

When ``--vbench-dimensions`` is passed AND `vbench` python package is installed,
this script will call it directly and merge results into ``metrics.json``.
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

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image


logger: logging.Logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------------

def load_prompts(prompts_path: str) -> List[Dict[str, Any]]:
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
# MP4 writing
# -----------------------------------------------------------------------

def write_mp4(frames: List[Image.Image], out_path: Path, fps: int = 8) -> None:
    """Encode a list of PIL frames as MP4. Prefers imageio[ffmpeg],
    falls back to OpenCV. Both are common Tuna deps."""
    try:
        import imageio.v2 as imageio  # type: ignore

        arrays = [np.array(f) for f in frames]
        imageio.mimsave(str(out_path), arrays, fps=fps, codec="libx264", quality=8)
        return
    except Exception as e:
        logger.debug(f"imageio failed ({e}); trying OpenCV.")

    import cv2  # already in Tuna deps via VideoDataset
    h, w = frames[0].size[1], frames[0].size[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    try:
        for f in frames:
            arr = cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR)
            writer.write(arr)
    finally:
        writer.release()


# -----------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------

def run_generation(args: argparse.Namespace) -> None:
    from tuna.inference.runner import TunaInference

    prompts = load_prompts(args.prompts)
    videos_dir = Path(args.output_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.output_dir) / "meta.jsonl"
    frames_dir = Path(args.output_dir) / "frames" if args.save_frames else None
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Resume-safe
    done_indices: set[int] = set()
    if meta_path.exists() and not args.overwrite:
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "video_path" in rec and Path(rec["video_path"]).exists():
                        done_indices.add(int(rec.get("index", -1)))
                except Exception:
                    pass
        logger.info(f"Resume mode: skipping {len(done_indices)} already-generated videos")

    # Build model + runner
    logger.info(f"Loading model config: {args.model_config}")
    cfg = OmegaConf.load(args.model_config)
    model = instantiate(cfg)

    inference_kwargs: Dict[str, Any] = {
        "inference_mode": "t2v_ar",
        "pipe": "Tuna2PixelARVideoPipeline",
        "use_ckpt": True,
        "ckpt_path": args.ckpt,
        "use_chat_template": False,
        "weight_dtype": "bfloat16",
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frames_per_chunk": args.frames_per_chunk,
        "num_diffusion_steps_per_chunk": args.steps_per_chunk,
        "patch_size": args.patch_size,
        "guidance_scale": args.guidance,
        "noise_scale": args.noise_scale,
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
            base_name = f"{idx:05d}_{_slug(prompt)}"
            out_path = videos_dir / f"{base_name}.mp4"

            data = {"text": [prompt]}
            t0 = time.time()
            with torch.no_grad():
                outputs = runner(data, seed=args.seed + idx)
            elapsed = time.time() - t0
            t_total += elapsed

            # Extract frames (PIL list)
            frames = _extract_frames(outputs)
            if not frames:
                logger.warning(f"No frames extracted for prompt {idx}, skipping save.")
                continue

            # Write MP4
            write_mp4(frames, out_path, fps=args.fps)

            # Save raw frames if requested (helps VBench frame-level metrics)
            if frames_dir is not None:
                fdir = frames_dir / base_name
                fdir.mkdir(exist_ok=True)
                for fi, frm in enumerate(frames):
                    frm.save(fdir / f"frame_{fi:03d}.png")

            entry = {
                "index": idx,
                "prompt": prompt,
                "video_path": str(out_path),
                "num_frames": len(frames),
                "fps": args.fps,
                "generation_time_s": round(elapsed, 2),
            }
            for k in ("category", "dimension", "tag"):
                if k in rec:
                    entry[k] = rec[k]
            meta_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            meta_f.flush()

            n_generated += 1
            logger.info(
                f"[{n_generated}/{len(prompts) - len(done_indices)}] "
                f"{elapsed:.1f}s — {prompt[:60]}..."
            )
    finally:
        meta_f.close()

    logger.info(
        f"Generation done: {n_generated} videos in {videos_dir}. "
        f"Total time: {t_total:.1f}s (avg {t_total / max(n_generated, 1):.1f}s/video)."
    )


def _extract_frames(outputs: dict) -> List[Image.Image]:
    """Pull a list of PIL frames out of the runner's output dict."""
    # Tuna2PixelARVideoPipeline returns {"frames": [PIL, PIL, ...], "sentence": ...}
    if "frames" in outputs and outputs["frames"]:
        v = outputs["frames"]
        if isinstance(v, list) and isinstance(v[0], Image.Image):
            return v
        # tensor list fallback
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            return [_tensor_to_pil(t) for t in v]
    # videos key fallback
    if "videos" in outputs:
        v = outputs["videos"]
        if isinstance(v, list) and v and isinstance(v[0], list):
            return v[0]  # first video
    return []


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    arr = ((t.permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).byte().numpy()
    return Image.fromarray(arr)


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------

def compute_vbench(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    """Run a subset of VBench dimensions on the generated videos.

    Requires `vbench` (https://github.com/Vchitect/VBench) installed:
        pip install vbench
    """
    if not args.vbench_dimensions:
        return None
    try:
        from vbench import VBench
    except ImportError:
        logger.error(
            "vbench not installed. To run VBench locally: pip install vbench"
        )
        return None

    videos_path = str(Path(args.output_dir) / "videos")
    output_path = str(Path(args.output_dir) / "vbench")
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Running VBench on {videos_path} for dimensions: {args.vbench_dimensions}")
    vbench = VBench(
        device="cuda",
        full_info_dir=Path(args.output_dir) / "meta.jsonl",
        output_path=output_path,
    )
    results: Dict[str, float] = {}
    for dim in args.vbench_dimensions:
        try:
            score = vbench.evaluate(
                videos_path=videos_path,
                name=f"tuna_gemma_{dim}",
                dimension_list=[dim],
                local=True,
            )
            results[dim] = float(score) if not isinstance(score, dict) else score
        except Exception as e:
            logger.warning(f"VBench dimension {dim} failed: {e}")
    logger.info(f"VBench results: {results}")
    return results


def compute_clip_score_video(args: argparse.Namespace) -> Optional[float]:
    """Average CLIP score across video frames vs prompt.

    Simple proxy for text-video alignment when VBench isn't available.
    """
    try:
        from torchmetrics.multimodal.clip_score import CLIPScore
    except ImportError:
        logger.error("torchmetrics CLIP not installed.")
        return None

    meta_path = Path(args.output_dir) / "meta.jsonl"
    if not meta_path.exists():
        return None

    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
    all_scores: List[float] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            vp = rec["video_path"]
            prompt = rec["prompt"]
            frames = _read_video_frames(vp, max_frames=8)
            if not frames:
                continue
            scores = []
            for frm in frames:
                t = torch.from_numpy(np.array(frm)).permute(2, 0, 1).unsqueeze(0).to("cuda")
                scores.append(metric(t, [prompt]).item())
            all_scores.append(sum(scores) / max(len(scores), 1))
    avg = sum(all_scores) / max(len(all_scores), 1)
    logger.info(f"CLIP score (video, n={len(all_scores)}) = {avg:.4f}")
    return float(avg)


def _read_video_frames(path: str, max_frames: int = 8) -> List[Image.Image]:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    indices = [int(i * total / max_frames) for i in range(min(max_frames, total))]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch pixel-AR video generation + evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument("--prompts", default="assets/prompts.txt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-frames", action="store_true",
                   help="Also dump raw PNG frames per video (large; some metrics need this).")

    # Model + ckpt
    p.add_argument("--model-config", default="configs/model/tuna_2_pixel_gemma_12b.yaml")
    p.add_argument("--ckpt", default=None,
                   help="Merged single-file checkpoint (.pt or .safetensors).")

    # Video layout
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--frames-per-chunk", type=int, default=4)
    p.add_argument("--steps-per-chunk", type=int, default=8)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--fps", type=int, default=8)

    # Sampling
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--noise-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negative-prompt",
                   default="ugly, distorted, blurry, low quality, watermark")

    # Metrics
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--vbench-dimensions", nargs="*", default=None,
                   help="Subset of VBench dimensions, e.g. subject_consistency motion_smoothness "
                        "aesthetic_quality temporal_flickering imaging_quality.")
    p.add_argument("--clip-score", action="store_true",
                   help="Average per-frame CLIP score vs prompt.")

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
        logger.info("--skip-gen: skipping video generation.")

    vbench = compute_vbench(args)
    if vbench:
        results["vbench"] = vbench

    if args.clip_score:
        clip = compute_clip_score_video(args)
        if clip is not None:
            results["clip_score_video"] = clip

    if results:
        results_path = Path(args.output_dir) / "metrics.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Wrote metrics → {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
