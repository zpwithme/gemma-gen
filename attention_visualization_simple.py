#!/usr/bin/env python3
"""
Tuna-2 Pixel (none-encoder, variant C) -- minimal attention visualization.

  1) Run a single forward pass and collect self-attention from every layer.
  2) Take target-word token -> image patches attention, average over all
     layers and heads.
  3) Reshape to (h_patches, w_patches), upsample to the original image size,
     overlay, and save.

Usage:
  python attention_visualization_simple.py \
      --image /path/to/image.png \
      --prompt "Where is the helmet?" \
      --target-word helmet \
      --ckpt /path/to/tuna_2_pixel_7b.pt
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

# Ensure the tuna-2 repo root is on sys.path so `tuna.*` imports work
_repo_root = str(pathlib.Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402


# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------
def resize_center_crop(img, size, interpolation=InterpolationMode.BILINEAR):
    target_h, target_w = int(size[0]), int(size[1])
    orig_w, orig_h = img.size
    target_ratio = target_h / target_w
    orig_ratio = orig_h / orig_w
    if math.isclose(target_ratio, orig_ratio, rel_tol=0.0, abs_tol=1e-12):
        return TF.resize(
            img, [target_h, target_w], interpolation=interpolation, antialias=True
        )
    scale = target_h / orig_h if target_ratio > orig_ratio else target_w / orig_w
    new_w = max(int(math.ceil(orig_w * scale)), target_w)
    new_h = max(int(math.ceil(orig_h * scale)), target_h)
    resized = TF.resize(
        img, [new_h, new_w], interpolation=interpolation, antialias=True
    )
    left = min(max(0, (new_w - target_w) // 2), new_w - target_w)
    top = min(max(0, (new_h - target_h) // 2), new_h - target_h)
    return TF.crop(resized, top=top, left=left, height=target_h, width=target_w)


def image_transform(image, resolution=(256, 256)):
    image = image.convert("RGB")
    image = resize_center_crop(image, resolution)
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True
    )(tensor)
    return tensor


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/model/tuna_2_pixel_7b.yaml",
        help="Tuna-2 pixel (variant C / none-encoder) config yaml.",
    )
    parser.add_argument("--ckpt", required=True, help="Path to tuna_2_pixel ckpt")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-word", required=True)
    # tuna_2_pixel default training resolution is 256
    # (image_latent_height/width=16, patch_size=16).
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="/tmp/attn_simple.png")
    args = parser.parse_args()

    DEVICE = args.device
    DTYPE = torch.bfloat16
    HEIGHT, WIDTH = args.height, args.width

    # ---- Resolve config path ----
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_path = os.path.join(_repo_root, config_path)

    # ---- Build model ----
    from tuna.models.tuna_2_pixel import Tuna2PixelModel

    config = OmegaConf.load(config_path)
    # Hydra `_target_` is not a constructor arg -- strip it before splatting.
    if "_target_" in config:
        OmegaConf.set_struct(config, False)
        config.pop("_target_")
    config.load_stage1_model = args.ckpt
    config.gradient_checkpointing = False

    print("Building model...")
    model = Tuna2PixelModel(**config)
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()

    # ---- Tokenizer + hyper params ----
    from tuna.pipelines.tuna_2_pixel_pipeline import get_hyper_params

    tokenizer = model.text_tokenizer
    tuna_token_ids = model.tuna_token_ids

    (
        num_image_tokens,
        _num_video_tokens,
        _max_seq_len,
        _max_text_len,
        _image_latent_dim,
        _patch_size,
        latent_width,
        latent_height,
        pad_id,
        _bos_id,
        _eos_id,
        boi_id,
        eoi_id,
        _bov_id,
        _eov_id,
        img_pad_id,
        _vid_pad_id,
        _guidance_scale,
    ) = get_hyper_params(
        tokenizer,
        tuna_token_ids,
        use_chat_template=False,
        add_aspect_ratio_embeds=False,
        height=HEIGHT,
        width=WIDTH,
        generation_mode="mmu",
        latent_frames=1,
    )

    # ---- Load and preprocess image ----
    pil_image = Image.open(args.image).convert("RGB")
    pixel_values = image_transform(pil_image, resolution=(HEIGHT, WIDTH))
    pixel_values = pixel_values.unsqueeze(0).to(DEVICE, DTYPE)

    # ---- Build the MMU input sequence ----
    conversation = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": "<image>\n" + args.prompt},
    ]
    conv_prompt = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    tokens_list = tokenizer(conv_prompt, add_special_tokens=False).input_ids

    img_id = tokenizer("<image>", add_special_tokens=False).input_ids[0]
    img_idx = tokens_list.index(img_id)
    tokens_list = (
        tokens_list[:img_idx]
        + [boi_id]
        + [img_pad_id] * num_image_tokens
        + [eoi_id]
        + tokens_list[img_idx + 1 :]
    )

    text_tokens = torch.tensor(tokens_list).unsqueeze(0).to(DEVICE)
    modality_positions = (
        torch.tensor([[img_idx + 1, num_image_tokens]]).unsqueeze(0).to(DEVICE)
    )

    img_offset = img_idx + 1
    img_patch_start = img_offset + 1  # skip time embed
    img_patch_end = img_offset + num_image_tokens

    # ---- Prepare latents / mask / embeds ----
    from tuna.models.omni_attention import omni_attn_mask_naive

    text_masks = torch.where(
        (text_tokens != img_pad_id) & (text_tokens != pad_id),
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens),
    )
    image_masks = torch.where(
        text_tokens == img_pad_id,
        torch.ones_like(text_tokens),
        torch.zeros_like(text_tokens),
    ).to(DEVICE)

    image_latents, t, image_labels, image_masks_out, _ = (
        model.prepare_latents_and_labels(pixel_values, ["mmu"], image_masks)
    )

    attention_mask = omni_attn_mask_naive(
        text_tokens.size(0), text_tokens.size(1), modality_positions, DEVICE
    ).to(DTYPE)

    with torch.no_grad():
        input_embeds = model.tuna_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(DTYPE),
            attention_mask=attention_mask,
            text_masks=text_masks,
            image_masks=image_masks_out,
            text_labels=None,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=DEVICE,
            return_input_embeds=True,
        )

        # ---- Forward to get attention ----
        # tuna-2 LLM backbone: model.tuna_model.tuna is a Qwen2ForCausalLM,
        # .tuna.model is the inner Qwen2Model (skips lm_head, saves memory).
        outputs = model.tuna_model.tuna.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
        )

    # outputs.attentions: tuple of (1, H, L, L), one per layer.
    # Average over all layers and heads -> shape (L, L).
    attn = torch.stack([a[0].float() for a in outputs.attentions]).mean(dim=(0, 1))
    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Locate the target-word token position ----
    skip_ids = {img_pad_id, boi_id, eoi_id, pad_id}
    target_pos = None
    for i, tid in enumerate(tokens_list):
        if tid in skip_ids:
            continue
        if args.target_word.lower() in tokenizer.decode([tid]).lower():
            target_pos = i
            break
    if target_pos is None:
        raise ValueError(f"Target word '{args.target_word}' not found in prompt tokens")
    print(f"Target word '{args.target_word}' -> token pos {target_pos}")

    # ---- target_pos -> image patches attention ----
    attn_1d = attn[target_pos, img_patch_start:img_patch_end].cpu().numpy()
    attn_2d = attn_1d.reshape(latent_height, latent_width)

    # Normalize to [0, 1]
    vmin, vmax = attn_2d.min(), attn_2d.max()
    if vmax > vmin:
        attn_2d = (attn_2d - vmin) / (vmax - vmin)

    # Upsample to the original image size
    heat_pil = Image.fromarray((attn_2d * 255).astype(np.uint8)).resize(
        (WIDTH, HEIGHT), Image.BILINEAR
    )
    heat = np.array(heat_pil) / 255.0

    # ---- Plot ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(pil_image.resize((WIDTH, HEIGHT)))
    ax.imshow(heat, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f"Attention: '{args.target_word}'  (avg all layers/heads)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
