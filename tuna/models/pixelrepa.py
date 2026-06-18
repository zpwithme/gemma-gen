# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PixelREPA — Masked Transformer Adapter for representation alignment.

Reference: "Representation Alignment for Just Image Transformers is not Easier
than You Think" (PixelREPA, arXiv:2603.14366).

Naive REPA — directly aligning a pixel-space diffusion's hidden state to a
compressed semantic encoder (DINOv2 / SigLIP) via per-token MLP — can collapse
diversity due to information asymmetry between high-dim pixel denoising and
low-dim semantic targets.

PixelREPA's fix:
  * Replace per-token MLP with a shallow Transformer Adapter (MTA), which
    allows neighboring-token context and breaks the per-token shortcut.
  * Apply partial masking to the input of MTA, forcing the adapter to predict
    semantic targets from surrounding context rather than pointwise regression.

This module is TRAINING-ONLY: it computes an auxiliary loss; the main
denoising path and inference are unaffected. The module can be toggled on/off
via a single flag, supporting clean ablation experiments.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PixelREPAModule(nn.Module):
    """MTA + frozen semantic target encoder.

    Args:
        llm_hidden_size: hidden dim of the main LMM backbone (e.g., 3072 for
            Gemma 4 12B, 3584 for Qwen2.5-7B).
        target_model_id: HF model id for the frozen semantic encoder.
            Default 'facebook/dinov2-large' (1024-dim).
        target_hidden_size: dim of target encoder. Should match the encoder.
        adapter_depth: number of transformer blocks in MTA (PixelREPA uses 2).
        adapter_heads: attention heads in MTA.
        mask_ratio: fraction of tokens replaced by the mask token before MTA.
        loss_weight: scalar multiplier applied inside this module so the caller
            can just add the returned loss directly.
        from_layer: which LMM hidden layer to hook (negative = from the end,
            e.g. -8 for "8 layers before the last").
    """

    def __init__(
        self,
        llm_hidden_size: int,
        target_model_id: str = "facebook/dinov2-large",
        target_hidden_size: int = 1024,
        adapter_depth: int = 2,
        adapter_heads: int = 8,
        mask_ratio: float = 0.5,
        loss_weight: float = 0.5,
        from_layer: int = -8,
        target_image_size: int = 224,
    ) -> None:
        super().__init__()
        self.llm_hidden_size = llm_hidden_size
        self.target_hidden_size = target_hidden_size
        self.mask_ratio = mask_ratio
        self.loss_weight = loss_weight
        self.from_layer = from_layer
        self.target_image_size = target_image_size

        # Learnable mask token (MAE / DeTok style)
        scale = llm_hidden_size ** -0.5
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, llm_hidden_size))

        # 2-block transformer adapter
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_hidden_size,
            nhead=adapter_heads,
            dim_feedforward=llm_hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=adapter_depth)

        # Projection to target encoder dim
        self.proj = nn.Linear(llm_hidden_size, target_hidden_size)

        # Frozen target encoder (DINOv2 by default)
        self.target_model_id = target_model_id
        self._target = None  # lazy-loaded in encode_target
        self._target_loaded = False

        # ImageNet normalization for DINOv2 input
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _lazy_load_target(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._target_loaded:
            return
        from transformers import AutoModel

        logger.info(f"[PixelREPA] Loading frozen target encoder: {self.target_model_id}")
        target = AutoModel.from_pretrained(self.target_model_id)
        for p in target.parameters():
            p.requires_grad = False
        target.eval()
        target = target.to(device=device, dtype=dtype)
        # Stored without registering as a submodule so it stays out of state_dict.
        object.__setattr__(self, "_target", target)
        self._target_loaded = True

    @torch.no_grad()
    def encode_target(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run frozen DINOv2 on clean pixels, return patch tokens (drop CLS).

        Args:
            pixel_values: [B, 3, H, W] in [-1, 1] (Tuna convention).

        Returns:
            [B, N_dino, target_hidden_size]
        """
        self._lazy_load_target(pixel_values.device, pixel_values.dtype)

        # [-1, 1] -> [0, 1] -> ImageNet normalize
        x = (pixel_values + 1.0) / 2.0
        x = (x - self._imagenet_mean.to(x.dtype)) / self._imagenet_std.to(x.dtype)

        # DINOv2 typically wants a fixed input size (e.g., 224 or 518).
        H = x.shape[-2]
        W = x.shape[-1]
        if H != self.target_image_size or W != self.target_image_size:
            x = F.interpolate(
                x,
                size=(self.target_image_size, self.target_image_size),
                mode="bilinear",
                align_corners=False,
            )

        out = self._target(pixel_values=x)
        # AutoModel for ViT-family returns BaseModelOutputWithPooling
        feats = out.last_hidden_state[:, 1:]  # drop CLS
        return feats

    def forward(
        self,
        llm_hidden: torch.Tensor,
        image_positions: torch.Tensor,
        clean_pixel_values: torch.Tensor,
        meta_token_count: int = 0,
    ) -> torch.Tensor:
        """Compute PixelREPA loss across all valid image spans in the batch.

        Args:
            llm_hidden: [B, L, D]
            image_positions: [B, N_imgs, 2] — (offset, length) per span (or -1 padded).
            clean_pixel_values: clean image source for the frozen target encoder.
                Either [B, 3, H, W] (one image per batch row) or [B*N, 3, H, W]
                (one image per (batch, span) pair, used when the wrapper has
                rearranged interleaved data).
            meta_token_count: number of leading meta tokens within each span
                (height/width/time embeddings) that should be SKIPPED so the
                MTA only sees actual image patches. Passed from the caller
                based on add_time_embeds + add_aspect_ratio_embeds.

        Returns:
            scalar loss (already multiplied by loss_weight).
        """
        B, _, D = llm_hidden.shape
        N_imgs = image_positions.shape[1]
        device = llm_hidden.device
        dtype = llm_hidden.dtype

        # Did the caller flatten (b, n) → b*n in clean_pixel_values?
        flat_pixels = clean_pixel_values.shape[0] == B * N_imgs

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        count = 0

        for b in range(B):
            for j in range(N_imgs):
                offset = int(image_positions[b, j, 0].item())
                length = int(image_positions[b, j, 1].item())
                if offset < 0 or length <= meta_token_count:
                    continue

                # Skip Tuna's leading meta tokens within the span.
                patch_start = offset + meta_token_count
                patch_end = offset + length
                patches = llm_hidden[b, patch_start:patch_end]  # [N_patch, D]
                N_patch = patches.shape[0]
                if N_patch <= 0:
                    continue

                # Infer 2D patch grid (assume square unless N_patch is rectangular
                # for one of Tuna's standard aspect ratios).
                hp, wp = self._infer_patch_grid(N_patch)
                if hp is None:
                    continue  # unrecognized layout — skip rather than crash

                # Select the corresponding source pixel image.
                pix_idx = (b * N_imgs + j) if flat_pixels else b
                if pix_idx >= clean_pixel_values.shape[0]:
                    continue
                pixels = clean_pixel_values[pix_idx : pix_idx + 1]  # [1, 3, H, W]

                # Random masking (50% by default).
                num_mask = int(N_patch * self.mask_ratio)
                masked = patches.clone().unsqueeze(0)  # [1, N_patch, D]
                if num_mask > 0:
                    idx = torch.randperm(N_patch, device=device)[:num_mask]
                    masked[0, idx] = self.mask_token.to(masked.dtype)

                # MTA + projection → [1, N_patch, target_dim]
                adapted = self.adapter(masked)
                predicted = self.proj(adapted)

                # Target features (frozen) → [1, N_dino, target_dim]
                target_feat = self.encode_target(pixels).to(predicted.dtype)

                # 2D align: reshape to grid → bilinear interp → flatten.
                target_aligned = self._align_grid(target_feat, hp, wp)

                # Cosine-similarity loss
                pred_norm = F.normalize(predicted, dim=-1)
                targ_norm = F.normalize(target_aligned, dim=-1)
                cos_sim = (pred_norm * targ_norm).sum(dim=-1)  # [1, N_patch]
                total_loss = total_loss + (1.0 - cos_sim).mean()
                count += 1

        if count == 0:
            # No valid image spans — return a graph-connected zero so FSDP
            # doesn't complain about unused parameters on text-only steps.
            zero = (
                self.mask_token.sum() * 0.0
                + sum(p.sum() * 0.0 for p in self.proj.parameters())
            ).to(dtype)
            return zero

        return (total_loss / count) * self.loss_weight

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    _STANDARD_GRIDS: tuple = (
        # (h_patches * w_patches, h_patches, w_patches) — common Tuna buckets.
        (16 * 16,  16, 16),
        (32 * 32,  32, 32),
        (64 * 64,  64, 64),
        (28 * 36,  28, 36),
        (36 * 28,  36, 28),
        (24 * 42,  24, 42),
        (42 * 24,  42, 24),
        (56 * 72,  56, 72),
        (72 * 56,  72, 56),
        (48 * 84,  48, 84),
        (84 * 48,  84, 48),
    )

    def _infer_patch_grid(self, n_patch: int):
        for total, h, w in self._STANDARD_GRIDS:
            if total == n_patch:
                return h, w
        # Fallback: square-root for arbitrary square count
        import math

        side = int(math.isqrt(n_patch))
        if side * side == n_patch:
            return side, side
        return None, None

    def _align_grid(
        self,
        target_feat: torch.Tensor,
        hp: int,
        wp: int,
    ) -> torch.Tensor:
        """Bilinear-resize DINOv2 feature grid to match the LMM patch grid.

        target_feat: [1, N_dino, D], where N_dino is a square count for
        DINOv2 input at fixed resolution. Reshape to 2D, bilinear to (hp, wp),
        flatten back.
        """
        import math

        N_dino = target_feat.shape[1]
        side = int(math.isqrt(N_dino))
        if side * side != N_dino:
            # DINOv2 returned non-square — fall back to 1D linear interp.
            t = target_feat.transpose(1, 2)
            t = F.interpolate(t, size=hp * wp, mode="linear", align_corners=False)
            return t.transpose(1, 2)

        # [1, N, D] → [1, D, side, side]
        t = target_feat.transpose(1, 2).reshape(1, -1, side, side)
        # → [1, D, hp, wp]
        t = F.interpolate(t, size=(hp, wp), mode="bilinear", align_corners=False)
        # → [1, hp*wp, D]
        return t.flatten(2).transpose(1, 2)
