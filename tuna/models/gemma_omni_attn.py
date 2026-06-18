# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Attention-mask utilities to merge Gemma 4's interleaved sliding+global
attention with Tuna's omni image-span rule.

Gemma 4 12B uses interleaved local (sliding-window) + global attention layers.
A naive sliding window would break Tuna's invariant that "patches within the
same image span see each other bidirectionally", because high-resolution image
spans can exceed the sliding window (1024 tokens).

This module provides:

  * `build_image_span_id(modality_positions, seq_len)` — per-position int label,
    0 for non-image tokens, 1..N for image span index.
  * `gemma_omni_mask_mod_factory(...)` — builds a flex_attention `mask_mod` that
    OR-merges Gemma's base rule (sliding/global causal) with Tuna's same-span
    bidirectional override and an optional cross-frame causal rule (video).
  * `build_gemma_omni_attn_mask_naive(...)` — equivalent dense mask for sdpa.

The same-frame bidirectional rule is the key to preserving image generation
quality under sliding-window LLMs. The cross-frame causal rule is the
extension that enables temporal AR video generation; for T=1 (single image)
it is vacuous and degenerates to the original Tuna mask.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch


def build_image_span_id(
    modality_positions: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Compute span_id of shape [B, L].

    Args:
        modality_positions: [B, N, 2] with (offset, length) per image span,
            -1 padded for absent spans.
        seq_len: total sequence length L.

    Returns:
        span_id[b, p] = 0 if position p is non-image, else 1..N indicating
        which image span it belongs to.
    """
    B, N, _ = modality_positions.shape
    device = modality_positions.device

    offs = modality_positions[..., 0].to(torch.int64)  # [B, N]
    lens = modality_positions[..., 1].to(torch.int64)  # [B, N]
    valid = offs >= 0  # [B, N]

    t = torch.arange(seq_len, device=device, dtype=torch.int64)
    t = t[None, None, :]  # [1, 1, L]

    spans = (
        (t >= offs[..., None])
        & (t < (offs + lens)[..., None])
        & valid[..., None]
    )  # [B, N, L]

    labels = torch.arange(1, N + 1, device=device, dtype=torch.int64)
    labels = labels[None, :, None]  # [1, N, 1]

    span_id = (spans.to(torch.int64) * labels).max(dim=1).values  # [B, L]
    return span_id.to(torch.int32)


def gemma_omni_mask_mod_factory(
    span_id: torch.Tensor,
    sliding_window: int = 1024,
    is_local_layer: bool = True,
    cross_frame_causal: bool = False,
) -> Callable:
    """Build a flex_attention mask_mod combining Gemma base + Tuna omni rules.

    Args:
        span_id: [B, L] int32 from `build_image_span_id`.
        sliding_window: Gemma's local-layer window size (e.g., 1024).
        is_local_layer: True for Gemma local layers; False for global layers.
        cross_frame_causal: If True, frames with iq > ik are allowed (video AR);
            this is a no-op when each batch has at most one image span (T=1).
    """

    def mask_mod(b, h, q, k):
        # 1) Gemma base rule
        if is_local_layer:
            base = (q >= k) & ((q - k) < sliding_window)
        else:
            base = q >= k

        # 2) Same-image-span bidirectional override
        iq = span_id[b, q.to(torch.long)]
        ik = span_id[b, k.to(torch.long)]
        same_span = (iq != 0) & (iq == ik)

        # 3) Optional cross-frame causal (video AR)
        if cross_frame_causal:
            cross_causal = (iq > ik) & (iq != 0) & (ik != 0)
            return base | same_span | cross_causal
        return base | same_span

    return mask_mod


def build_gemma_omni_block_mask(
    modality_positions: torch.Tensor,
    seq_len: int,
    layer_idx: int,
    layer_pattern: list,
    sliding_window: int = 1024,
    num_heads: int = 1,
    cross_frame_causal: bool = False,
    device: Optional[torch.device] = None,
    block_size: int = 128,
):
    """Convenience builder for flex_attention BlockMask given a layer pattern.

    Args:
        layer_pattern: e.g. ["local"]*5 + ["global"], repeated.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    if device is None:
        device = modality_positions.device

    span_id = build_image_span_id(modality_positions, seq_len).to(device)
    is_local = layer_pattern[layer_idx % len(layer_pattern)] == "local"
    mask_mod = gemma_omni_mask_mod_factory(
        span_id,
        sliding_window=sliding_window,
        is_local_layer=is_local,
        cross_frame_causal=cross_frame_causal,
    )
    B = modality_positions.shape[0]
    return create_block_mask(
        mask_mod,
        B=B,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=block_size,
    )


def build_gemma_omni_attn_mask_naive(
    modality_positions: torch.Tensor,
    seq_len: int,
    sliding_window: int = 1024,
    is_local_layer: bool = True,
    cross_frame_causal: bool = False,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    inverted: bool = True,
) -> torch.Tensor:
    """Dense [B, 1, L, L] additive mask for sdpa backend.

    Returns an additive mask: 0 where allowed, -inf where blocked (when
    inverted=True, the conventional sdpa attention-mask convention).
    """
    if device is None:
        device = modality_positions.device

    B = modality_positions.shape[0]
    L = seq_len
    q = torch.arange(L, device=device)[:, None]
    k = torch.arange(L, device=device)[None, :]

    # Base
    if is_local_layer:
        base = (q >= k) & ((q - k) < sliding_window)
    else:
        base = q >= k
    base = base[None].expand(B, L, L).clone()  # [B, L, L]

    # Omni overrides per-batch
    span_id = build_image_span_id(modality_positions, L).to(device)
    for b in range(B):
        # Same-span bidirectional
        unique_spans = span_id[b].unique()
        for s in unique_spans.tolist():
            if s == 0:
                continue
            pos = (span_id[b] == s).nonzero(as_tuple=True)[0]
            base[b, pos[:, None], pos[None, :]] = True

        # Cross-frame causal — base tril already gives later-sees-earlier.
        # For LOCAL layers, however, sliding can cut cross-frame visibility
        # if frames are far apart. Lift this when cross_frame_causal=True.
        if cross_frame_causal and is_local_layer:
            for s_q in unique_spans.tolist():
                if s_q == 0:
                    continue
                for s_k in unique_spans.tolist():
                    if s_k == 0 or s_k >= s_q:
                        continue
                    qpos = (span_id[b] == s_q).nonzero(as_tuple=True)[0]
                    kpos = (span_id[b] == s_k).nonzero(as_tuple=True)[0]
                    base[b, qpos[:, None], kpos[None, :]] = True

    mask = base[:, None]  # [B, 1, L, L]

    if inverted:
        # 0 where allowed, -inf where blocked (sdpa convention)
        additive = torch.zeros_like(mask, dtype=dtype)
        additive = additive.masked_fill(~mask, float("-inf"))
        return additive
    return mask


# ---------------------------------------------------------------------------
# Monkey-patch helper: install omni mask into Gemma 4 HF model
# ---------------------------------------------------------------------------

def install_omni_mask_on_gemma(
    gemma_model,
    sliding_window: int = 1024,
    cross_frame_causal: bool = False,
):
    """Light-touch monkey-patch: stores the omni-mask config on the model.

    The actual mask wiring happens at the wrapper level (Tuna2PixelGemma) where
    we have access to modality_positions. This function just sets attributes
    so the wrapper's `create_attention_mask` knows what to build.
    """
    gemma_model._omni_sliding_window = sliding_window
    gemma_model._omni_cross_frame_causal = cross_frame_causal
    return gemma_model
