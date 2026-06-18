# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pixel-space AR video generation pipeline for Tuna-2 + Gemma.

Chunk-wise autoregressive: each chunk of K frames is denoised via diffusion
(bidirectional spatial attention), conditioned on previously generated clean
frames. The cross-frame causal mask ensures frame t+1 sees frame t but not
the reverse.

Generation contract (per chunk step):
  * `image_latents` shape `[B, 3, T_so_far, H, W]`:
      past frames are clean RGB, current chunk frames are the noisy state.
  * `t` shape `[B*T_so_far]`:
      past frames carry t=1.0 (clean signal in Tuna's convention);
      current chunk frames carry the current diffusion timestep.
  * `modality_positions` shape `[B, T_so_far, 2]`:
      one span per frame so the cross-frame causal mask applies.

Model returns clean-image predictions `x0_pred` of shape
`[num_imgs, C, T, H, W]`. We extract the CURRENT chunk's prediction, convert
to velocity via `v = (x0 - z) / max(1 - t, eps)`, and Euler-step the chunk's
noisy state forward in t (toward t=1, the clean end of the rectified flow).

Notes on the current implementation:
  * Past-prefix attention is recomputed every chunk (no KV cache yet); this
    is correct but slower than Self-Forcing's KV-cache approach. Real KV
    cache requires plumbing `past_key_values` through `Tuna2PixelGemma.forward`,
    which is a follow-up.
  * CFG noise is SHARED between conditional and unconditional branches so the
    guidance combines comparable trajectories.
  * Negative prompts are wired through.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from tuna.pipelines._common import denorm
from tuna.pipelines._pipeline_base import TunaPipelineBase

logger = logging.getLogger(__name__)


class Tuna2PixelARVideoPipeline(TunaPipelineBase):
    """Chunk-wise AR video pipeline in pure pixel space."""

    def __init__(
        self,
        model,
        text_tokenizer=None,
        tuna_token_ids=None,
        config=None,
        weight_dtype=torch.bfloat16,
        device: str = "cuda",
        use_tf32: bool = True,
        use_chat_template: bool = False,
        height: int = 512,
        width: int = 512,
        num_frames: int = 16,
        frames_per_chunk: int = 4,
        num_diffusion_steps_per_chunk: int = 8,
        patch_size: int = 16,
        add_aspect_ratio_embeds: bool = False,
        max_seq_len: int = 8192,
        **kwargs,  # silently absorb other generic pipeline kwargs
    ):
        self.model = model
        self.text_tokenizer = text_tokenizer
        self.tuna_token_ids = tuna_token_ids
        self.config = config
        self.device = device
        self.weight_dtype = weight_dtype
        self.use_chat_template = use_chat_template
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds

        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.frames_per_chunk = frames_per_chunk
        self.num_diffusion_steps_per_chunk = num_diffusion_steps_per_chunk
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        if use_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True

        self.h_patches = height // patch_size
        self.w_patches = width // patch_size
        self.n_meta = 3 if add_aspect_ratio_embeds else 1
        self.num_tokens_per_frame = self.h_patches * self.w_patches + self.n_meta

        # Token ids
        self.pad_id = text_tokenizer.pad_token_id
        self.bos_id = tuna_token_ids["bos_id"]
        self.eos_id = tuna_token_ids["eos_id"]
        self.boi_id = tuna_token_ids["boi_id"]
        self.eoi_id = tuna_token_ids["eoi_id"]
        self.img_pad_id = tuna_token_ids["img_pad_id"]

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    @torch.no_grad()
    def t2v_ar(
        self,
        prompt: Union[str, List[str]],
        guidance_scale: float = 4.0,
        noise_scale: float = 2.0,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
        eps_t: float = 5e-2,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate video frames chunk by chunk."""
        torch.manual_seed(seed)

        if isinstance(prompt, str):
            prompt = [prompt]
        assert len(prompt) == 1, "AR video pipeline currently supports batch_size=1"

        n_chunks = self.num_frames // self.frames_per_chunk
        all_clean_frames: List[torch.Tensor] = []  # list of [3, H, W]

        # Per-frame timestep schedule (within a chunk).
        timesteps = torch.linspace(
            0.0,
            1.0,
            self.num_diffusion_steps_per_chunk + 1,
            device=self.device,
            dtype=self.weight_dtype,
        )

        for chunk_idx in range(n_chunks):
            logger.info(
                f"[AR-video] chunk {chunk_idx + 1}/{n_chunks} "
                f"(frames so far: {len(all_clean_frames)})"
            )

            num_past = len(all_clean_frames)
            num_cur = self.frames_per_chunk
            num_total = num_past + num_cur

            # 1) Build text+visual layout for THIS chunk (cond + uncond stacked).
            text_tokens_cfg, modality_positions_cfg = self._build_cfg_inputs(
                prompt[0],
                num_past=num_past,
                num_cur=num_cur,
                negative_prompt=negative_prompt or "",
            )

            # 2) Initialize current-chunk noise (SHARED across cond/uncond).
            cur_noise = noise_scale * torch.randn(
                (1, 3, num_cur, self.height, self.width),
                device=self.device,
                dtype=self.weight_dtype,
                generator=None,
            )

            # 3) Stack past clean + current noisy for the model's image_latents.
            if num_past > 0:
                past_stack = torch.stack(all_clean_frames, dim=0)  # [num_past, 3, H, W]
                past_stack = past_stack.to(self.device, self.weight_dtype).permute(1, 0, 2, 3)
                past_5d = past_stack.unsqueeze(0)  # [1, 3, num_past, H, W]
            else:
                past_5d = torch.zeros(
                    (1, 3, 0, self.height, self.width),
                    device=self.device,
                    dtype=self.weight_dtype,
                )

            # CFG duplicates: cond + uncond batches share the same noise / past.
            past_5d_cfg = torch.cat([past_5d, past_5d], dim=0)

            # 4) Diffusion sub-steps within the chunk
            cur_state = cur_noise.clone()
            for step_idx in range(self.num_diffusion_steps_per_chunk):
                t_cur = timesteps[step_idx]
                t_next = timesteps[step_idx + 1]

                # Build image_latents = [past_clean | cur_state] for each branch.
                cur_state_cfg = torch.cat([cur_state, cur_state], dim=0)
                image_latents = torch.cat([past_5d_cfg, cur_state_cfg], dim=2)
                # → [2, 3, num_total, H, W]

                # Build per-frame t: past=1.0, current=t_cur. Layout matches
                # _patchify_5d(reshape_frame_to_batch_dim=True) → [B*T, ...].
                t_per_frame = torch.cat(
                    [
                        torch.full(
                            (2 * num_past,),
                            1.0,
                            device=self.device,
                            dtype=self.weight_dtype,
                        ),
                        torch.full(
                            (2 * num_cur,),
                            t_cur.item(),
                            device=self.device,
                            dtype=self.weight_dtype,
                        ),
                    ],
                    dim=0,
                )
                # Re-arrange to interleave per-batch: [past, cur, past, cur, ...]
                # Actually _patchify_5d(reshape_frame_to_batch_dim=True) folds T
                # into batch dim like rearrange(B, C, T, ...) → (B*T, C, ...).
                # That means for B=2 batches, order is (b=0 frames, b=1 frames).
                t_per_frame = (
                    torch.cat([
                        torch.full((num_past,), 1.0, device=self.device, dtype=self.weight_dtype),
                        torch.full((num_cur,), t_cur.item(), device=self.device, dtype=self.weight_dtype),
                    ])
                    .unsqueeze(0)
                    .expand(2, -1)
                    .reshape(-1)
                )

                # Build attention mask (cross-frame causal enabled).
                attn_mask, diff_attn_mask = self._build_ar_attention_mask(
                    text_tokens_cfg, modality_positions_cfg
                )

                # Forward
                forward_out = self.model.tuna_model(
                    text_tokens=text_tokens_cfg,
                    image_latents=image_latents,
                    t=t_per_frame,
                    attention_mask=attn_mask,
                    diffhead_attention_mask=diff_attn_mask,
                    modality_positions=modality_positions_cfg,
                    output_hidden_states=False,
                    max_seq_len=text_tokens_cfg.size(1),
                    device=self.device,
                )
                # Inference branch returns (logits, x0_pred) where x0_pred is
                # [num_imgs, C, T, H, W]. num_imgs = B * T_per_batch.
                _, x0_pred = forward_out

                # x0_pred shape: [2*num_total, C, 1, H, W] or similar — reshape
                # to [2, C, num_total, H, W] for easy chunk slicing.
                x0_pred = x0_pred.reshape(2, num_total, *x0_pred.shape[-3:])
                # → [2, num_total, C, H, W]; swap to [2, C, num_total, H, W]
                x0_pred = x0_pred.permute(0, 2, 1, 3, 4)

                # Slice CURRENT-chunk frames only
                x0_cur = x0_pred[:, :, num_past:, :, :]  # [2, C, num_cur, H, W]

                # CFG guidance on x0 (equivalent to guidance on v at fixed t).
                if guidance_scale > 0:
                    x0_cond, x0_uncond = torch.chunk(x0_cur, 2, dim=0)
                    x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
                else:
                    x0_guided = x0_cur[:1]

                # x0 → v conversion (Tuna JiT convention: t=0 noise, t=1 clean)
                one_minus_t = max(1.0 - t_cur.item(), eps_t)
                v_pred = (x0_guided - cur_state) / one_minus_t

                # Euler step: integrate from t_cur toward t_next (advancing t)
                dt = (t_next - t_cur).item()
                cur_state = cur_state + dt * v_pred

            # 5) Commit clean chunk frames
            for f_idx in range(num_cur):
                all_clean_frames.append(cur_state[0, :, f_idx].cpu())

            # 6) Truncation guard
            est_seq_len = self._estimate_seq_len(num_total + num_cur, prompt[0])
            if est_seq_len > self.max_seq_len * 0.9:
                logger.warning(
                    f"[AR-video] sequence approaching max_seq_len "
                    f"({est_seq_len}/{self.max_seq_len}) — stopping early."
                )
                break

        return self._frames_to_pil(all_clean_frames)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_text_layout(
        self, prompt: str, num_frames_in_seq: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct (text_tokens, modality_positions) for a single sample."""
        text_ids = self.text_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        total_visual = self.num_tokens_per_frame * num_frames_in_seq

        tokens = (
            [self.bos_id]
            + text_ids
            + [self.boi_id]
            + [self.img_pad_id] * total_visual
            + [self.eoi_id]
            + [self.eos_id]
        )
        tokens_t = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        first_frame_offset = 1 + len(text_ids) + 1
        spans = []
        for f_idx in range(num_frames_in_seq):
            offset = first_frame_offset + f_idx * self.num_tokens_per_frame
            spans.append([offset, self.num_tokens_per_frame])
        spans_t = torch.tensor(spans, dtype=torch.long, device=self.device).unsqueeze(0)

        return tokens_t, spans_t

    def _build_cfg_inputs(
        self,
        prompt: str,
        num_past: int,
        num_cur: int,
        negative_prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a (cond, uncond) batch with matching modality_positions."""
        num_total = num_past + num_cur

        tokens_cond, spans_cond = self._build_text_layout(prompt, num_total)
        tokens_uncond, spans_uncond = self._build_text_layout(negative_prompt, num_total)

        # Pad to the same length (in case prompt lengths differ).
        L_cond = tokens_cond.shape[1]
        L_uncond = tokens_uncond.shape[1]
        max_L = max(L_cond, L_uncond)

        def _pad(t: torch.Tensor) -> torch.Tensor:
            if t.shape[1] >= max_L:
                return t[:, :max_L]
            pad = torch.full(
                (1, max_L - t.shape[1]),
                self.pad_id,
                dtype=t.dtype,
                device=t.device,
            )
            return torch.cat([t, pad], dim=1)

        tokens_cond = _pad(tokens_cond)
        tokens_uncond = _pad(tokens_uncond)

        tokens_cfg = torch.cat([tokens_cond, tokens_uncond], dim=0)
        spans_cfg = torch.cat([spans_cond, spans_uncond], dim=0)
        return tokens_cfg, spans_cfg

    def _build_ar_attention_mask(self, text_tokens, modality_positions):
        """Build attention masks with cross-frame causal enabled."""
        seq_len = text_tokens.size(1)
        device = text_tokens.device

        if self.model.attention_backend == "sdpa":
            from tuna.models.gemma_omni_attn import build_gemma_omni_attn_mask_naive

            mask = build_gemma_omni_attn_mask_naive(
                modality_positions=modality_positions,
                seq_len=seq_len,
                sliding_window=self.model.sliding_window,
                is_local_layer=False,           # full-causal base (correct union)
                cross_frame_causal=True,        # KEY DIFFERENCE vs image pipeline
                device=device,
                dtype=self.weight_dtype,
                inverted=True,
            )
            return mask, None
        else:
            from tuna.models.gemma_omni_attn import build_gemma_omni_block_mask

            block_mask = build_gemma_omni_block_mask(
                modality_positions=modality_positions,
                seq_len=seq_len,
                layer_idx=0,
                layer_pattern=["global"],
                sliding_window=self.model.sliding_window,
                num_heads=self.model.num_attention_heads,
                cross_frame_causal=True,
                device=device,
            )
            return block_mask, block_mask

    def _estimate_seq_len(self, num_frames_in_seq: int, prompt: str) -> int:
        text_ids = self.text_tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return (
            1  # BOS
            + len(text_ids)
            + 1  # BOI
            + num_frames_in_seq * self.num_tokens_per_frame
            + 1  # EOI
            + 1  # EOS
        )

    def _frames_to_pil(self, frames: List[torch.Tensor]) -> List[Image.Image]:
        out = []
        for f in frames:
            img = denorm(f.unsqueeze(0))[0]
            out.append(Image.fromarray(img))
        return out
