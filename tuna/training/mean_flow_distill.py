# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mean Flow distillation for compressing flow matching to few-step inference.

Reference: MiniT2I `mean_flow_distill` branch (Wang, He et al., 2026).

Idea:
  A many-step flow-matching teacher is distilled into a few-step student.
  In each interval [t_start, t_end], the teacher takes K Euler sub-steps to
  produce a ground-truth "mean velocity" target; the student predicts that
  mean velocity in ONE forward pass given the chunk's start state and a
  representative timestep (t_avg).

Important Tuna integration notes:
  * Tuna's diffusion head emits x0 predictions (clean-image prediction), not
    velocity, even though the config says `prediction='velocity'`. This is
    a JiT-style design — see `tuna/models/jit_utils.py` for the canonical
    conversion `v = (x0_pred - z_t) / max(1 - t, 5e-2)`.
    Both teacher and student forwards therefore go through the same x0→v
    conversion before Euler stepping or loss computation.
  * Attention masks MUST be built via the wrapper's `create_attention_mask`
    so Tuna's omni span rule (image patches bidirectional within a span)
    is preserved. Passing `attention_mask=None` to Gemma falls back to plain
    causal which violates the unified-MLLM training invariant.
  * Student/teacher both receive `t_avg = (t_start + t_end) / 2` as a single
    scalar input — Tuna's `TimestepEmbedder` only takes a single t. This is
    an approximation; equal-width intervals make it bijective with the
    interval index. A future architectural change could let the student
    accept (t_start, t_end) explicitly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _x0_to_velocity(
    x0_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor, eps_t: float = 5e-2
) -> torch.Tensor:
    """Convert clean-image prediction to flow-matching velocity.

    Tuna JiT convention: t=0 is noise, t=1 is data. The interpolant is
    `z_t = t * x0 + (1 - t) * noise`, so the rectified-flow velocity is
    `(x0 - noise) = (x0 - z_t) / (1 - t)` (with eps for stability near t=1).
    """
    one_minus_t = (1.0 - t).clamp_min(eps_t)
    # Broadcast t over spatial/temporal dims
    while one_minus_t.dim() < x0_pred.dim():
        one_minus_t = one_minus_t.unsqueeze(-1)
    return (x0_pred - z_t) / one_minus_t


class MeanFlowDistillationWrapper(nn.Module):
    """Wraps a (student, teacher) pair for Mean Flow distillation training."""

    def __init__(
        self,
        student_model,
        teacher_model,
        num_intervals: int = 4,
        teacher_substeps: int = 4,
        loss_type: str = "l2",
        eps_t: float = 5e-2,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.num_intervals = num_intervals
        self.teacher_substeps = teacher_substeps
        self.loss_type = loss_type
        self.eps_t = eps_t

        self.register_buffer(
            "t_anchors", torch.linspace(0.0, 1.0, num_intervals + 1)
        )

    # ------------------------------------------------------------------
    # Attention mask: delegate to the student wrapper so Tuna's omni span
    # rule (image patches bidirectional within a span) is preserved.
    # ------------------------------------------------------------------
    def _build_masks(
        self,
        batch_size: int,
        seq_len: int,
        modality_positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        attn_mask, diff_mask = self.student.create_attention_mask(
            batch_size, seq_len, modality_positions, device, dtype
        )
        return attn_mask, diff_mask

    # ------------------------------------------------------------------
    # Teacher: compute the ground-truth mean velocity over [t_start, t_end]
    # by running K Euler sub-steps. Each sub-step converts x0→v first.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_teacher_mean_velocity(
        self,
        x_t_start: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask,
        diffhead_mask,
        modality_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = x_t_start.clone()
        K = self.teacher_substeps
        dt = (t_end - t_start) / K  # [B]

        for i in range(K):
            t_i = t_start + i * dt  # [B]

            _, x0_pred = self.teacher.tuna_model(
                text_tokens=text_tokens,
                image_latents=x,
                t=t_i,
                attention_mask=attention_mask,
                diffhead_attention_mask=diffhead_mask,
                modality_positions=modality_positions,
                output_hidden_states=False,
                max_seq_len=text_tokens.size(1),
                device=x.device,
            )
            # x0 → v conversion in Tuna's flow-matching convention
            v_i = _x0_to_velocity(x0_pred, x, t_i, eps_t=self.eps_t)
            dt_b = dt
            while dt_b.dim() < x.dim():
                dt_b = dt_b.unsqueeze(-1)
            x = x + dt_b * v_i

        # Mean velocity = displacement / interval width
        denom = (t_end - t_start)
        while denom.dim() < x.dim():
            denom = denom.unsqueeze(-1)
        mean_v = (x - x_t_start) / denom
        return mean_v

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        device = batch["images"].device
        text_tokens = batch["text_tokens"]
        modality_positions = batch["modality_positions"]
        B = text_tokens.shape[0]
        dtype = next(self.student.parameters()).dtype

        # Build attention masks via the student wrapper (preserves omni rule).
        attn_mask, diff_mask = self._build_masks(
            B, text_tokens.size(1), modality_positions, device, dtype
        )

        # 1) Random interval
        idx = torch.randint(0, self.num_intervals, (B,), device=device)
        t_start = self.t_anchors[idx].to(dtype)
        t_end = self.t_anchors[idx + 1].to(dtype)

        # 2) Sample noisy state at t_start
        x_clean = batch["images"]
        if x_clean.dim() == 4:
            x_clean = x_clean.unsqueeze(2)  # [B, C, 1, H, W]
        x_clean = x_clean.to(dtype)
        noise = torch.randn_like(x_clean)
        t_start_b = t_start
        while t_start_b.dim() < x_clean.dim():
            t_start_b = t_start_b.unsqueeze(-1)
        x_t_start = t_start_b * x_clean + (1 - t_start_b) * noise

        # 3) Teacher mean velocity (no_grad)
        v_target = self.compute_teacher_mean_velocity(
            x_t_start, t_start, t_end,
            text_tokens, attn_mask, diff_mask, modality_positions,
        )

        # 4) Student one-step prediction at t_avg, x0→v conversion
        t_avg = (t_start + t_end) / 2.0
        _, x0_pred_student = self.student.tuna_model(
            text_tokens=text_tokens,
            image_latents=x_t_start,
            t=t_avg,
            attention_mask=attn_mask,
            diffhead_attention_mask=diff_mask,
            modality_positions=modality_positions,
            output_hidden_states=False,
            max_seq_len=text_tokens.size(1),
            device=device,
        )
        v_pred = _x0_to_velocity(x0_pred_student, x_t_start, t_avg, eps_t=self.eps_t)

        # 5) Loss
        if self.loss_type == "l2":
            loss = ((v_pred - v_target) ** 2).mean()
        elif self.loss_type == "huber":
            loss = torch.nn.functional.huber_loss(v_pred, v_target, delta=0.1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return {"loss": loss, "v_pred": v_pred, "v_target": v_target}

    # ------------------------------------------------------------------
    # Few-step inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_few_step(
        self,
        x_T: torch.Tensor,
        text_tokens: torch.Tensor,
        modality_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Few-step inference: K Euler steps with student-predicted velocity."""
        device = x_T.device
        dtype = x_T.dtype
        B = text_tokens.shape[0]

        attn_mask, diff_mask = self._build_masks(
            B, text_tokens.size(1), modality_positions, device, dtype
        )

        x = x_T.clone()
        for i in range(self.num_intervals):
            t_start = self.t_anchors[i].expand(B).to(device=device, dtype=dtype)
            t_end = self.t_anchors[i + 1].expand(B).to(device=device, dtype=dtype)
            t_avg = (t_start + t_end) / 2.0

            _, x0_pred = self.student.tuna_model(
                text_tokens=text_tokens,
                image_latents=x,
                t=t_avg,
                attention_mask=attn_mask,
                diffhead_attention_mask=diff_mask,
                modality_positions=modality_positions,
                output_hidden_states=False,
                max_seq_len=text_tokens.size(1),
                device=device,
            )
            v = _x0_to_velocity(x0_pred, x, t_avg, eps_t=self.eps_t)
            dt = (t_end - t_start)
            while dt.dim() < x.dim():
                dt = dt.unsqueeze(-1)
            x = x + dt * v

        return x
