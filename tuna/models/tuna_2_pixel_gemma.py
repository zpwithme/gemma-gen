# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tuna-2 pixel variant with Gemma 4 backbone.

Three changes vs the original `Tuna2Pixel` (`tuna/models/tuna_2_pixel.py`):

  1. LLM backbone loaded via `AutoModelForCausalLM.from_pretrained` so any HF
     CausalLM (Gemma 4 12B by default) can plug in. Gemma 4 12B is an
     encoder-free unified multimodal model, so the pretrained weights already
     carry vision-language alignment.
  2. Vision encoder is built via `build_patch_embedding`, so the user can
     switch between Tuna-2's simple Conv2d and MiniT2I-style
     `BottleneckPatchEmbedding` from config.
  3. Optional `PixelREPAModule` auxiliary alignment (training only).

Attention masks for Gemma's interleaved sliding+global layers are handled
inside `Tuna2PixelGemmaModel.create_attention_mask` (see wrapper below),
which uses `gemma_omni_attn` to OR Gemma's base rule with Tuna's same-span
bidirectional override.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM

from tuna.models.backbones.modules_new_mm import (
    DiffusionHeadConfig,
    FinalLayer,
    ModulatedAttentionBlock,
    TimestepEmbedder,
)
from tuna.models._common import _patchify_5d, build_rope
from tuna.models._inner_base import TunaInnerBase
from tuna.models._wrapper_base import JiTWrapperMixin, TunaWrapperBase
from tuna.models.misc import get_text_tokenizer, next_token_prediction
from tuna.models.pixelrepa import PixelREPAModule
from tuna.models.vision.patch_embed import build_patch_embedding

logger = logging.getLogger(__name__)


class Tuna2PixelGemma(TunaInnerBase):
    """Pure-patchify Tuna inner model with Gemma 4 backbone."""

    @register_to_config
    def __init__(
        self,
        llm_vocab_size=None,
        llm_model_path: str = "google/gemma-4-12b",
        llm_family: str = "gemma4",
        init_llm_from_config: bool = False,
        image_latent_dim: int = 3,
        image_latent_height: int = 16,
        image_latent_width: int = 16,
        video_latent_height: int = 16,
        video_latent_width: int = 16,
        reshape_frame_to_batch_dim: bool = False,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        patch_size: int = 16,
        hidden_size: int = 3072,
        num_diffusion_layers: int = 16,
        add_aspect_ratio_embeds: bool = False,
        add_time_embeds: bool = True,
        use_disp: bool = False,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_kwargs=None,
        enable_mask_token: bool = False,
        masked_image_ratio_min: float = -0.7,
        masked_image_ratio: float = 0.3,
        # --- New for Gemma version ---
        vision_encoder_type: str = "simple",
        vision_bottleneck_dim: int = 64,
        enable_pixelrepa: bool = False,
        pixelrepa_config: Optional[dict] = None,
        sliding_window: int = 1024,
        attn_implementation: str = "sdpa",
        **kwargs,
    ):
        super().__init__()
        self.use_disp = use_disp
        self.llm_family = llm_family
        self.sliding_window = sliding_window

        # ---- LLM backbone (Gemma 4 12B by default, any HF CausalLM works)
        llm_config = AutoConfig.from_pretrained(llm_model_path)
        # Map Tuna's attention_backend → HF's attn_implementation string.
        # HF accepts: "eager", "sdpa", "flash_attention_2", "flex_attention".
        # Tuna internally uses "flexattention" (no underscore) for its own mask
        # code; HF needs the underscored name.
        hf_attn_impl = {
            "sdpa": "sdpa",
            "flexattention": "flex_attention",
            "flex_attention": "flex_attention",
            "eager": "eager",
            "flash_attention_2": "flash_attention_2",
        }.get(attn_implementation, "sdpa")
        if init_llm_from_config:
            self.tuna = AutoModelForCausalLM.from_config(llm_config)
        else:
            self.tuna = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                attn_implementation=hf_attn_impl,
            )
        # Add Tuna special tokens (handled by wrapper before calling here).
        if llm_vocab_size is not None:
            self.tuna.resize_token_embeddings(llm_vocab_size)

        # Pull actual hidden size from the loaded config to avoid mismatch.
        llm_hidden = getattr(self.tuna.config, "hidden_size", None) or hidden_size
        llm_intermediate = getattr(self.tuna.config, "intermediate_size", llm_hidden * 4)
        llm_max_pos = getattr(self.tuna.config, "max_position_embeddings", 4096)

        # ---- Patch embedding (Conv2d or Bottleneck, MiniT2I-style)
        self.vision_encoder = build_patch_embedding(
            vision_encoder_type=vision_encoder_type,
            patch_size=patch_size,
            hidden_size=llm_hidden,
            in_channels=image_latent_dim,
            bottleneck_dim=vision_bottleneck_dim,
        )

        self.register_buffer(
            "image_position_ids",
            torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
            persistent=False,
        )

        # ---- Diffusion head (16 layers ModulatedAttentionBlock + FinalLayer)
        self.diffusion_head_config = DiffusionHeadConfig(
            hidden_size=llm_hidden,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=llm_intermediate,
            max_position_embeddings=llm_max_pos,
        )
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if add_aspect_ratio_embeds:
            self.aspect_ratio_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(
                    self.diffusion_head_config.hidden_size,
                    self.diffusion_head_config.hidden_size,
                ),
            )
            self.time_embed_proj = nn.Linear(
                self.diffusion_head_config.hidden_size, hidden_size
            )
            if add_aspect_ratio_embeds:
                self.ar_embed_proj = nn.Linear(
                    self.diffusion_head_config.hidden_size, hidden_size
                )

        self.diffusion_head_a = nn.ModuleList(
            [
                ModulatedAttentionBlock(self.diffusion_head_config, layer_idx)
                for layer_idx in range(num_diffusion_layers)
            ]
        )
        self.diffusion_head_b = FinalLayer(
            self.diffusion_head_config.hidden_size, patch_size, image_latent_dim
        )

        # ---- Masked image token (DeTok-style)
        self.enable_mask_token = enable_mask_token
        self.masked_image_ratio = masked_image_ratio
        self.masked_image_ratio_min = masked_image_ratio_min
        if self.enable_mask_token:
            scale = llm_hidden ** -0.5
            self.mask_token = nn.Parameter(scale * torch.randn(1, 1, llm_hidden))

        # ---- PixelREPA (optional auxiliary alignment, training-only)
        self.enable_pixelrepa = enable_pixelrepa
        if enable_pixelrepa:
            cfg = dict(pixelrepa_config or {})
            self.pixelrepa = PixelREPAModule(llm_hidden_size=llm_hidden, **cfg)
            logger.info("[Tuna2PixelGemma] PixelREPA ENABLED")
        else:
            self.pixelrepa = None

        # ---- Gradient checkpointing
        self.gradient_checkpointing = False
        if gradient_checkpointing:
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = functools.partial(
                checkpoint, **(gradient_checkpointing_kwargs or {})
            )
            if hasattr(self.tuna, "gradient_checkpointing_enable"):
                self.tuna.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
                )

        self.reset_parameters()

    def reset_parameters(self):
        """Match Tuna2Pixel's init scheme; handle both Simple and Bottleneck."""
        ve = self.vision_encoder
        if hasattr(ve, "patch_embedding"):  # SimplePatchEmbedding
            w = ve.patch_embedding.weight.data
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            nn.init.constant_(ve.patch_embedding.bias, 0)
        else:  # BottleneckPatchEmbedding
            w1 = ve.stage1.weight.data
            nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
            nn.init.constant_(ve.stage1.bias, 0)
            w2 = ve.stage2.weight.data
            nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
            nn.init.constant_(ve.stage2.bias, 0)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.diffusion_head_a)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers (DiT convention)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def _image_meta_token_count(self) -> int:
        """Number of meta tokens written at the START of each image span by
        `_inner_base._prepare_input`. Per the convention there:

            add_time_embeds + add_aspect_ratio_embeds → 3 (height, width, time)
            add_time_embeds only                       → 1 (time)
            neither                                    → 0
        """
        if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
            return 3
        if self.config.add_time_embeds:
            return 1
        return 0

    # -------------------------------------------------------------------
    # Forward (mirrors Tuna2Pixel.forward + PixelREPA hook)
    # -------------------------------------------------------------------
    def forward(
        self,
        text_tokens=None,
        image_latents=None,
        t=None,
        attention_mask=None,
        diffhead_attention_mask=None,
        text_masks=None,
        image_masks=None,
        text_labels=None,
        image_labels=None,
        modality_positions=None,
        first_frame_as_cond=False,
        only_denoise_last_image=False,
        guidance_scale=0.0,
        output_hidden_states=True,
        max_seq_len=None,
        device="cuda:0",
        label=None,
        return_input_embeds=False,
        image_grid_thw=None,
        clean_pixel_values=None,  # NEW: required when enable_pixelrepa
        **kwargs,
    ):
        clean_image_embeds = kwargs.get("clean_image_embeds", None)

        # Text embeddings via Gemma's embed layer
        input_embeds = self.tuna.get_input_embeddings()(text_tokens)
        dtype = input_embeds.dtype

        # Replace text embeddings with clean image embeddings (reconstruction mode)
        if clean_image_embeds is not None and modality_positions is not None:
            clean_seq_len = clean_image_embeds.shape[1]
            for i in range(input_embeds.shape[0]):
                img_offset = int(modality_positions[i][0][0])
                replace_start = img_offset - clean_seq_len - 1
                if replace_start >= 0:
                    input_embeds[i, replace_start : replace_start + clean_seq_len, :] = (
                        clean_image_embeds[i].to(dtype)
                    )

        (
            image_embeds,
            time_embeds,
            time_embeds_proj,
            height_embeds_proj,
            width_embeds_proj,
            rope_3d,
        ) = self._prepare_embeds(image_latents, t, device, dtype)

        b = T = h = w = h_ = w_ = None
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            p = self.config.patch_size
            h_, w_ = h // p, w // p

        new_image_labels = None
        if image_labels is not None:
            image_labels = rearrange(image_labels, "b c t h w -> b (t h w) c")
            image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
            image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))
            new_image_labels = torch.zeros(
                [image_embeds.shape[0], max_seq_len, p * p * c],
                device=device,
                dtype=dtype,
            )
            image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

        if image_embeds is not None:
            input_embeds, new_image_labels, image_masks = self._prepare_input(
                input_embeds,
                image_embeds,
                image_labels,
                image_masks,
                new_image_labels,
                modality_positions,
                height_embeds_proj,
                width_embeds_proj,
                time_embeds_proj,
            )

        if return_input_embeds:
            return input_embeds

        # Only force hidden states when PixelREPA is active and we are training.
        # Otherwise honor the caller (Mean Flow distill / AR video pipeline pass
        # False to save memory — Gemma 12B's 48 hidden states are an OOM bomb).
        need_hidden = (
            output_hidden_states
            or (self.enable_pixelrepa and self.training and self.pixelrepa is not None)
        )
        outputs = self.tuna(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
        )

        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if need_hidden:
            last_hidden_states = (
                outputs.hidden_states[-1]
                if hasattr(outputs, "hidden_states")
                else outputs["hidden_states"][-1]
            )
        else:
            # The diffusion head requires last_hidden_states; if caller asked
            # for inference-only mode (no hidden states), reuse logits hidden
            # via a model-specific path. For Gemma the last hidden state is
            # also accessible via outputs.last_hidden_state on the base model.
            last_hidden_states = getattr(outputs, "last_hidden_state", None)
            if last_hidden_states is None:
                # Fallback: re-run with hidden states (rare path).
                outputs = self.tuna(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden_states = outputs.hidden_states[-1]

        # PixelREPA hook: pull a configurable mid-layer
        loss_repa = torch.tensor(0.0, device=device, dtype=dtype)
        if (
            self.enable_pixelrepa
            and self.training
            and self.pixelrepa is not None
            and clean_pixel_values is not None
            and outputs is not None
            and hasattr(outputs, "hidden_states")
            and outputs.hidden_states is not None
        ):
            mid_hidden = outputs.hidden_states[self.pixelrepa.from_layer]
            loss_repa = self.pixelrepa(
                llm_hidden=mid_hidden,
                image_positions=modality_positions,
                clean_pixel_values=clean_pixel_values,
                # NEW: tell PixelREPA where the real patches start within each
                # span (skip Tuna's time/aspect-ratio meta tokens at offset 0..N).
                meta_token_count=self._image_meta_token_count(),
            )

        position_ids = torch.arange(
            last_hidden_states.shape[1], device=last_hidden_states.device
        ).unsqueeze(0)

        if hasattr(self, "diff_proj"):
            last_hidden_states = self.diff_proj(last_hidden_states)

        if diffhead_attention_mask is None:
            diffhead_attention_mask = attention_mask

        act = []
        for layer in self.diffusion_head_a:
            if self.gradient_checkpointing and self.training:
                last_hidden_states = self._gradient_checkpointing_func(
                    layer,
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=None,
                    modality_positions=modality_positions,
                )[0]
            else:
                last_hidden_states = layer(
                    hidden_states=last_hidden_states,
                    adaln_input=time_embeds,
                    attention_mask=diffhead_attention_mask,
                    position_ids=position_ids,
                    rope_3d=rope_3d,
                    modality_positions=modality_positions,
                )[0]
            act.append(last_hidden_states)

        x0_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)

        loss_disp = torch.tensor(0.0, device=device)
        if image_latents is None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            loss_flow = torch.tensor(0.0, device=device)
            return logits, loss_ntp, loss_flow, loss_disp, loss_repa

        if text_labels is not None and image_labels is not None:
            from tuna.models.misc import jit_x0_prediction_loss

            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            if t.shape[0] == x0_pred.shape[0]:
                t_for_loss = t
            elif t.shape[0] == x0_pred.shape[0] * 2:
                t_for_loss = t[1::2]
            else:
                t_for_loss = None
            loss_flow = jit_x0_prediction_loss(
                x0_pred, new_image_labels[: x0_pred.shape[0]], t_for_loss, image_masks
            )
            if self.use_disp:
                loss_disp = self.disp_loss(act[-1])
            return logits, loss_ntp, loss_flow, loss_disp, loss_repa

        else:
            # Inference: return x0 predictions
            x0_pred_ = []
            num_imgs = 0
            for i, modality_batch in enumerate(modality_positions):
                for _, (offset, length) in enumerate(modality_batch):
                    if length == 0:
                        break
                    x0_pred_.append(x0_pred[i, offset : offset + length])
                    num_imgs += 1
            x0_pred_ = torch.stack(x0_pred_)

            if self.config.add_time_embeds and self.config.add_aspect_ratio_embeds:
                x0_pred_ = x0_pred_[:, 3:, :]
            elif self.config.add_time_embeds:
                x0_pred_ = x0_pred_[:, 1:, :]

            x0_pred_ = self.unpatchify(x0_pred_, h_, w_, T=T)
            x0_pred_ = x0_pred_.permute(0, 3, 1, 2)
            x0_pred_ = x0_pred_.reshape(
                num_imgs,
                self.config.image_latent_dim,
                T,
                h_ * self.config.patch_size,
                w_ * self.config.patch_size,
            )
            return logits, x0_pred_

    # -------------------------------------------------------------------
    # Helpers (identical to Tuna2Pixel)
    # -------------------------------------------------------------------
    def _prepare_embeds(self, image_latents, t, device, dtype):
        image_embeds = None
        rope_3d = None
        h_ = w_ = None
        b = None
        if image_latents is not None:
            b, c, T, h, w = image_latents.shape
            rope_3d = (
                build_rope(latent_shape=[T, h, w], patch_size=16, attention_head_dim=64)
                .to(device)
                .to(dtype)
            )
            p = self.config.patch_size
            h_, w_ = h // p, w // p
            image_embeds = _patchify_5d(
                image_latents.to(dtype),
                self.vision_encoder,
                self.config.reshape_frame_to_batch_dim,
            )
            if self.enable_mask_token and self.training:
                import random

                import numpy as np

                B_, N, D = image_embeds.shape
                for bi in range(B_):
                    mask_ratio = random.uniform(
                        self.config.masked_image_ratio_min, self.masked_image_ratio
                    )
                    num_masked = int(N * mask_ratio)
                    if num_masked > 0:
                        idx = np.random.choice(N, num_masked, replace=False)
                        image_embeds[bi, idx] = self.mask_token

        time_embeds = self.time_embed(t, dtype)
        time_embeds_proj = (
            self.time_embed_proj(time_embeds) if hasattr(self, "time_embed_proj") else time_embeds
        )

        height_embeds_proj = width_embeds_proj = None
        if hasattr(self, "aspect_ratio_embed") and h_ is not None:
            lh = torch.tensor(h_, device=device).repeat(b)
            lw = torch.tensor(w_, device=device).repeat(b)
            h_emb = self.aspect_ratio_embed(lh, dtype)
            w_emb = self.aspect_ratio_embed(lw, dtype)
            if hasattr(self, "ar_embed_proj"):
                height_embeds_proj = self.ar_embed_proj(h_emb)
                width_embeds_proj = self.ar_embed_proj(w_emb)
            else:
                height_embeds_proj = h_emb
                width_embeds_proj = w_emb
        return (
            image_embeds,
            time_embeds,
            time_embeds_proj,
            height_embeds_proj,
            width_embeds_proj,
            rope_3d,
        )

    @torch.no_grad()
    def t2i_generate(self, *args, **kwargs):
        """Reuse Tuna2Pixel's CFG/sampling logic — same signature."""
        from tuna.models.tuna_2_pixel import Tuna2Pixel
        return Tuna2Pixel.t2i_generate(self, *args, **kwargs)

    @torch.no_grad()
    def t2i_generate_edit(self, *args, **kwargs):
        from tuna.models.tuna_2_pixel import Tuna2Pixel
        return Tuna2Pixel.t2i_generate_edit(self, *args, **kwargs)


# =====================================================================
# Wrapper (Hydra-instantiable)
# =====================================================================

class Tuna2PixelGemmaModel(JiTWrapperMixin, TunaWrapperBase):
    """Hydra-instantiable training wrapper for Tuna2PixelGemma.

    Subclasses Tuna2PixelModel-like behavior but loads Gemma 4 backbone and
    threads clean_pixel_values + PixelREPA loss through training step.
    """

    def __init__(
        self,
        llm_model_path: str = "google/gemma-4-12b",
        llm_family: str = "gemma4",
        load_stage1_model: Optional[str] = None,
        frozen_params: Optional[List[str]] = None,
        hidden_size: int = 3072,
        image_latent_dim: int = 3,
        image_latent_height: int = 16,
        image_latent_width: int = 16,
        hq_image_latent_height: int = 64,
        hq_image_latent_width: int = 64,
        mixed_modal_latent_height: int = 32,
        mixed_modal_latent_width: int = 32,
        patch_size: int = 16,
        add_time_embeds: bool = True,
        add_aspect_ratio_embeds: bool = False,
        mrope_type: str = "dit_3drope_mm",
        reshape_frame_to_batch_dim: bool = False,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 8,
        attention_backend: str = "sdpa",
        ntp_coeff: float = 1.0,
        flow_coeff: float = 1.0,
        und_max_t0: float = 1.0,
        use_disp: bool = False,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_kwargs: Optional[DictConfig] = None,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: Optional[str] = None,
        train_eps: Optional[float] = 1e-5,
        sample_eps: Optional[float] = 1e-3,
        snr_type: str = "lognorm",
        do_shift: bool = True,
        sampling_method: str = "euler",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
        time_shifting_factor: float = 3.0,
        dtype: str = "bf16",
        flow_head_num: int = 16,
        enable_mask_token: bool = False,
        masked_image_ratio_min: float = -0.7,
        masked_image_ratio: float = 0.3,
        mmu_noise_prob: float = 0.1,
        mmu_noise_level: float = 0.1,
        noise_scale: float = 2.0,
        # --- New for Gemma version ---
        vision_encoder_type: str = "simple",
        vision_bottleneck_dim: int = 64,
        enable_pixelrepa: bool = False,
        pixelrepa_config: Optional[dict] = None,
        sliding_window: int = 1024,
        gemma_omni_override: bool = True,
        preserve_text_sliding: bool = True,
    ):
        super().__init__()

        # Store config (mirrors Tuna2PixelModel.__init__)
        self.llm_model_path = llm_model_path
        self.llm_family = llm_family
        self.frozen_params = frozen_params
        self.hidden_size = hidden_size
        self.image_latent_dim = image_latent_dim
        self.image_latent_height = image_latent_height
        self.image_latent_width = image_latent_width
        self.hq_image_latent_height = hq_image_latent_height
        self.hq_image_latent_width = hq_image_latent_width
        self.mixed_modal_latent_height = mixed_modal_latent_height
        self.mixed_modal_latent_width = mixed_modal_latent_width
        self.patch_size = patch_size
        self.add_time_embeds = add_time_embeds
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds
        self.load_stage1_model = load_stage1_model
        self.flow_head_num = flow_head_num
        self.mrope_type = mrope_type
        self.reshape_frame_to_batch_dim = reshape_frame_to_batch_dim
        self.attention_backend = attention_backend

        self.ntp_coeff = ntp_coeff
        self.flow_coeff = flow_coeff
        self.und_max_t0 = und_max_t0
        self.use_disp = use_disp
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_kwargs = (
            OmegaConf.to_container(gradient_checkpointing_kwargs)
            if gradient_checkpointing_kwargs is not None
            else None
        )

        self.path_type = path_type
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.snr_type = snr_type
        self.do_shift = do_shift

        self.sampling_method = sampling_method
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.atol = atol
        self.rtol = rtol
        self.reverse = reverse
        self.time_shifting_factor = time_shifting_factor

        self.enable_mask_token = enable_mask_token
        self.masked_image_ratio_min = masked_image_ratio_min
        self.masked_image_ratio = masked_image_ratio
        self.mmu_noise_prob = mmu_noise_prob
        self.mmu_noise_level = mmu_noise_level
        self.dtype = dtype

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.noise_scale = noise_scale

        # New
        self.vision_encoder_type = vision_encoder_type
        self.vision_bottleneck_dim = vision_bottleneck_dim
        self.enable_pixelrepa = enable_pixelrepa
        self.pixelrepa_config = (
            OmegaConf.to_container(pixelrepa_config)
            if isinstance(pixelrepa_config, DictConfig)
            else pixelrepa_config
        )
        self.sliding_window = sliding_window
        self.gemma_omni_override = gemma_omni_override
        # NOTE: preserve_text_sliding is intentionally NOT stored — the
        # current mask implementation uses a full-causal base + same-span
        # union, which already satisfies "sliding for text" via HF's own
        # per-layer sliding inside Gemma. Keeping the arg in __init__ for
        # back-compat with old yaml configs, but it's a no-op.

        self.build_models()

    def build_models(self):
        from tuna.models.misc import JiTNoiseScheduler
        from tuna.models.transport.define import create_transport
        from tuna.models.transport.transport import Sampler

        # Tokenizer with Gemma family
        self.text_tokenizer, self.tuna_token_ids = get_text_tokenizer(
            self.llm_model_path,
            add_tuna_tokens=True,
            return_tuna_token_ids=True,
            llm_name=self.llm_family,
        )
        self.llm_vocab_size = len(self.text_tokenizer)

        model_config = dict(
            llm_vocab_size=self.llm_vocab_size,
            llm_model_path=self.llm_model_path,
            llm_family=self.llm_family,
            init_llm_from_config=False,
            image_latent_dim=self.image_latent_dim,
            image_latent_height=self.image_latent_height,
            image_latent_width=self.image_latent_width,
            video_latent_height=self.image_latent_height,
            video_latent_width=self.image_latent_width,
            reshape_frame_to_batch_dim=self.reshape_frame_to_batch_dim,
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            add_time_embeds=self.add_time_embeds,
            add_aspect_ratio_embeds=self.add_aspect_ratio_embeds,
            num_diffusion_layers=self.flow_head_num,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            use_disp=self.use_disp,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            enable_mask_token=self.enable_mask_token,
            masked_image_ratio_min=self.masked_image_ratio_min,
            masked_image_ratio=self.masked_image_ratio,
            vision_encoder_type=self.vision_encoder_type,
            vision_bottleneck_dim=self.vision_bottleneck_dim,
            enable_pixelrepa=self.enable_pixelrepa,
            pixelrepa_config=self.pixelrepa_config,
            sliding_window=self.sliding_window,
            attn_implementation=self.attention_backend,
        )

        self.tuna_model = Tuna2PixelGemma(**model_config)

        if self.load_stage1_model is not None and self.load_stage1_model != "no":
            ckpt = torch.load(self.load_stage1_model, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            model_dict = self.tuna_model.state_dict()
            filtered = {
                k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape
            }
            logger.info(f"Loaded {len(filtered)} / {len(model_dict)} keys from checkpoint.")
            self.tuna_model.load_state_dict(filtered, strict=False)

        self._freeze_params(self.tuna_model, self.frozen_params)

        # Transport / sampler
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
            snr_type=self.snr_type,
            do_shift=self.do_shift,
        )
        self.sampler = Sampler(self.transport)

        # JiT-style noise scheduler (pixel variant)
        self.jit_noise_scheduler = JiTNoiseScheduler(
            P_mean=-0.8,
            P_std=0.8,
            noise_scale=self.noise_scale,
            t_eps=5e-2,
        )

    @torch.no_grad()
    def prepare_clean_image_embeds(self, pixel_values_low: torch.Tensor) -> torch.Tensor:
        if len(pixel_values_low.shape) == 4:
            pixel_values_low = pixel_values_low.unsqueeze(2)
        dtype = next(self.tuna_model.parameters()).dtype
        return _patchify_5d(
            pixel_values_low.to(dtype),
            self.tuna_model.vision_encoder,
            self.tuna_model.config.reshape_frame_to_batch_dim,
        )

    # ------------------------------------------------------------------
    # Override create_attention_mask to use Gemma's omni override
    # ------------------------------------------------------------------
    def create_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        modality_positions,
        device,
        dtype: torch.dtype,
    ):
        """Use Gemma sliding+global rule OR-merged with omni same-span rule.

        cross_frame_causal is enabled when BOTH:
          * `self._video_cross_frame_causal` is True (set by train.py when
            stage='ar_teacher_force' or by the AR-video pipeline at inference)
          * the current batch has multi-frame modality_positions
            (`modality_positions.shape[1] > 1`).

        Implementation detail: we build ONE mask used by all layers via the
        UNION of layer behaviors. The right base is the WEAKEST per-layer rule
        (full causal = global layer's pattern), then we OR in the same-span
        bidirectional override. Using `is_local_layer=True` here would
        over-restrict global layers (sliding ⊂ causal), killing Gemma's
        pretrained long-range attention.
        """
        if not self.gemma_omni_override:
            return super().create_attention_mask(
                batch_size, seq_length, modality_positions, device, dtype
            )

        ar_video_flag = getattr(self, "_video_cross_frame_causal", False)
        is_multi_frame = (
            modality_positions is not None
            and modality_positions.ndim == 3
            and modality_positions.shape[1] > 1
        )
        use_cross_frame_causal = bool(ar_video_flag and is_multi_frame)

        from tuna.models.gemma_omni_attn import build_gemma_omni_attn_mask_naive

        if self.attention_backend == "sdpa":
            mask = build_gemma_omni_attn_mask_naive(
                modality_positions=modality_positions,
                seq_len=seq_length,
                sliding_window=self.sliding_window,
                is_local_layer=False,
                cross_frame_causal=use_cross_frame_causal,
                device=device,
                dtype=dtype,
                inverted=True,
            )
            return mask, None
        else:
            from tuna.models.gemma_omni_attn import build_gemma_omni_block_mask

            block_mask = build_gemma_omni_block_mask(
                modality_positions=modality_positions,
                seq_len=seq_length,
                layer_idx=0,
                layer_pattern=["global"],
                sliding_window=self.sliding_window,
                num_heads=self.num_attention_heads,
                cross_frame_causal=use_cross_frame_causal,
                device=device,
            )
            return block_mask, block_mask

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        weight_type = torch.bfloat16 if self.dtype == "bf16" else torch.float32

        text_tokens = batch["text_tokens"]
        text_labels = batch["text_labels"]
        pixel_values = batch["images"]
        pixel_values_low = batch.get("images_clip", None)
        text_masks = batch["text_masks"]
        image_masks = batch["image_masks"]
        modality_positions = batch["modality_positions"]
        data_type = batch["data_type"]

        if data_type[0] == "mmu_interleaved" or data_type[0] == "edit_interleaved":
            b, n = pixel_values.shape[:2]
            pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
            data_type = data_type * n
        elif data_type[0] in ("t2v_pixel", "t2v") and pixel_values.dim() == 5:
            # Video path: rearrange [B, C, T, H, W] -> [B*T, C, 1, H, W] so the
            # downstream prepare_latents_and_labels / _prepare_input pipeline
            # treats each frame as a single image with its own t. The matching
            # modality_positions [B, T, 2] from PixelVideoDataset slots each
            # frame into its own visual span. Combined with cross_frame_causal
            # in the attention mask, this is the AR-video forward layout.
            b, c, T, h_, w_ = pixel_values.shape
            pixel_values = rearrange(pixel_values, "b c t h w -> (b t) c h w").unsqueeze(2)
            data_type = list(data_type) * T

        if data_type[0] != "mmu_text":
            (
                image_latents,
                t,
                image_labels,
                image_masks,
                image_latents_clean,
            ) = self.prepare_latents_and_labels(pixel_values, data_type, image_masks)
            # Clean pixels (needed for PixelREPA)
            clean_pixel_values_for_repa = (
                pixel_values.to(weight_type) if self.enable_pixelrepa else None
            )
        else:
            image_latents = None
            t = torch.tensor([0.0] * text_tokens.shape[0], device=text_tokens.device)
            image_labels = None
            image_masks = None
            clean_pixel_values_for_repa = None

        clean_image_embeds = None
        if pixel_values_low is not None:
            clean_image_embeds = self.prepare_clean_image_embeds(pixel_values_low)

        block_mask, block_mask_diffhead = self.create_attention_mask(
            text_tokens.size(0),
            text_tokens.size(1),
            modality_positions,
            text_tokens.device,
            weight_type,
        )

        out = self.tuna_model(
            text_tokens=text_tokens,
            image_latents=image_latents,
            t=t.to(weight_type),
            attention_mask=block_mask,
            diffhead_attention_mask=block_mask_diffhead,
            text_masks=text_masks,
            image_masks=image_masks,
            text_labels=text_labels,
            image_labels=image_labels,
            modality_positions=modality_positions,
            output_hidden_states=True,
            max_seq_len=text_tokens.size(1),
            device=text_tokens.device,
            clean_image_embeds=clean_image_embeds,
            clean_pixel_values=clean_pixel_values_for_repa,
        )

        # The Gemma forward returns one extra item: loss_repa
        logits, loss_ntp, loss_flow, loss_disp, loss_repa = out

        total_loss = self.flow_coeff * loss_flow + self.ntp_coeff * loss_ntp
        if self.use_disp:
            total_loss = total_loss + 0.25 * loss_disp
        if self.enable_pixelrepa:
            total_loss = total_loss + loss_repa  # weight already applied inside module

        return {
            "loss": total_loss,
            "loss_ntp": loss_ntp,
            "loss_flow": loss_flow,
            "loss_disp": loss_disp,
            "loss_repa": loss_repa,
            "logits": logits,
            "recons_images": None,
        }
