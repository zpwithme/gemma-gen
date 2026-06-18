# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from __future__ import annotations

import torch
import torch.nn as nn


class SimplePatchEmbedding(nn.Module):
    """Simple Conv2D-based patchifier with RMSNorm.

    Used by the "no encoder" Tuna variant (variant C). Takes raw RGB pixel
    values, splits them into non-overlapping patches via a strided Conv2d,
    and applies RMSNorm on the channel dimension.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 1152,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels

        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embedding(pixel_values)
        b, c, h, w = embeddings.shape
        embeddings = embeddings.reshape(b, c, h * w).transpose(1, 2)
        embeddings = self.norm(embeddings)
        return embeddings


class BottleneckPatchEmbedding(nn.Module):
    """Two-stage patchifier inspired by MiniT2I (Wang, He et al., 2026).

    Stage 1: Conv2d(in_channels -> bottleneck_dim) with stride=patch_size
             learns PCA-like color/edge structure at low channel dim.
    Stage 2: Conv2d(bottleneck_dim -> hidden_size, k=1) does channel mixing
             to the LLM hidden dimension.

    The bottleneck forces the spatial conv to be expressive in a low-dim
    color/edge basis before being lifted to the LLM hidden dim — empirically
    helps image quality at small/medium scale.
    """

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 1152,
        in_channels: int = 3,
        bottleneck_dim: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim

        # Stage 1: spatial Conv at bottleneck channel dim
        self.stage1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        # Stage 2: 1x1 channel mix to LLM hidden
        self.stage2 = nn.Conv2d(
            in_channels=bottleneck_dim,
            out_channels=hidden_size,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.stage1(pixel_values)
        x = self.stage2(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = self.norm(x)
        return x


def build_patch_embedding(
    vision_encoder_type: str,
    patch_size: int,
    hidden_size: int,
    in_channels: int,
    bottleneck_dim: int = 64,
) -> nn.Module:
    """Factory for patch embedding modules.

    Args:
        vision_encoder_type: 'simple' (Tuna-2 default) or 'bottleneck' (MiniT2I-style).
        patch_size: spatial patch size (e.g. 16).
        hidden_size: target LLM hidden dim.
        in_channels: input channels (3 for RGB).
        bottleneck_dim: bottleneck width when type=='bottleneck'.
    """
    if vision_encoder_type == "bottleneck":
        return BottleneckPatchEmbedding(
            patch_size=patch_size,
            hidden_size=hidden_size,
            in_channels=in_channels,
            bottleneck_dim=bottleneck_dim,
        )
    elif vision_encoder_type == "simple":
        return SimplePatchEmbedding(
            patch_size=patch_size,
            hidden_size=hidden_size,
            in_channels=in_channels,
        )
    else:
        raise ValueError(
            f"Unknown vision_encoder_type: {vision_encoder_type!r}. "
            "Choose 'simple' or 'bottleneck'."
        )
