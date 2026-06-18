# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pixel-space video dataset for Tuna-2 pixel + AR video training.

Differs from `VideoDataset` (`tuna/data/datasets/video_dataset.py`) in two
critical ways:

  1. No VAE compression assumed.
     - `num_tokens_per_frame = (H/patch_size) * (W/patch_size)` directly.
     - No temporal compression (`temporal_ds = 1`).

  2. Per-frame `modality_positions` spans.
     - Original `VideoDataset` emits one big span covering all frames
       (which makes the LLM attend bidirectionally across the whole video).
     - This dataset emits N spans (one per frame), enabling the AR mask
       (frame-internal bidirectional + cross-frame causal) used by the
       AR video pipeline.

When `num_frames=1` the per-frame splitting degenerates to a single span,
so the same dataset can serve as an image dataset for image+video
joint training (matching MovieGen / HunyuanVideo recipe).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import format_sequence_gen_qwen2_5
from tuna.data.transforms import build_image_transform

logger = logging.getLogger(__name__)


class PixelVideoDataset(Dataset):
    """Pixel-space video-text dataset (no VAE, per-frame spans).

    Args:
        jsonl_path: JSONL manifest — each line has {video, caption} where
            `video` is either a video file or a directory of frames.
        image_root: Root dir for relative paths.
        tokenizer: HuggingFace tokenizer with Tuna special tokens registered.
        image_size: (H, W) per frame.
        num_frames: Number of frames to load per clip (T).
        patch_size: Pixel patch size (Conv2d stride, default 16).
        max_text_length: Max unified-sequence length.
        tuna_token_ids: Token ID dict from the model wrapper.
        per_frame_spans: If True, emit one span per frame (for AR mask).
            If False, emit one big span (legacy bidirectional behavior).
        use_chat_template: Whether to use chat-template formatting.
    """

    _VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = (512, 512),
        num_frames: int = 1,
        patch_size: int = 16,
        max_text_length: int = 2048,
        video_field: str = "video",
        text_field: str = "caption",
        center_crop: bool = True,
        tuna_token_ids: dict[str, int] | None = None,
        per_frame_spans: bool = True,
        use_chat_template: bool = False,
        data_type: str = "t2v_pixel",
        add_time_embeds: bool = True,
        add_aspect_ratio_embeds: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.max_text_length = max_text_length
        self.video_field = video_field
        self.text_field = text_field
        self.per_frame_spans = per_frame_spans
        self.use_chat_template = use_chat_template
        self.data_type = data_type
        self.add_time_embeds = add_time_embeds
        self.add_aspect_ratio_embeds = add_aspect_ratio_embeds

        self.records = self._load_jsonl(jsonl_path)
        self.image_transform = build_image_transform(self.image_size, center_crop)

        # No VAE: tokens per frame = (H/p) * (W/p)
        h_patches = self.image_size[0] // patch_size
        w_patches = self.image_size[1] // patch_size
        self.num_tokens_per_frame = h_patches * w_patches

        # Meta tokens written at the START of each span by _prepare_input.
        # 3 (height, width, time) if both add_aspect_ratio_embeds + add_time_embeds;
        # 1 (time) if only add_time_embeds; 0 otherwise.
        if add_time_embeds and add_aspect_ratio_embeds:
            self.n_meta = 3
        elif add_time_embeds:
            self.n_meta = 1
        else:
            self.n_meta = 0
        self.num_tokens_per_frame_with_meta = self.num_tokens_per_frame + self.n_meta

        # Total visual tokens for this video
        self.total_num_visual_tokens = (
            self.num_tokens_per_frame_with_meta * self.num_frames
        )

        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.img_pad_id = tuna_token_ids["img_pad_id"]
        else:
            raise ValueError("tuna_token_ids is required for PixelVideoDataset")

        logger.info(
            f"PixelVideoDataset: loaded {len(self.records)} records from {jsonl_path} "
            f"(num_frames={num_frames}, image_size={self.image_size}, "
            f"patch_size={patch_size}, tokens_per_frame={self.num_tokens_per_frame_with_meta}, "
            f"per_frame_spans={per_frame_spans})"
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        records = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"{path}:{ln}: skipping malformed JSON: {e}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, rel_or_abs: str) -> str:
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.image_root, rel_or_abs)

    def _load_frames(self, video_path: str) -> list[Image.Image]:
        full_path = self._resolve_path(video_path)
        if os.path.isfile(full_path) and full_path.lower().endswith(self._VIDEO_EXTENSIONS):
            return self._load_frames_from_video(full_path)
        return self._load_frames_from_dir(full_path)

    def _load_frames_from_video(self, video_file: str) -> list[Image.Image]:
        import cv2

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_file}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"Video has no frames: {video_file}")

        if total >= self.num_frames:
            indices = [int(i * total / self.num_frames) for i in range(self.num_frames)]
        else:
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(total - 1)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise RuntimeError(f"Failed to read frame {idx}: {video_file}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        cap.release()
        return frames

    def _load_frames_from_dir(self, frame_dir: str) -> list[Image.Image]:
        files = sorted(
            f for f in os.listdir(frame_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if len(files) > self.num_frames:
            indices = [int(i * len(files) / self.num_frames) for i in range(self.num_frames)]
            files = [files[i] for i in indices]
        elif len(files) < self.num_frames:
            while len(files) < self.num_frames:
                files.append(files[-1])
        frames = []
        for fname in files:
            with Image.open(os.path.join(frame_dir, fname)) as img:
                img.load()
                if img.mode != "RGB":
                    img = img.convert("RGB")
                frames.append(img)
        return frames

    def _build_per_frame_modality_positions(
        self, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Build [num_frames, 2] modality_positions, one span per frame.

        Finds the boi marker and slices the visual block into N equal frame
        sub-spans. This is what enables the AR mask (per-frame bidirectional +
        cross-frame causal).
        """
        boi_positions = (text_tokens == self.boi_id).nonzero(as_tuple=True)[0]
        if len(boi_positions) == 0:
            # No BOI found — return null spans
            return torch.full((self.num_frames, 2), -1, dtype=torch.long)
        first_frame_offset = int(boi_positions[0].item()) + 1
        positions = []
        for t_idx in range(self.num_frames):
            offset = first_frame_offset + t_idx * self.num_tokens_per_frame_with_meta
            positions.append([offset, self.num_tokens_per_frame_with_meta])
        return torch.tensor(positions, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        # 1. Load frames -> [C, T, H, W]
        frames = self._load_frames(record[self.video_field])
        frame_tensors = [self.image_transform(f) for f in frames]
        video_tensor = torch.stack(frame_tensors).permute(1, 0, 2, 3)

        # 2. Tokenize caption
        caption = record.get(self.text_field, "")
        text_token_ids = self.tokenizer(
            caption, add_special_tokens=False, truncation=True,
            max_length=self.max_text_length,
        )["input_ids"]

        reserve = 4 + self.total_num_visual_tokens
        max_text = max(0, self.max_text_length - reserve)
        if len(text_token_ids) > max_text:
            text_token_ids = text_token_ids[:max_text]

        # 3. Format unified sequence — pass num_image_tokens = total for now,
        # then re-derive per-frame spans below if requested.
        tt, tl, mp, tm, im = format_sequence_gen_qwen2_5(
            text_tokens=text_token_ids,
            system_tokens=None,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=self.total_num_visual_tokens,
            max_seq_len=self.max_text_length,
            system_token_len=0,
        )

        # 4. Optionally rebuild modality_positions as per-frame spans
        if self.per_frame_spans and self.num_frames > 1:
            mp = self._build_per_frame_modality_positions(tt.long())

        return {
            "images": video_tensor,
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": self.data_type,
            "sentence": caption,
            # video AR metadata for the pipeline / loss
            "num_frames": self.num_frames,
            "patch_size": self.patch_size,
        }
