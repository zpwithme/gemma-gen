#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Merge a FSDP-sharded checkpoint into a single-file ``model.pt`` for inference.

Background
----------
``LocalCheckpointCallback`` supports two save formats:

* ``"torch"``: a single ``step_<N>/model.pt`` written via ``torch.save``.
  This is safe ONLY when FSDP uses ``FULL_STATE_DICT`` (rank 0 holds the
  full model after gather). With ``SHARDED_STATE_DICT`` the saved file is
  incomplete — rank 0 only sees its own shard.

* ``"dcp"``: a directory of distributed checkpoint shards
  (``__0_0.distcp``, ``__1_0.distcp``, ..., plus ``.metadata`` and an
  optional ``meta.pt`` with ``step``/lr_scheduler). All ranks participate
  in the save; the resulting directory is REQUIRED for ``SHARDED_STATE_DICT``.

This script reads a DCP directory (or detects an already-single-file ckpt)
and writes a single ``model.pt`` whose contents are compatible with
``tuna.inference.checkpoint_loader.load_checkpoint`` — i.e. a dict with
``"model"`` (state_dict), ``"step"``, and optionally ``"lr_scheduler"``.

Usage
-----
::

    # Convert a DCP-format checkpoint to a single .pt
    python -m tuna.scripts.merge_fsdp_ckpt \\
        --src ./outputs/train_gemma/checkpoints/step_50000 \\
        --dst ./outputs/train_gemma/merged/step_50000.pt

    # Or to safetensors (model-only, no optimizer)
    python -m tuna.scripts.merge_fsdp_ckpt \\
        --src ./outputs/train_gemma/checkpoints/step_50000 \\
        --dst ./outputs/train_gemma/merged/step_50000.safetensors \\
        --format safetensors

The output file can then be passed directly to inference::

    python -m tuna.scripts.predict --config-name t2v_pixel_gemma \\
        inference.ckpt_path=./outputs/train_gemma/merged/step_50000.pt

This script runs on a SINGLE machine (no distributed launch needed). It does
NOT require GPU — pure CPU is fine for the merge.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import torch


logger: logging.Logger = logging.getLogger(__name__)


def _is_dcp_dir(path: str) -> bool:
    """A DCP-format ckpt is a directory containing ``.metadata`` and
    one or more ``*.distcp`` shard files."""
    if not os.path.isdir(path):
        return False
    contents = set(os.listdir(path))
    has_metadata = ".metadata" in contents
    has_distcp = any(name.endswith(".distcp") for name in contents)
    return has_metadata and has_distcp


def _load_dcp(src: str) -> Dict[str, Any]:
    """Load a distributed-checkpoint directory into a full state dict.

    Uses the modern ``torch.distributed.checkpoint`` API. Tensors are
    materialized on CPU.
    """
    import torch.distributed.checkpoint as dcp

    # Read the saved metadata to discover the keys + shapes.
    storage_reader = dcp.FileSystemReader(src)
    metadata = storage_reader.read_metadata()

    # Build empty target tensors with the right shapes/dtypes, then let
    # DCP fill them in. This is the offline (no process group) read pattern.
    empty_state: Dict[str, Any] = {}

    def _materialize(planner_dict, target_dict):
        for fqn, md in planner_dict.items():
            if hasattr(md, "size"):
                target_dict[fqn] = torch.empty(
                    md.size, dtype=md.properties.dtype
                )
            else:
                # Non-tensor (e.g. int / dict). Use a placeholder; DCP will
                # overwrite via the planner.
                target_dict[fqn] = None

    # metadata.state_dict_metadata maps full fqns ("model.module.weight" etc).
    _materialize(metadata.state_dict_metadata, empty_state)

    # DCP's load_state_dict mutates in-place. We need to provide a state_dict
    # whose structure matches what was saved — re-create the {"model": ...,
    # "optimizer": ...} top level.
    def _rebuild_hierarchy(flat: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for fqn, val in flat.items():
            head, _, rest = fqn.partition(".")
            if rest:
                out.setdefault(head, {})[rest] = val
            else:
                out[head] = val
        return out

    state = _rebuild_hierarchy(empty_state)

    # Use the no-pg loader (works without `torch.distributed.init_process_group`)
    try:
        dcp.load(state_dict=state, storage_reader=storage_reader)
    except AttributeError:
        # Older API
        dcp.load_state_dict(state_dict=state, storage_reader=storage_reader)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load DCP from {src}. Make sure the directory contains "
            f"a valid .metadata and *.distcp shards. Original error: {e}"
        )

    return state


def _load_torch(src: str) -> Dict[str, Any]:
    """Load a legacy single-file checkpoint."""
    obj = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected checkpoint payload type at {src}: {type(obj)}")


def _load_meta_companion(src: str) -> Dict[str, Any]:
    """Pick up the ``meta.pt`` companion stored next to the DCP shards
    (contains ``step`` and lr_scheduler state)."""
    meta_path = os.path.join(src, "meta.pt")
    if os.path.isfile(meta_path):
        try:
            return torch.load(meta_path, map_location="cpu", weights_only=True)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not load meta companion {meta_path}: {e}")
    return {}


def merge(src: str, dst: str, fmt: str = "torch", include_optimizer: bool = False) -> None:
    """Merge a DCP or single-file checkpoint into a single output file.

    Args:
        src: Source path. Either a DCP directory (``step_<N>/``) or a single
            ``.pt``/``.safetensors`` file.
        dst: Output file path. Use ``.pt`` for torch.save format,
            ``.safetensors`` for safetensors (set ``fmt='safetensors'``).
        fmt: Output format, ``"torch"`` or ``"safetensors"``.
        include_optimizer: If True and source contains optimizer state,
            include it in the output (torch format only).
    """
    if os.path.isdir(src) and _is_dcp_dir(src):
        logger.info(f"Detected DCP-format checkpoint at {src}; merging shards...")
        state = _load_dcp(src)
        meta = _load_meta_companion(src)
    elif os.path.isfile(src):
        logger.info(f"Loading single-file checkpoint from {src}")
        state = _load_torch(src)
        meta = {}
        # If the user pointed at a step directory's model.pt, also look for a
        # sibling meta.pt.
        meta_sibling = os.path.join(os.path.dirname(src), "meta.pt")
        if os.path.isfile(meta_sibling):
            try:
                meta = torch.load(meta_sibling, map_location="cpu", weights_only=True)
            except Exception:  # pragma: no cover
                pass
    else:
        raise FileNotFoundError(
            f"src must be a DCP directory or a checkpoint file: {src}"
        )

    # Normalize: pull "model" out, drop the rest unless include_optimizer.
    if "model" in state and isinstance(state["model"], dict):
        model_state = state["model"]
    else:
        # Source already a flat state dict
        model_state = state

    payload: Dict[str, Any] = {"model": model_state}
    if "step" in meta:
        payload["step"] = int(meta["step"])
    if include_optimizer and "optimizer" in state:
        payload["optimizer"] = state["optimizer"]
        logger.info("Including optimizer state in output.")

    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)

    if fmt == "torch":
        torch.save(payload, dst)
        logger.info(f"Saved merged checkpoint → {dst}")
    elif fmt == "safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as e:
            raise RuntimeError(
                "safetensors not installed; pip install safetensors"
            ) from e
        cpu_state = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model_state.items()
        }
        # safetensors only stores tensors; warn if non-tensors get dropped.
        non_tensor = [k for k, v in cpu_state.items() if not isinstance(v, torch.Tensor)]
        if non_tensor:
            logger.warning(
                f"safetensors will drop {len(non_tensor)} non-tensor entries: "
                f"{non_tensor[:5]}{'...' if len(non_tensor) > 5 else ''}"
            )
            cpu_state = {k: v for k, v in cpu_state.items() if isinstance(v, torch.Tensor)}
        save_file(cpu_state, dst)
        logger.info(f"Saved merged checkpoint (safetensors) → {dst}")
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Choose 'torch' or 'safetensors'.")

    # Quick stat for sanity
    n_params = sum(v.numel() for v in model_state.values() if isinstance(v, torch.Tensor))
    n_keys = sum(1 for v in model_state.values() if isinstance(v, torch.Tensor))
    logger.info(
        f"Merged checkpoint contains {n_keys} tensors, "
        f"{n_params / 1e9:.2f}B params total."
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge a FSDP-sharded DCP checkpoint into a single file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src", required=True,
        help="Source path. Either a DCP directory (e.g. step_50000/) or a single .pt file."
    )
    p.add_argument(
        "--dst", required=True,
        help="Output file path. Use .pt for torch format, .safetensors for safetensors."
    )
    p.add_argument(
        "--format", default="torch", choices=["torch", "safetensors"],
        help="Output format."
    )
    p.add_argument(
        "--include-optimizer", action="store_true",
        help="Include optimizer state in output (torch format only). "
             "Usually NOT needed for inference."
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging."
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    try:
        merge(
            src=args.src,
            dst=args.dst,
            fmt=args.format,
            include_optimizer=args.include_optimizer,
        )
    except Exception as e:
        logger.error(f"Merge failed: {e}", exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
