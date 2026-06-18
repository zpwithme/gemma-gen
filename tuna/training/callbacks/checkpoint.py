# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Local-filesystem checkpoint callback.

Local checkpoint saving callback for Tuna training.
Writes ``model_step_<N>.pt`` (or
``.safetensors``) under a local directory and prunes old checkpoints to keep
the most recent ``keep_last`` files.
"""

from __future__ import annotations

# pyre-unsafe

import logging
import os
import re
from typing import Any

import torch
import torch.distributed as dist
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank


logger: logging.Logger = logging.getLogger(__name__)

try:
    from safetensors.torch import save_file as _safetensors_save_file
except Exception:  # pragma: no cover - safetensors is in requirements but be safe
    _safetensors_save_file = None  # type: ignore[assignment]


class LocalCheckpointCallback(Callback):
    """Save model + optimizer state to a local directory every ``N`` steps.

    Two storage formats are supported (selected via ``save_format``):

      * ``"torch"`` (default, legacy): writes ``step_<N>/model.pt`` via
        ``torch.save``. **Only safe when FSDP uses ``FULL_STATE_DICT``** —
        otherwise rank 0 only sees its own shard and the file is incomplete.
      * ``"dcp"``: writes ``step_<N>/`` as a distributed checkpoint via
        ``torch.distributed.checkpoint``. **Required for ``SHARDED_STATE_DICT``**.
        Use ``tuna.scripts.merge_fsdp_ckpt`` to flatten the DCP directory into
        a single ``model.pt`` for inference.

    Args:
        checkpoint_dir: Directory under which to write ``step_<N>/...``.
        save_every_n_steps: Save frequency. ``<=0`` disables periodic saves.
        keep_last: Maximum historical checkpoints to keep on disk.
        use_safetensors: If ``True`` (and ``save_format='torch'``), write
            ``model.safetensors`` (no optimizer state).
        save_optimizer: Whether to include optimizer / lr_scheduler state.
        save_format: ``"torch"`` (single .pt per save) or ``"dcp"`` (distributed
            checkpoint dir). Auto-pick ``"dcp"`` if you set FSDP to
            ``SHARDED_STATE_DICT``.
    """

    _STEP_RE = re.compile(r"^step_(\d+)$")

    def __init__(
        self,
        checkpoint_dir: str,
        save_every_n_steps: int = 1000,
        keep_last: int = 3,
        use_safetensors: bool = False,
        save_optimizer: bool = True,
        save_format: str = "torch",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = int(save_every_n_steps)
        self.keep_last = int(keep_last)
        self.use_safetensors = use_safetensors
        self.save_optimizer = save_optimizer
        if save_format not in ("torch", "dcp"):
            raise ValueError(
                f"Unknown save_format: {save_format!r}. Choose 'torch' or 'dcp'."
            )
        self.save_format = save_format

        if get_global_rank() == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ---- TorchTNT hooks ----------------------------------------------------
    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        if self.save_every_n_steps <= 0:
            return
        step = unit.train_progress.num_steps_completed
        if step == 0 or step % self.save_every_n_steps != 0:
            return
        self._save(unit, step)

    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        self._save(unit, step)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        step = unit.train_progress.num_steps_completed
        self._save(unit, step)

    # ---- Internals ---------------------------------------------------------
    def _save(self, unit: TTrainUnit, step: int) -> None:
        # FSDP shard collection happens collectively; gather across all ranks
        # via the AutoUnit's state_dict (SHARDED_STATE_DICT vs FULL_STATE_DICT
        # is decided by the FSDPStrategy on the unit).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        rank = get_global_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step}")
        if rank == 0:
            os.makedirs(step_dir, exist_ok=True)

        module = getattr(unit, "module", None)
        if module is None:
            logger.warning("LocalCheckpointCallback: unit has no .module; skipping.")
            return

        # DCP path — write a distributed checkpoint directory; every rank
        # participates. Use this when FSDP state_dict_type=SHARDED_STATE_DICT.
        if self.save_format == "dcp":
            self._save_dcp(unit, module, step_dir, rank, step)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return

        # Legacy torch.save path (rank 0 only). Only safe with FULL_STATE_DICT.
        try:
            state_dict = module.state_dict()
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to gather state_dict for save: {e}")
            return

        if rank == 0:
            if self.use_safetensors and _safetensors_save_file is not None:
                cpu_state = {
                    k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in state_dict.items()
                }
                _safetensors_save_file(cpu_state, os.path.join(step_dir, "model.safetensors"))
            else:
                payload: dict[str, Any] = {"model": state_dict, "step": step}
                if self.save_optimizer:
                    optimizer = getattr(unit, "optimizer", None)
                    if optimizer is not None:
                        try:
                            payload["optimizer"] = optimizer.state_dict()
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Could not save optimizer state: {e}")
                    lr_scheduler = getattr(unit, "lr_scheduler", None)
                    if lr_scheduler is not None:
                        try:
                            payload["lr_scheduler"] = lr_scheduler.state_dict()
                        except Exception:  # pragma: no cover
                            pass
                torch.save(payload, os.path.join(step_dir, "model.pt"))
            logger.info(f"Saved checkpoint to {step_dir}")
            self._prune_old()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _save_dcp(self, unit, module, step_dir: str, rank: int, step: int) -> None:
        """Distributed checkpoint save — every rank writes its own shard files
        into ``step_dir/`` via ``torch.distributed.checkpoint``.

        Output layout::

            step_dir/
                .metadata
                __0_0.distcp
                __1_0.distcp
                ...
                meta.pt                # rank-0-only: step + optimizer scaffold

        Convert to a single-file ``model.pt`` for inference using
        ``python -m tuna.scripts.merge_fsdp_ckpt --src <step_dir> --dst out.pt``.
        """
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError as e:  # pragma: no cover
            logger.warning(
                f"DCP not available ({e}); falling back to torch.save (may be incomplete with sharded FSDP)."
            )
            self.save_format = "torch"
            return self._save(unit, step)

        state = {"model": module.state_dict()}
        if self.save_optimizer:
            optimizer = getattr(unit, "optimizer", None)
            if optimizer is not None:
                try:
                    state["optimizer"] = optimizer.state_dict()
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Could not include optimizer in DCP save: {e}")

        storage_writer = dcp.FileSystemWriter(step_dir)
        try:
            # Newer API: `dcp.save` accepts state dict directly.
            dcp.save(state_dict=state, storage_writer=storage_writer)
        except AttributeError:  # pragma: no cover
            # Older API fallback
            dcp.save_state_dict(state_dict=state, storage_writer=storage_writer)

        if rank == 0:
            # Stash step number + lr_scheduler in a small companion file.
            meta: dict[str, Any] = {"step": step}
            lr_scheduler = getattr(unit, "lr_scheduler", None)
            if lr_scheduler is not None:
                try:
                    meta["lr_scheduler"] = lr_scheduler.state_dict()
                except Exception:  # pragma: no cover
                    pass
            torch.save(meta, os.path.join(step_dir, "meta.pt"))
            logger.info(f"Saved DCP checkpoint to {step_dir}")
            self._prune_old()

    def _prune_old(self) -> None:
        if self.keep_last <= 0:
            return
        try:
            entries = os.listdir(self.checkpoint_dir)
        except FileNotFoundError:
            return
        kept = []
        for name in entries:
            m = self._STEP_RE.match(name)
            if m is not None:
                kept.append((int(m.group(1)), name))
        kept.sort(reverse=True)
        for _, name in kept[self.keep_last:]:
            path = os.path.join(self.checkpoint_dir, name)
            try:
                for root, _dirs, files in os.walk(path, topdown=False):
                    for f in files:
                        os.remove(os.path.join(root, f))
                    os.rmdir(root)
                logger.info(f"Pruned old checkpoint: {path}")
            except OSError as e:
                logger.warning(f"Could not prune {path}: {e}")
