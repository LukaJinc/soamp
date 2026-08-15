"""Checkpointer: owns save/load I/O only, implementing CLAUDE.md sec 4's
exact naming scheme:

  <exp_id>_<iter>.pth              -- weights only
  <exp_id>_best.pth                -- weights only
  <exp_id>_checkpoint_<iter>.pth   -- full state (model + optimizer)

Metadata (resolved config + git SHA + seed) is embedded directly in the
.pth dict -- a single file is self-describing, no sidecar needed.
"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CheckpointMetadata:
    exp_id: str
    iter: int
    git_sha: str
    seed: int
    resolved_config: dict[str, Any]


class Checkpointer:
    def __init__(self, checkpoint_dir: Path, exp_id: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.exp_id = exp_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _weights_path(self, iter: int, best: bool) -> Path:
        name = f"{self.exp_id}_best.pth" if best else f"{self.exp_id}_{iter}.pth"
        return self.checkpoint_dir / name

    def _full_state_path(self, iter: int) -> Path:
        return self.checkpoint_dir / f"{self.exp_id}_checkpoint_{iter}.pth"

    def save_weights(
        self, model: nn.Module, iter: int, metadata: CheckpointMetadata, best: bool = False
    ) -> Path:
        path = self._weights_path(iter, best)
        torch.save(
            {"model_state_dict": model.state_dict(), "metadata": dataclasses.asdict(metadata)},
            path,
        )
        return path

    def save_full_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        iter: int,
        metadata: CheckpointMetadata,
    ) -> Path:
        path = self._full_state_path(iter)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metadata": dataclasses.asdict(metadata),
            },
            path,
        )
        return path

    @staticmethod
    def load_weights(path: Path, model: nn.Module) -> CheckpointMetadata:
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return CheckpointMetadata(**checkpoint["metadata"])

    @staticmethod
    def load_full_state(
        path: Path, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> CheckpointMetadata:
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return CheckpointMetadata(**checkpoint["metadata"])
