from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_state: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": training_state,
        "config": config,
        "seed": int(seed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates = sorted(checkpoint_dir.glob("step_*.pt"))
    return candidates[-1] if candidates else None
