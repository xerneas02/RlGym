from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_latest_global(checkpoints_root: Path) -> Path:
    from src.utils.checkpointing import find_latest_checkpoint

    latest = None
    for run_dir in sorted([path for path in checkpoints_root.iterdir() if path.is_dir()]):
        candidate = find_latest_checkpoint(run_dir)
        if candidate is not None:
            latest = candidate
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoints_root}")
    return latest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    ensure_rocketsim_available()
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    from src.rl.trainer import PPOTrainer

    config = load_yaml(PROJECT_ROOT / "configs" / "training.yaml")
    rewards = load_yaml(PROJECT_ROOT / "configs" / "rewards.yaml")
    curriculum = load_yaml(PROJECT_ROOT / "configs" / "curriculum.yaml")
    checkpoint = args.checkpoint or find_latest_global(PROJECT_ROOT / config["paths"]["checkpoints_dir"])
    trainer = PPOTrainer(PROJECT_ROOT, config, rewards, curriculum, resume_checkpoint=checkpoint)
    trainer.train()


if __name__ == "__main__":
    main()
