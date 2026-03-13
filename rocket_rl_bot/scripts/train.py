from __future__ import annotations

from pathlib import Path

import yaml

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    ensure_rocketsim_available()
    print("[preflight] RocketSim installe, verification de l'arena...", flush=True)
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    print("[preflight] Arena RocketSim OK", flush=True)
    from src.rl.trainer import PPOTrainer

    config = load_yaml(PROJECT_ROOT / "configs" / "training.yaml")
    rewards = load_yaml(PROJECT_ROOT / "configs" / "rewards.yaml")
    curriculum = load_yaml(PROJECT_ROOT / "configs" / "curriculum.yaml")
    trainer = PPOTrainer(PROJECT_ROOT, config, rewards, curriculum)
    trainer.train()


if __name__ == "__main__":
    main()
