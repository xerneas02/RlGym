from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--matches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-2d", action="store_true", help="Affiche une vue simplifiee du match en evaluation.")
    parser.add_argument("--fps", type=int, default=15, help="FPS cible pour le viewer 2D ou le replay.")
    parser.add_argument("--save-trajectory", type=Path, default=None, help="Sauvegarde les frames de l'evaluation en JSON.")
    args = parser.parse_args()

    ensure_rocketsim_available()
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    from src.rl.evaluator import evaluate_checkpoint

    config = load_yaml(PROJECT_ROOT / "configs" / "training.yaml")
    rewards = load_yaml(PROJECT_ROOT / "configs" / "rewards.yaml")
    curriculum = load_yaml(PROJECT_ROOT / "configs" / "curriculum.yaml")
    num_matches = int(args.matches or config["evaluation"]["num_matches"])
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    metrics = evaluate_checkpoint(
        args.checkpoint,
        config["environment"],
        rewards,
        curriculum,
        device,
        num_matches,
        render_2d=args.render_2d,
        render_fps=int(args.fps),
        save_trajectory_path=args.save_trajectory,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
