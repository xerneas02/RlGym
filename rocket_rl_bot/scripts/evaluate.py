from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


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
    parser.add_argument("--matches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-2d", action="store_true", help="Affiche une vue simplifiee du match en evaluation.")
    parser.add_argument("--fps", type=int, default=15, help="FPS cible pour le viewer 2D ou le replay.")
    parser.add_argument("--save-trajectory", type=Path, default=None, help="Sauvegarde les frames de l'evaluation en JSON.")
    args = parser.parse_args()

    ensure_rocketsim_available()
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    from src.rl.evaluator import evaluate_checkpoint
    from src.utils.config_loader import load_project_configs

    config, rewards, curriculum = load_project_configs(PROJECT_ROOT)
    checkpoint = args.checkpoint or find_latest_global(PROJECT_ROOT / config["paths"]["checkpoints_dir"])
    if args.checkpoint is None:
        print(f"[checkpoint] using latest checkpoint: {checkpoint}")
    num_matches = int(args.matches or config["evaluation"]["num_matches"])
    device = torch.device("cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    metrics = evaluate_checkpoint(
        checkpoint,
        config["environment"],
        rewards,
        curriculum,
        config["evaluation"],
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
