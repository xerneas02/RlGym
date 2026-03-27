from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


def find_latest_matching_checkpoint(checkpoints_root: Path, run_prefixes: str | tuple[str, ...]) -> Path | None:
    prefixes = (run_prefixes,) if isinstance(run_prefixes, str) else run_prefixes
    if not checkpoints_root.exists():
        return None

    run_dirs = sorted(
        [path for path in checkpoints_root.iterdir() if path.is_dir() and path.name.startswith(prefixes)],
        reverse=True,
    )
    for run_dir in run_dirs:
        final_checkpoint = run_dir / "final.pt"
        if final_checkpoint.exists():
            return final_checkpoint

        step_checkpoints = sorted(run_dir.glob("step_*.pt"))
        if step_checkpoints:
            return step_checkpoints[-1]
    return None


def find_latest_bc_checkpoint(checkpoints_root: Path) -> Path | None:
    if not checkpoints_root.exists():
        return None

    run_dirs = sorted(
        [path for path in checkpoints_root.iterdir() if path.is_dir() and path.name.startswith("bc_")],
        reverse=True,
    )
    for run_dir in run_dirs:
        final_checkpoint = run_dir / "final.pt"
        if final_checkpoint.exists():
            return final_checkpoint

        step_checkpoints = sorted(run_dir.glob("step_*.pt"))
        if step_checkpoints:
            return step_checkpoints[-1]
    return None


def resolve_checkpoint_path(candidate: Path | None) -> Path | None:
    if candidate is None:
        return None
    checkpoint_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance un self-play PPO focalise d'abord sur toucher la balle puis la pousser vers le but adverse."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint replay-only a charger comme bootstrap de policy.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint PPO touch-focus a reprendre integralement avec optimizer et compteur de steps.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Demarre de zero sans reprise PPO ni bootstrap replay-only.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Ignore la reprise PPO automatique et charge uniquement un checkpoint de bootstrap (ideal apres BC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_rocketsim_available()
    print("[preflight] RocketSim installe, verification de l'arena...", flush=True)
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    print("[preflight] Arena RocketSim OK", flush=True)

    from configs.training_presets import apply_touch_focus_preset
    from src.rl.trainer import PPOTrainer
    from src.utils.checkpointing import load_checkpoint
    from src.utils.config_loader import load_project_configs

    config, rewards, curriculum = load_project_configs(PROJECT_ROOT)
    config, rewards, curriculum = apply_touch_focus_preset(config, rewards, curriculum)
    checkpoints_root = PROJECT_ROOT / config["paths"]["checkpoints_dir"]

    resume_checkpoint = None
    bootstrap_checkpoint = None

    if args.fresh and args.bootstrap_only:
        raise ValueError("--fresh et --bootstrap-only sont incompatibles")

    if args.bootstrap_only:
        bootstrap_checkpoint = resolve_checkpoint_path(args.checkpoint)
        if bootstrap_checkpoint is None:
            bootstrap_checkpoint = find_latest_bc_checkpoint(checkpoints_root)
            bootstrap_checkpoint = resolve_checkpoint_path(bootstrap_checkpoint)
        if bootstrap_checkpoint is None:
            raise FileNotFoundError("Aucun checkpoint BC trouve. Passe --checkpoint explicitement.")
    elif not args.fresh:
        resume_checkpoint = resolve_checkpoint_path(args.resume_checkpoint)
        if resume_checkpoint is None:
            resume_checkpoint = find_latest_matching_checkpoint(
                checkpoints_root,
                ("ppo_goal_finish_v1", "ppo_touch_focus_v3", "ppo_touch_focus_v2", "ppo_touch_focus_v1"),
            )
            resume_checkpoint = resolve_checkpoint_path(resume_checkpoint)

        if resume_checkpoint is None:
            bootstrap_checkpoint = resolve_checkpoint_path(args.checkpoint)
            if bootstrap_checkpoint is None:
                bootstrap_checkpoint = find_latest_bc_checkpoint(checkpoints_root)
                bootstrap_checkpoint = resolve_checkpoint_path(bootstrap_checkpoint)

    trainer = PPOTrainer(PROJECT_ROOT, config, rewards, curriculum, resume_checkpoint=resume_checkpoint)

    if resume_checkpoint is not None:
        print(f"[touch-focus] resumed full PPO state from: {resume_checkpoint}", flush=True)
    elif bootstrap_checkpoint is not None:
        load_checkpoint(bootstrap_checkpoint, trainer.model, optimizer=None, map_location=trainer.device)
        print(f"[touch-focus] loaded bootstrap weights from: {bootstrap_checkpoint}", flush=True)
    else:
        print("[touch-focus] no checkpoint loaded; training will start from scratch.", flush=True)

    print(
        "[touch-focus] reward weights="
        f"{rewards['weights']} | curriculum_start={curriculum['stages'][0]['weights']} "
        f"| entropy_coef={config['training']['entropy_coef']} learning_rate={config['training']['learning_rate']}",
        flush=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()
