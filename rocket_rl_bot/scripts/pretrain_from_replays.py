from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from _bootstrap import bootstrap_project_root

PROJECT_ROOT = bootstrap_project_root()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-entraine la policy par imitation a partir des replays deja telecharges."
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "Dossier de replays a inclure. Peut etre passe plusieurs fois. "
            "Par defaut, utilise replays_ssl_like et xern_replays s'ils existent."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("replay_bc_cache"))
    parser.add_argument("--max-replays", type=int, default=2000)
    parser.add_argument("--validation-replays", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--max-samples-per-replay", type=int, default=1200)
    parser.add_argument("--log-every-replays", type=int, default=25)
    parser.add_argument("--checkpoint-every-replays", type=int, default=250)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-name-prefix", default="bc_replay_1v1")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--verbose-parser",
        action="store_true",
        help="Affiche les logs bruts de carball/boxcars pour le diagnostic.",
    )
    return parser.parse_args()


def resolve_replay_dirs(requested_dirs: list[Path] | None) -> list[Path]:
    if requested_dirs:
        candidate_dirs = requested_dirs
    else:
        candidate_dirs = [Path("replays_ssl_like"), Path("xern_replays")]

    resolved_dirs: list[Path] = []
    seen: set[Path] = set()
    for replay_dir in candidate_dirs:
        absolute_dir = (PROJECT_ROOT / replay_dir).resolve()
        if absolute_dir in seen:
            continue
        seen.add(absolute_dir)
        if absolute_dir.exists():
            resolved_dirs.append(absolute_dir)

    if not resolved_dirs:
        joined = ", ".join(str(PROJECT_ROOT / path) for path in candidate_dirs)
        raise FileNotFoundError(f"No replay directory found among: {joined}")

    return resolved_dirs


def main() -> None:
    from src.replays.behavior_cloning import run_behavior_cloning
    from src.utils.config_loader import load_project_configs
    from src.utils.seeding import set_global_seeds

    args = parse_args()
    config, _, _ = load_project_configs(PROJECT_ROOT)
    run_config = copy.deepcopy(config)
    run_config["project"]["run_name_prefix"] = args.run_name_prefix

    seed = set_global_seeds(
        int(run_config["project"]["seed"]),
        deterministic_torch=bool(run_config["runtime"].get("deterministic_torch", False)),
    )
    torch.set_num_threads(int(run_config["runtime"].get("torch_num_threads", 1)))
    device = resolve_device(run_config["project"].get("device", args.device) if args.device == "auto" else args.device)

    replay_dirs = resolve_replay_dirs(args.replay_dir)
    replay_files: list[Path] = []
    seen_replays: set[Path] = set()
    for replay_dir in replay_dirs:
        for replay_path in sorted(replay_dir.glob("*.replay")):
            resolved_replay = replay_path.resolve()
            if resolved_replay in seen_replays:
                continue
            seen_replays.add(resolved_replay)
            replay_files.append(replay_path)

    if not replay_files:
        joined = ", ".join(str(path) for path in replay_dirs)
        raise FileNotFoundError(f"No replay files found in {joined}")

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(replay_files)

    if args.max_replays is not None and args.max_replays > 0:
        replay_files = replay_files[: args.max_replays]

    validation_replays = max(0, min(args.validation_replays, len(replay_files) // 5, len(replay_files) - 1))
    if validation_replays > 0:
        validation_files = replay_files[-validation_replays:]
        train_files = replay_files[:-validation_replays]
    else:
        validation_files = []
        train_files = replay_files

    cache_dir = PROJECT_ROOT / args.cache_dir if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bc][startup] seed={seed} device={device}")
    print(
        f"[bc][startup] replay_dirs={', '.join(str(path) for path in replay_dirs)} "
        f"train_replays={len(train_files)} validation_replays={len(validation_files)} "
        f"cache_dir={cache_dir}"
    )
    print(
        f"[bc][startup] sample_fps={args.sample_fps} max_samples_per_replay={args.max_samples_per_replay} "
        f"batch_size={args.batch_size} epochs={args.epochs}"
    )

    final_checkpoint = run_behavior_cloning(
        project_root=PROJECT_ROOT,
        run_config=run_config,
        replay_files=train_files,
        validation_files=validation_files,
        cache_dir=cache_dir,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        sample_fps=args.sample_fps,
        max_samples_per_replay=args.max_samples_per_replay,
        log_every_replays=args.log_every_replays,
        checkpoint_every_replays=args.checkpoint_every_replays,
        suppress_parser_output=not args.verbose_parser,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(f"[bc][next] PPO fine-tuning can start from: {final_checkpoint}")


if __name__ == "__main__":
    main()
