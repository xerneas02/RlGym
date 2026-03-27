from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap_project_root

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exporte un checkpoint PPO vers un package policy simple pour RLBot."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint PPO source (step_*.pt ou final.pt). Si absent, prend le dernier global.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../ZZeer/rlbot_policy.pt"),
        help="Fichier policy exporte pour RLBot.",
    )
    return parser.parse_args()


def main() -> None:
    from src.env.actions import OptimizedDiscreteAction
    from src.env.observations import CompactObservationBuilder
    from src.utils.config_loader import load_project_configs

    args = parse_args()
    config, _, _ = load_project_configs(PROJECT_ROOT)
    checkpoints_root = PROJECT_ROOT / config["paths"]["checkpoints_dir"]
    checkpoint_path = args.checkpoint or find_latest_global(checkpoints_root)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
    output_path = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output).resolve()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Invalid checkpoint format (missing model_state_dict): {checkpoint_path}")

    parser = OptimizedDiscreteAction()
    export_payload = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint_path),
        "obs_dim": int(CompactObservationBuilder.OBSERVATION_DIM),
        "action_dim": int(parser.summary.size),
        "lookup_table": parser.lookup_table.astype("float32"),
        "model_state_dict": checkpoint["model_state_dict"],
        "training_state": checkpoint.get("training_state", {}),
        "config": checkpoint.get("config", {}),
        "seed": checkpoint.get("seed", None),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_payload, output_path)
    print(f"[export] source checkpoint: {checkpoint_path}")
    print(f"[export] target file: {output_path}")
    print(f"[export] obs_dim={export_payload['obs_dim']} action_dim={export_payload['action_dim']}")


if __name__ == "__main__":
    main()
