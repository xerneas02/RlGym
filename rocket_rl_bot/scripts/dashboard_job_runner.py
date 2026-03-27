from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import bootstrap_project_root, ensure_rocketsim_arena_ready, ensure_rocketsim_available

PROJECT_ROOT = bootstrap_project_root()


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_dashboard_meta(run_name: str, payload: dict[str, Any]) -> None:
    log_dir = PROJECT_ROOT / "logs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / "dashboard_meta.json"
    existing = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(
        {
            "display_name": payload.get("display_name", run_name),
            "notes": payload.get("notes", ""),
            "mode": payload.get("mode", "custom"),
            "run_type": payload.get("run_type", "ppo"),
            "dashboard_payload": payload,
        }
    )
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train(payload: dict[str, Any]) -> None:
    from src.rl.trainer import PPOTrainer
    from src.utils.config_loader import load_project_configs

    ensure_rocketsim_available()
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    config, rewards, curriculum = load_project_configs(PROJECT_ROOT, inline_overrides=payload.get("bundle"))
    trainer = PPOTrainer(PROJECT_ROOT, config, rewards, curriculum)
    _write_dashboard_meta(
        trainer.directories.run_name,
        {
            **payload,
            "run_type": "ppo",
            "resolved_run_name": trainer.directories.run_name,
        },
    )
    print(f"[dashboard][train] run={trainer.directories.run_name}", flush=True)
    trainer.train()


def _collect_replay_files(replay_dirs: list[Path], max_replays: int | None, validation_replays: int, seed: int) -> tuple[list[Path], list[Path]]:
    replay_files: list[Path] = []
    seen: set[Path] = set()
    for replay_dir in replay_dirs:
        for replay_path in sorted(replay_dir.glob("*.replay")):
            resolved = replay_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            replay_files.append(replay_path)
    if not replay_files:
        raise FileNotFoundError(f"No replay files found in: {', '.join(str(path) for path in replay_dirs)}")
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(replay_files)
    if max_replays is not None and max_replays > 0:
        replay_files = replay_files[:max_replays]
    validation_count = max(0, min(int(validation_replays), len(replay_files) // 5, len(replay_files) - 1))
    if validation_count > 0:
        return replay_files[:-validation_count], replay_files[-validation_count:]
    return replay_files, []


def run_pretrain(payload: dict[str, Any]) -> None:
    from src.replays.behavior_cloning import run_behavior_cloning
    from src.utils.config_loader import load_project_configs
    from src.utils.seeding import set_global_seeds

    bundle = copy.deepcopy(payload.get("bundle") or {})
    bundle.setdefault("config", {})
    bundle["config"].setdefault("environment", {})
    bundle["config"].setdefault("project", {})
    if payload.get("state_timeout_override") not in (None, ""):
        bundle["config"]["environment"]["timeout_steps"] = int(payload["state_timeout_override"])
    if payload.get("team_size") not in (None, ""):
        bundle["config"]["environment"]["team_size"] = int(payload["team_size"])
    if payload.get("run_name_prefix"):
        bundle["config"]["project"]["run_name_prefix"] = str(payload["run_name_prefix"])
    if payload.get("device"):
        bundle["config"]["project"]["device"] = str(payload["device"])

    config, _rewards, _curriculum = load_project_configs(PROJECT_ROOT, inline_overrides=bundle)
    run_config = copy.deepcopy(config)
    requested_prefix = str(payload.get("run_name_prefix", run_config["project"].get("run_name_prefix", "bc_replay_dashboard")))
    run_config["project"]["run_name_prefix"] = requested_prefix
    if payload.get("device"):
        run_config["project"]["device"] = payload["device"]
    seed = set_global_seeds(int(run_config["project"].get("seed", 42)), deterministic_torch=bool(run_config["runtime"].get("deterministic_torch", False)))
    torch.set_num_threads(int(run_config["runtime"].get("torch_num_threads", 1)))
    device = _resolve_device(str(payload.get("device", run_config["project"].get("device", "auto"))))

    replay_dirs = [Path(path).resolve() for path in payload.get("replay_dirs", []) if Path(path).exists()]
    if not replay_dirs:
        fallback = [PROJECT_ROOT / "replays_ssl_like", PROJECT_ROOT / "xern_replays"]
        replay_dirs = [path.resolve() for path in fallback if path.exists()]
    if not replay_dirs:
        raise FileNotFoundError("No replay directories available for behavior cloning")

    validation_replays = int(payload.get("validation_replays", 50))
    max_replays = payload.get("max_replays")
    train_files, validation_files = _collect_replay_files(replay_dirs, max_replays, validation_replays, seed)
    cache_dir = PROJECT_ROOT / str(payload.get("cache_dir", "replay_bc_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dashboard][pretrain] run_prefix={requested_prefix} train_replays={len(train_files)} validation={len(validation_files)}", flush=True)
    final_checkpoint = run_behavior_cloning(
        project_root=PROJECT_ROOT,
        run_config=run_config,
        replay_files=train_files,
        validation_files=validation_files,
        cache_dir=cache_dir,
        device=device,
        learning_rate=float(payload.get("learning_rate", 3e-4)),
        weight_decay=float(payload.get("weight_decay", 1e-6)),
        batch_size=int(payload.get("batch_size", 4096)),
        epochs=int(payload.get("epochs", 1)),
        sample_fps=float(payload.get("sample_fps", 2.0)),
        max_samples_per_replay=int(payload.get("max_samples_per_replay", 1200)) if payload.get("max_samples_per_replay") is not None else None,
        log_every_replays=int(payload.get("log_every_replays", 25)),
        checkpoint_every_replays=int(payload.get("checkpoint_every_replays", 250)),
        suppress_parser_output=not bool(payload.get("verbose_parser", False)),
        resume_checkpoint=Path(payload["resume_checkpoint"]).resolve() if payload.get("resume_checkpoint") else None,
    )
    run_name = final_checkpoint.parent.name
    _write_dashboard_meta(
        run_name,
        {
            **payload,
            "run_type": "behavior_cloning",
            "resolved_run_name": run_name,
        },
    )
    print(f"[dashboard][pretrain] final_checkpoint={final_checkpoint}", flush=True)


def run_evaluate(payload: dict[str, Any]) -> None:
    from src.dashboard.runs import resolve_checkpoint_path
    from src.rl.evaluator import evaluate_checkpoint
    from src.utils.config_loader import load_project_configs

    ensure_rocketsim_available()
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    config, rewards, curriculum = load_project_configs(PROJECT_ROOT, inline_overrides=payload.get("bundle"))
    checkpoint_path = Path(payload["checkpoint_path"]).resolve() if payload.get("checkpoint_path") else resolve_checkpoint_path(PROJECT_ROOT, payload["run_name"], payload.get("checkpoint_name"))
    device = _resolve_device(str(payload.get("device", config["project"].get("device", "auto"))))
    metrics = evaluate_checkpoint(
        checkpoint_path,
        config["environment"],
        rewards,
        curriculum,
        config["evaluation"],
        device,
        int(payload.get("matches", config["evaluation"].get("num_matches", 16))),
        opponent_mode=payload.get("opponent") or None,
        render_2d=bool(payload.get("render_2d", True)),
        render_fps=int(payload.get("fps", 15)),
        save_trajectory_path=Path(payload["save_trajectory_path"]).resolve() if payload.get("save_trajectory_path") else None,
        project_root=PROJECT_ROOT,
    )
    print(json.dumps(metrics, indent=2), flush=True)


def run_export(payload: dict[str, Any]) -> None:
    from src.dashboard.runs import resolve_checkpoint_path
    from src.env.actions import OptimizedDiscreteAction
    from src.env.observations import CompactObservationBuilder

    checkpoint_path = Path(payload["checkpoint_path"]).resolve() if payload.get("checkpoint_path") else resolve_checkpoint_path(PROJECT_ROOT, payload["run_name"], payload.get("checkpoint_name"))
    output_path = Path(payload.get("output_path") or (PROJECT_ROOT / ".." / "ZZeer" / "rlbot_policy.pt")).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    parser = OptimizedDiscreteAction()
    export_payload = {
        "format_version": 1,
        "source_checkpoint": str(checkpoint_path),
        "display_name": str(payload.get("display_name", checkpoint_path.parent.name)),
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
    print(f"[dashboard][export] source={checkpoint_path}", flush=True)
    print(f"[dashboard][export] output={output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashboard background job runner")
    parser.add_argument("kind", choices=("train", "pretrain", "evaluate", "export"))
    parser.add_argument("--payload", required=True, type=Path)
    args = parser.parse_args()
    payload = _load_payload(args.payload)
    if args.kind == "train":
        run_train(payload)
    elif args.kind == "pretrain":
        run_pretrain(payload)
    elif args.kind == "evaluate":
        run_evaluate(payload)
    else:
        run_export(payload)
