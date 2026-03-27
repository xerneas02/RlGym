from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            casted: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    casted[key] = value
                    continue
                try:
                    casted[key] = float(value)
                except ValueError:
                    casted[key] = value
            rows.append(casted)
    return rows


def _latest_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return rows[-1] if rows else {}


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "dashboard_meta.json"


def _checkpoint_dir(project_root: Path, run_name: str) -> Path:
    return project_root / "checkpoints" / run_name


def list_runs(project_root: Path) -> list[dict[str, Any]]:
    logs_root = Path(project_root) / "logs"
    if not logs_root.exists():
        return []
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted([path for path in logs_root.iterdir() if path.is_dir()], key=lambda item: item.stat().st_mtime, reverse=True):
        config = _read_json(run_dir / "resolved_config.json", {}) or {}
        meta = _read_json(_run_meta_path(run_dir), {}) or {}
        training_rows = _read_csv_rows(run_dir / "training_metrics.csv")
        evaluation_rows = _read_csv_rows(run_dir / "evaluation_metrics.csv")
        reward_rows = _read_csv_rows(run_dir / "reward_components.csv")
        checkpoint_dir = _checkpoint_dir(project_root, run_dir.name)
        checkpoints = []
        if checkpoint_dir.exists():
            checkpoints = [path.name for path in sorted(checkpoint_dir.glob("*.pt"))]
        env_cfg = dict(config.get("environment", {}))
        team_size = int(env_cfg.get("team_size", 1))
        mode_label = {1: "1v1", 2: "2v2", 3: "3v3"}.get(team_size, f"{team_size}v{team_size}")
        summaries.append(
            {
                "run_name": run_dir.name,
                "display_name": str(meta.get("display_name", run_dir.name)),
                "notes": str(meta.get("notes", "")),
                "mode": str(meta.get("mode", mode_label)),
                "run_type": str(meta.get("run_type", "bc" if run_dir.name.startswith("bc_") else "ppo")),
                "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
                "latest_training": _latest_row(training_rows),
                "latest_evaluation": _latest_row(evaluation_rows),
                "latest_rewards": _latest_row(reward_rows),
                "checkpoint_count": len(checkpoints),
                "checkpoints": checkpoints,
            }
        )
    return summaries


def get_run_detail(project_root: Path, run_name: str) -> dict[str, Any]:
    run_dir = Path(project_root) / "logs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_name}")
    config = _read_json(run_dir / "resolved_config.json", {}) or {}
    meta = _read_json(_run_meta_path(run_dir), {}) or {}
    training_rows = _read_csv_rows(run_dir / "training_metrics.csv")
    evaluation_rows = _read_csv_rows(run_dir / "evaluation_metrics.csv")
    reward_rows = _read_csv_rows(run_dir / "reward_components.csv")
    checkpoint_dir = _checkpoint_dir(project_root, run_name)
    checkpoints = []
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.pt")):
            checkpoints.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": int(path.stat().st_size),
                    "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )
    return {
        "run_name": run_name,
        "display_name": str(meta.get("display_name", run_name)),
        "notes": str(meta.get("notes", "")),
        "meta": meta,
        "resolved_config": config,
        "training_metrics": training_rows,
        "evaluation_metrics": evaluation_rows,
        "reward_components": reward_rows,
        "checkpoints": checkpoints,
        "files": {
            "training_csv": str(run_dir / "training_metrics.csv"),
            "evaluation_csv": str(run_dir / "evaluation_metrics.csv"),
            "reward_csv": str(run_dir / "reward_components.csv"),
            "config": str(run_dir / "resolved_config.json"),
        },
    }


def update_run_meta(project_root: Path, run_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(project_root) / "logs" / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_name}")
    meta_path = _run_meta_path(run_dir)
    meta = _read_json(meta_path, {}) or {}
    meta.update(payload)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def resolve_checkpoint_path(project_root: Path, run_name: str, checkpoint_name: str | None = None) -> Path:
    checkpoint_dir = _checkpoint_dir(project_root, run_name)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found for run: {run_name}")
    if checkpoint_name:
        path = checkpoint_dir / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    candidates = sorted(checkpoint_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    if (checkpoint_dir / "final.pt").exists():
        return checkpoint_dir / "final.pt"
    return candidates[-1]
