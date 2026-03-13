from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class RunDirectories:
    run_name: str
    root: Path
    log_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path
    training_csv: Path
    evaluation_csv: Path
    rewards_csv: Path
    config_snapshot: Path


class CSVLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fieldnames = None

    def write_row(self, row: Dict[str, object]) -> None:
        fieldnames = list(row.keys())
        write_header = not self.path.exists() or self._fieldnames != fieldnames
        self._fieldnames = fieldnames
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class TrainingLogger:
    def __init__(self, directories: RunDirectories) -> None:
        self.directories = directories
        self.writer = SummaryWriter(log_dir=str(directories.tensorboard_dir))
        self.training_csv = CSVLogger(directories.training_csv)
        self.evaluation_csv = CSVLogger(directories.evaluation_csv)
        self.rewards_csv = CSVLogger(directories.rewards_csv)

    def log_training(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": int(step), **metrics}
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", float(value), int(step))
        self.training_csv.write_row(row)

    def log_evaluation(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": int(step), **metrics}
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", float(value), int(step))
        self.evaluation_csv.write_row(row)

    def log_reward_components(self, step: int, metrics: Dict[str, float]) -> None:
        row = {"step": int(step), **metrics}
        for key, value in metrics.items():
            self.writer.add_scalar(f"reward/{key}", float(value), int(step))
        self.rewards_csv.write_row(row)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def create_run_directories(project_root: Path, config: Dict) -> RunDirectories:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config['project']['run_name_prefix']}_{timestamp}"
    logs_root = project_root / config["paths"]["logs_dir"]
    checkpoints_root = project_root / config["paths"]["checkpoints_dir"]
    run_root = logs_root / run_name
    checkpoint_dir = checkpoints_root / run_name
    tensorboard_dir = run_root / "tensorboard"

    for path in [run_root, checkpoint_dir, tensorboard_dir]:
        path.mkdir(parents=True, exist_ok=True)

    directories = RunDirectories(
        run_name=run_name,
        root=run_root,
        log_dir=run_root,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=tensorboard_dir,
        training_csv=run_root / "training_metrics.csv",
        evaluation_csv=run_root / "evaluation_metrics.csv",
        rewards_csv=run_root / "reward_components.csv",
        config_snapshot=run_root / "resolved_config.json",
    )
    return directories


def snapshot_config(path: Path, config: Dict) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
