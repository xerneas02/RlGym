from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--metric", default="episode_reward")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 6))
    for run_dir in args.run_dirs:
        training = pd.read_csv(run_dir / "training_metrics.csv")
        ax.plot(training["step"], training[args.metric], label=run_dir.name)

    ax.set_title(f"Run comparison: {args.metric}")
    ax.set_xlabel("Training steps")
    ax.set_ylabel(args.metric)
    ax.legend()
    fig.tight_layout()
    output = args.output or Path("compare_runs.png")
    fig.savefig(output, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
