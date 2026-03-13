from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rewards = pd.read_csv(args.run_dir / "reward_components.csv")
    columns = [column for column in rewards.columns if column != "step"]

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, max(10, 2.5 * len(columns))), sharex=True)
    if len(columns) == 1:
        axes = [axes]
    for axis, column in zip(axes, columns):
        rewards.plot(x="step", y=column, ax=axis, title=column)
    axes[-1].set_xlabel("Training steps")
    fig.tight_layout()

    output = args.output or (args.run_dir / "reward_analysis.png")
    fig.savefig(output, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
