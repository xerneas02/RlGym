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

    training = pd.read_csv(args.run_dir / "training_metrics.csv")

    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    training.plot(x="step", y="episode_reward", ax=axes[0], title="Episode Reward")
    training.plot(x="step", y="goal_rate", ax=axes[1], title="Goal Rate")
    training.plot(x="step", y="ball_touches", ax=axes[2], title="Touch Rate")
    training.plot(x="step", y="training_loss", ax=axes[3], title="Training Loss")
    training.plot(x="step", y="entropy", ax=axes[4], title="Entropy")
    axes[4].set_xlabel("Training steps")
    fig.tight_layout()

    output = args.output or (args.run_dir / "training_curves.png")
    fig.savefig(output, dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
