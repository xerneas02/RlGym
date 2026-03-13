from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.topdown_viewer import play_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejoue une trajectoire Rocket League simplifiee en vue 2D.")
    parser.add_argument("--trajectory", required=True, type=Path, help="Chemin vers le JSON de trajectoire genere par evaluate.py.")
    parser.add_argument("--fps", type=int, default=15, help="FPS cible du replay.")
    parser.add_argument("--loop", action="store_true", help="Relance le replay en boucle.")
    args = parser.parse_args()

    play_trajectory(args.trajectory, fps=max(1, int(args.fps)), loop=bool(args.loop))


if __name__ == "__main__":
    main()
