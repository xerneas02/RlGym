from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_project_root

PROJECT_ROOT = bootstrap_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive 2D editor for Rocket League starting scenarios.")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=PROJECT_ROOT / "configs" / "scenarios" / "default_setup.json",
        help="Scenario JSON to load or create.",
    )
    args = parser.parse_args()

    from src.tools.scenario_editor import run_editor

    run_editor(args.scenario)


if __name__ == "__main__":
    main()
