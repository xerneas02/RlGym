from __future__ import annotations

import argparse

from _bootstrap import bootstrap_project_root

PROJECT_ROOT = bootstrap_project_root()


def main() -> None:
    from src.dashboard.server import create_server

    parser = argparse.ArgumentParser(description="Local dashboard for Rocket RL Bot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8717)
    args = parser.parse_args()

    server = create_server(PROJECT_ROOT, args.host, args.port)
    print(f"[dashboard] http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
