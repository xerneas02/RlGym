from __future__ import annotations

from _bootstrap import bootstrap_project_root, ensure_rocketsim_available, ensure_rocketsim_arena_ready

PROJECT_ROOT = bootstrap_project_root()


def main() -> None:
    ensure_rocketsim_available()
    print("[preflight] RocketSim installe, verification de l'arena...", flush=True)
    ensure_rocketsim_arena_ready(PROJECT_ROOT)
    print("[preflight] Arena RocketSim OK", flush=True)
    from src.rl.trainer import PPOTrainer
    from src.utils.config_loader import load_project_configs

    config, rewards, curriculum = load_project_configs(PROJECT_ROOT)
    trainer = PPOTrainer(PROJECT_ROOT, config, rewards, curriculum)
    trainer.train()


if __name__ == "__main__":
    main()
