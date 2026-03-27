from __future__ import annotations

import pathlib
import sys
from typing import Optional

import numpy as np
import torch


class TrainedAgent:
    def __init__(self, model_path: Optional[str] = None, device: str = "auto") -> None:
        self.base_path = pathlib.Path(__file__).parent.resolve()
        self.project_root = self.base_path.parent
        self.rocket_rl_root = self.project_root / "rocket_rl_bot"
        if str(self.rocket_rl_root) not in sys.path:
            sys.path.append(str(self.rocket_rl_root))

        from src.rl.model import ActorCritic

        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        policy_path = pathlib.Path(model_path).resolve() if model_path else self.base_path / "rlbot_policy.pt"
        if not policy_path.exists():
            raise FileNotFoundError(
                f"RLBot policy file not found: {policy_path}. Run export_to_rlbot.py first."
            )

        payload = torch.load(policy_path, map_location="cpu")
        self.lookup_table = np.asarray(payload["lookup_table"], dtype=np.float32)
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])

        self.actor = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        self.actor.load_state_dict(payload["model_state_dict"])
        self.actor.eval()

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action_index, _, _ = self.actor.act(obs_t, deterministic=True)
        return self.lookup_table[int(action_index.item())].copy()
