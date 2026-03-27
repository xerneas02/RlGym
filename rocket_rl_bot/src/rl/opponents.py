from __future__ import annotations

import importlib.util
import math
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.env.actions import OptimizedDiscreteAction
from src.env.observations import CompactObservationBuilder

try:
    from rlgym_sim.utils import common_values as sim_common_values
except ImportError:  # pragma: no cover
    from rlgym.utils import common_values as sim_common_values


BOOST_LOCATIONS = np.asarray(sim_common_values.BOOST_LOCATIONS, dtype=np.float32)
BOOST_IS_LARGE = BOOST_LOCATIONS[:, 2] > 71.0


@dataclass(frozen=True)
class OpponentSpec:
    name: str
    weight: float = 1.0


def normalize_opponent_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized in {"bronze", "bronze_chaser", "scripted"}:
        return "bronze_chaser"
    if normalized in {"aggressive", "aggressive_chaser", "aggressive_ball_chaser"}:
        return "aggressive_chaser"
    if normalized in {"necto"}:
        return "necto"
    if normalized in {"seer"}:
        return "seer"
    if normalized in {"self_play", "self", "mirror", "policy", "current_policy"}:
        return "self_play"
    if normalized in {"historical_self_play", "history", "history_pool", "checkpoint_pool", "old_self_play"}:
        return "historical_self_play"
    if normalized in {"none", "off", "disabled"}:
        return "self_play"
    raise ValueError(
        f"Unsupported opponent: {name}. "
        "Use one of: self_play, historical_self_play, bronze_chaser, aggressive_chaser, necto, seer."
    )


def _parse_weighted_specs(entries: Sequence[Any], weight_scale: float = 1.0) -> list[OpponentSpec]:
    specs: list[OpponentSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            name = normalize_opponent_name(entry)
            weight = 1.0
        elif isinstance(entry, Mapping):
            raw_name = entry.get("name", entry.get("opponent", "self_play"))
            name = normalize_opponent_name(str(raw_name))
            weight = float(entry.get("weight", 1.0))
        else:
            raise TypeError(f"Unsupported opponent entry: {entry!r}")

        weight *= float(weight_scale)
        if weight <= 0.0:
            continue
        specs.append(OpponentSpec(name=name, weight=weight))
    return specs


def parse_opponent_specs(config: Optional[Mapping[str, Any]]) -> list[OpponentSpec]:
    default_specs = [OpponentSpec(name="self_play", weight=1.0)]
    if not config:
        return default_specs

    opponent_pool = config.get("opponent_pool")
    if opponent_pool is not None:
        specs = _parse_weighted_specs(list(opponent_pool))
        return specs or default_specs

    scripted_prob = float(config.get("scripted_opponent_prob", 0.0))
    scripted_entries = config.get("scripted_opponents")
    if scripted_entries is None:
        scripted_name = str(config.get("scripted_opponent", "self_play"))
        scripted_entries = [scripted_name]

    scripted_specs = [
        spec
        for spec in _parse_weighted_specs(list(scripted_entries), weight_scale=max(0.0, scripted_prob))
        if spec.name != "self_play"
    ]
    self_play_weight = max(0.0, 1.0 - scripted_prob)

    combined_specs: list[OpponentSpec] = []
    if self_play_weight > 0.0:
        combined_specs.append(OpponentSpec(name="self_play", weight=self_play_weight))
    combined_specs.extend(scripted_specs)
    return combined_specs or default_specs


class BronzeChaserPolicy:
    def __init__(self, action_parser: OptimizedDiscreteAction) -> None:
        self.action_parser = action_parser
        self.slices = CompactObservationBuilder.SLICES

    def reset(self, state: Any | None = None) -> None:
        return None

    def act(
        self,
        observation: np.ndarray,
        state: Any | None = None,
        player_index: int | None = None,
        previous_action: np.ndarray | None = None,
    ) -> int:
        rel_ball = observation[self.slices.relative_ball_position]
        ball_position = observation[self.slices.ball_position]
        on_ground = observation[25] > 0.5
        throttle = 1.0 if rel_ball[1] > -0.05 else -1.0
        steer = float(np.clip(rel_ball[0] * 2.5, -1.0, 1.0))
        yaw = steer
        pitch = -1.0 if ball_position[2] > 0.20 else 0.0
        boost = 1 if throttle > 0 and abs(steer) < 0.20 and rel_ball[1] > 0.18 else 0
        jump = 1 if on_ground and np.linalg.norm(rel_ball[:2]) < 0.08 and ball_position[2] > 0.12 else 0
        handbrake = 1 if abs(steer) > 0.75 and rel_ball[1] > 0.05 else 0
        action = [throttle, steer, pitch, yaw, 0.0, jump, boost, handbrake]
        return self.action_parser.nearest_action_index(action)


class AggressiveChaserPolicy:
    def __init__(self, action_parser: OptimizedDiscreteAction) -> None:
        self.action_parser = action_parser
        self.slices = CompactObservationBuilder.SLICES

    def reset(self, state: Any | None = None) -> None:
        return None

    def act(
        self,
        observation: np.ndarray,
        state: Any | None = None,
        player_index: int | None = None,
        previous_action: np.ndarray | None = None,
    ) -> int:
        rel_ball = observation[self.slices.relative_ball_position]
        ball_position = observation[self.slices.ball_position]
        on_ground = observation[25] > 0.5
        forward = float(rel_ball[1])
        lateral = float(rel_ball[0])
        planar_distance = float(np.linalg.norm(rel_ball[:2]))
        throttle = 1.0 if forward > -0.20 else -1.0
        steer = float(np.clip(lateral * 3.4, -1.0, 1.0))
        yaw = steer
        pitch = -1.0 if ball_position[2] > 0.22 else 0.0
        boost = 1 if throttle > 0 and forward > -0.10 and abs(steer) < 0.65 else 0
        jump = 1 if on_ground and planar_distance < 0.12 and (ball_position[2] > 0.10 or forward > 0.05) else 0
        handbrake = 1 if abs(steer) > 0.85 and forward > -0.15 else 0
        action = [throttle, steer, pitch, yaw, 0.0, jump, boost, handbrake]
        return self.action_parser.nearest_action_index(action)


class _NectoObservationBuilder:
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4, dtype=np.float32)
    _norm = np.array([1.0] * 5 + [2300.0] * 6 + [1.0] * 6 + [5.5] * 3 + [1.0] * 4, dtype=np.float32)

    def __init__(self, tick_skip: int = 8) -> None:
        self.tick_skip = int(tick_skip)
        self.demo_timers = Counter()
        self.boost_timers: Optional[np.ndarray] = None

    def reset(self, state: Any) -> None:
        boost_pads = np.asarray(getattr(state, "boost_pads", []), dtype=np.float32)
        self.demo_timers = Counter()
        self.boost_timers = np.zeros(len(boost_pads), dtype=np.float32)

    def _maybe_update_obs(self, state: Any) -> tuple[np.ndarray, np.ndarray]:
        boost_pads = np.asarray(getattr(state, "boost_pads", []), dtype=np.float32)
        if self.boost_timers is None or len(self.boost_timers) != len(boost_pads):
            self.reset(state)

        qkv = np.zeros((1, 1 + len(state.players) + len(boost_pads), 24), dtype=np.float32)
        ball = state.ball
        qkv[0, 0, 3] = 1.0
        qkv[0, 0, 5:8] = np.asarray(ball.position, dtype=np.float32)
        qkv[0, 0, 8:11] = np.asarray(ball.linear_velocity, dtype=np.float32)
        qkv[0, 0, 17:20] = np.asarray(ball.angular_velocity, dtype=np.float32)

        index = 1
        for player in state.players:
            qkv[0, index, 1 if player.team_num == sim_common_values.BLUE_TEAM else 2] = 1.0
            car_data = player.car_data
            qkv[0, index, 5:8] = np.asarray(car_data.position, dtype=np.float32)
            qkv[0, index, 8:11] = np.asarray(car_data.linear_velocity, dtype=np.float32)
            qkv[0, index, 11:14] = np.asarray(car_data.forward(), dtype=np.float32)
            qkv[0, index, 14:17] = np.asarray(car_data.up(), dtype=np.float32)
            qkv[0, index, 17:20] = np.asarray(car_data.angular_velocity, dtype=np.float32)
            qkv[0, index, 20] = float(player.boost_amount)
            qkv[0, index, 22] = float(player.on_ground)
            qkv[0, index, 23] = float(player.has_flip)
            if self.demo_timers[player.car_id] <= 0:
                self.demo_timers[player.car_id] = 3.0
            else:
                self.demo_timers[player.car_id] = max(self.demo_timers[player.car_id] - self.tick_skip / 120.0, 0.0)
            qkv[0, index, 21] = float(self.demo_timers[player.car_id] / 10.0)
            index += 1

        qkv[0, index:, 4] = 1.0
        qkv[0, index:, 5:8] = BOOST_LOCATIONS[: len(boost_pads)]
        qkv[0, index:, 20] = 0.12 + 0.88 * BOOST_IS_LARGE[: len(boost_pads)]
        new_boost_grabs = (boost_pads == 1.0) & (self.boost_timers == 0.0)
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * BOOST_IS_LARGE[new_boost_grabs]
        self.boost_timers *= boost_pads
        qkv[0, index:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200.0
        self.boost_timers[self.boost_timers < 0.0] = 0.0

        mask = np.zeros((1, qkv.shape[1]), dtype=np.float32)
        return qkv / self._norm, mask

    def build_obs(self, player: Any, state: Any, previous_action: np.ndarray) -> Any:
        qkv, mask = self._maybe_update_obs(state)
        invert = player.team_num == sim_common_values.ORANGE_TEAM
        qkv = qkv.copy()
        mask = mask.copy()
        main_index = state.players.index(player) + 1
        qkv[0, main_index, 0] = 1.0
        if invert:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]
            qkv *= self._invert
        q = qkv[0, main_index, :]
        q = np.expand_dims(np.concatenate((q, previous_action.astype(np.float32)), axis=0), axis=(0, 1))
        kv = qkv
        kv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, kv, mask


class NectoPolicy:
    _shared_actor: Any = None
    _shared_actor_path: Optional[Path] = None

    def __init__(
        self,
        action_parser: OptimizedDiscreteAction,
        project_root: Optional[Path] = None,
        beta: float = 1.0,
    ) -> None:
        self.action_parser = action_parser
        self.beta = float(beta)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.model_path = self.project_root / "Necto" / "Necto" / "necto-model.pt"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Necto model file not found: {self.model_path}")
        if NectoPolicy._shared_actor is None or NectoPolicy._shared_actor_path != self.model_path:
            with self.model_path.open("rb") as handle:
                NectoPolicy._shared_actor = torch.jit.load(handle, map_location="cpu")
            NectoPolicy._shared_actor.eval()
            NectoPolicy._shared_actor_path = self.model_path
        self.actor = NectoPolicy._shared_actor
        self.obs_builder = _NectoObservationBuilder()

    def reset(self, state: Any | None = None) -> None:
        if state is not None:
            self.obs_builder.reset(state)
        else:
            self.obs_builder.boost_timers = None
            self.obs_builder.demo_timers = Counter()

    def act(
        self,
        observation: np.ndarray,
        state: Any | None = None,
        player_index: int | None = None,
        previous_action: np.ndarray | None = None,
    ) -> int:
        if state is None or player_index is None:
            raise ValueError("NectoPolicy requires the raw env state and the local player index.")
        local_index = int(player_index)
        if local_index < 0 or local_index >= len(state.players):
            raise IndexError(f"Invalid player index {local_index} for state with {len(state.players)} players")
        previous_action = np.asarray(previous_action if previous_action is not None else np.zeros(8), dtype=np.float32)
        player = state.players[local_index]
        teammates = [other for idx, other in enumerate(state.players) if idx != local_index and other.team_num == player.team_num]
        opponents = [other for idx, other in enumerate(state.players) if idx != local_index and other.team_num != player.team_num]
        ordered_state = SimpleNamespace(
            ball=state.ball,
            boost_pads=np.asarray(state.boost_pads, dtype=np.float32),
            players=[player, *teammates, *opponents],
        )
        obs = self.obs_builder.build_obs(player, ordered_state, previous_action)
        continuous_action = self._run_model(obs, self.beta)
        return self.action_parser.nearest_action_index(continuous_action)

    def _run_model(self, state: Sequence[np.ndarray], beta: float) -> np.ndarray:
        state_tensors = tuple(torch.from_numpy(np.asarray(component)).float() for component in state)
        with torch.no_grad():
            outputs, _ = self.actor(state_tensors)
        max_shape = max(output.shape[-1] for output in outputs)
        logits = torch.stack(
            [
                output
                if output.shape[-1] == max_shape
                else F.pad(
                    output,
                    pad=(0, max_shape - output.shape[-1]),
                    value=float("-inf") if beta >= 0 else float("inf"),
                )
                for output in outputs
            ]
        ).swapdims(0, 1).squeeze()
        if beta == 1.0:
            actions = torch.argmax(logits, dim=-1).cpu().numpy()
        elif beta == -1.0:
            actions = torch.argmin(logits, dim=-1).cpu().numpy()
        else:
            sampled_logits = logits.clone()
            if beta == 0.0:
                sampled_logits[torch.isfinite(sampled_logits)] = 0.0
            else:
                sampled_logits *= math.log((beta + 1.0) / (1.0 - beta), 3)
            actions = Categorical(logits=sampled_logits).sample().cpu().numpy()
        actions = np.asarray(actions, dtype=np.int64).reshape((-1, 5))
        actions[:, 0] -= 1
        actions[:, 1] -= 1
        parsed = np.zeros((actions.shape[0], 8), dtype=np.float32)
        parsed[:, 0] = actions[:, 0]
        parsed[:, 1] = actions[:, 1]
        parsed[:, 2] = actions[:, 0]
        parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])
        parsed[:, 4] = actions[:, 1] * actions[:, 4]
        parsed[:, 5] = actions[:, 2]
        parsed[:, 6] = actions[:, 3]
        parsed[:, 7] = actions[:, 4]
        return parsed[0]


def _ensure_optional_module_stubs() -> None:
    if "numba" not in sys.modules:
        try:
            import numba  # type: ignore  # noqa: F401
        except ImportError:
            numba_stub = types.ModuleType("numba")

            def jit(*args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            numba_stub.jit = jit
            sys.modules["numba"] = numba_stub

    if "sklearn.preprocessing" not in sys.modules:
        try:
            from sklearn.preprocessing import OneHotEncoder  # type: ignore  # noqa: F401
        except ImportError:
            sklearn_stub = types.ModuleType("sklearn")
            preprocessing_stub = types.ModuleType("sklearn.preprocessing")

            class OneHotEncoder:  # pragma: no cover - import fallback only
                def __init__(self, *args, **kwargs):
                    pass

                def fit_transform(self, values):
                    return values

            preprocessing_stub.OneHotEncoder = OneHotEncoder
            sklearn_stub.preprocessing = preprocessing_stub
            sys.modules["sklearn"] = sklearn_stub
            sys.modules["sklearn.preprocessing"] = preprocessing_stub


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    _ensure_optional_module_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeerPolicy:
    _shared_model: Any = None
    _shared_model_path: Optional[Path] = None
    _helper_module: Optional[ModuleType] = None
    _helper_path: Optional[Path] = None

    def __init__(self, action_parser: OptimizedDiscreteAction, project_root: Optional[Path] = None) -> None:
        self.action_parser = action_parser
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.base_path = self.project_root / "Seer"
        self.model_path = self.base_path / "Seer.pt"
        helper_path = self.base_path / "helper.py"
        if not self.model_path.exists():
            raise FileNotFoundError(f"Seer model file not found: {self.model_path}")
        if not helper_path.exists():
            raise FileNotFoundError(f"Seer helper file not found: {helper_path}")
        if SeerPolicy._helper_module is None or SeerPolicy._helper_path != helper_path:
            SeerPolicy._helper_module = _load_module_from_path("rocket_rl_bot_seer_helper", helper_path)
            SeerPolicy._helper_path = helper_path
        self.helper = SeerPolicy._helper_module
        if SeerPolicy._shared_model is None or SeerPolicy._shared_model_path != self.model_path:
            model = self.helper.Seer_Network()
            state_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()
            SeerPolicy._shared_model = model
            SeerPolicy._shared_model_path = self.model_path
        self.model = SeerPolicy._shared_model
        self.tick_skip = 8
        self.reset(None)

    def reset(self, state: Any | None = None) -> None:
        self.recurrent_state = (
            torch.zeros(1, 1, 512, dtype=torch.float32),
            torch.zeros(1, 1, 512, dtype=torch.float32),
        )
        self.prev_action = np.array([1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.demo_timers: dict[int, float] = {}
        self.prev_boost_available = None if state is None else np.asarray(state.boost_pads, dtype=np.float32) > 0.5
        self.boost_timers = np.zeros(34, dtype=np.float32)

    def _encode_prev_action(self) -> np.ndarray:
        throttle = np.zeros(3, dtype=np.float32)
        steer = np.zeros(5, dtype=np.float32)
        pitch = np.zeros(5, dtype=np.float32)
        roll = np.zeros(3, dtype=np.float32)
        throttle[int(np.clip(self.prev_action[0], 0, 2))] = 1.0
        steer[int(np.clip(self.prev_action[1], 0, 4))] = 1.0
        pitch[int(np.clip(self.prev_action[2], 0, 4))] = 1.0
        roll[int(np.clip(self.prev_action[3], 0, 2))] = 1.0
        binary = np.asarray(
            [
                float(self.prev_action[4] > 0.5),
                float(self.prev_action[5] > 0.5),
                float(self.prev_action[6] > 0.5),
            ],
            dtype=np.float32,
        )
        return np.concatenate((throttle, steer, pitch, roll, binary), dtype=np.float32)

    def _update_boost_timers(self, state: Any) -> np.ndarray:
        available = np.asarray(state.boost_pads, dtype=np.float32) > 0.5
        if self.prev_boost_available is None or len(self.prev_boost_available) != len(available):
            self.prev_boost_available = available.copy()
            self.boost_timers = np.zeros(len(available), dtype=np.float32)
            return self.boost_timers.copy()
        self.boost_timers = np.maximum(self.boost_timers - self.tick_skip / 120.0, 0.0)
        newly_unavailable = (~available) & self.prev_boost_available
        self.boost_timers[newly_unavailable] = np.where(BOOST_IS_LARGE[newly_unavailable], 10.0, 4.0)
        self.boost_timers[available] = 0.0
        self.prev_boost_available = available.copy()
        return self.boost_timers.copy()

    def _update_demo_timer(self, player: Any) -> float:
        car_id = int(player.car_id)
        if bool(player.is_demoed):
            self.demo_timers[car_id] = float(self.demo_timers.get(car_id, 0.0) + self.tick_skip / 120.0)
        else:
            self.demo_timers[car_id] = 0.0
        return float(self.demo_timers[car_id])

    def _encode_player(self, player: Any, inverted: bool) -> np.ndarray:
        car_data = player.car_data
        data = np.array(
            [
                car_data.position[0],
                car_data.position[1],
                car_data.position[2],
                car_data.pitch(),
                car_data.yaw(),
                car_data.roll(),
                car_data.linear_velocity[0],
                car_data.linear_velocity[1],
                car_data.linear_velocity[2],
                car_data.angular_velocity[0],
                car_data.angular_velocity[1],
                car_data.angular_velocity[2],
                self._update_demo_timer(player),
                float(player.boost_amount) * 100.0,
                float(player.on_ground),
                float(player.has_flip),
            ],
            dtype=np.float32,
        )
        if inverted:
            data = self.helper.invert_player_data(data)
        return data

    def _encode_ball(self, state: Any, inverted: bool) -> np.ndarray:
        ball = state.ball
        data = np.array(
            [
                ball.position[0],
                ball.position[1],
                ball.position[2],
                ball.linear_velocity[0],
                ball.linear_velocity[1],
                ball.linear_velocity[2],
                ball.angular_velocity[0],
                ball.angular_velocity[1],
                ball.angular_velocity[2],
            ],
            dtype=np.float32,
        )
        if inverted:
            data = self.helper.invert_ball_data(data)
        return data

    def _encode_boost(self, state: Any, inverted: bool) -> np.ndarray:
        timers = self._update_boost_timers(state)
        if inverted:
            timers = self.helper.invert_boost_data(timers)
        return timers

    def act(
        self,
        observation: np.ndarray,
        state: Any | None = None,
        player_index: int | None = None,
        previous_action: np.ndarray | None = None,
    ) -> int:
        if state is None or player_index is None:
            raise ValueError("SeerPolicy requires the raw env state and the local player index.")
        local_index = int(player_index)
        if local_index < 0 or local_index >= len(state.players):
            raise IndexError(f"Invalid player index {local_index} for state with {len(state.players)} players")
        player = state.players[local_index]
        opponents = [other for idx, other in enumerate(state.players) if idx != local_index and other.team_num != player.team_num]
        teammates = [other for idx, other in enumerate(state.players) if idx != local_index and other.team_num == player.team_num]
        other_players = opponents if opponents else teammates
        if not other_players:
            raise ValueError("SeerPolicy expects at least one opponent or teammate to build observations.")
        opponent = other_players[0]
        inverted = player.team_num == sim_common_values.ORANGE_TEAM
        player_data = self._encode_player(player, inverted)
        opponent_data = self._encode_player(opponent, inverted)
        boost_data = self._encode_boost(state, inverted)
        ball_data = self._encode_ball(state, inverted)
        prev_action_encoding = self._encode_prev_action()
        input_array = self.helper.impute_features(player_data, opponent_data, boost_data, ball_data, prev_action_encoding)
        with torch.no_grad():
            input_tensor = torch.as_tensor(input_array.reshape(1, -1), dtype=torch.float32)
            episode_starts = torch.zeros(1, dtype=torch.float32)
            actions, _, _, self.recurrent_state = self.model(input_tensor, self.recurrent_state, episode_starts, True)
            actions_np = actions.cpu().numpy()[0].astype(np.float32)
        self.prev_action = actions_np.copy()
        continuous_action = np.array(
            [
                actions_np[0] - 1.0,
                actions_np[1] * 0.5 - 1.0,
                actions_np[2] * 0.5 - 1.0,
                actions_np[1] * 0.5 - 1.0,
                actions_np[3] - 1.0,
                actions_np[4],
                actions_np[5],
                actions_np[6],
            ],
            dtype=np.float32,
        )
        return self.action_parser.nearest_action_index(continuous_action)


def create_scripted_policy(
    name: str,
    action_parser: Optional[OptimizedDiscreteAction] = None,
    project_root: Optional[Path] = None,
):
    parser = action_parser or OptimizedDiscreteAction()
    normalized = normalize_opponent_name(name)
    if normalized == "bronze_chaser":
        return BronzeChaserPolicy(parser)
    if normalized == "aggressive_chaser":
        return AggressiveChaserPolicy(parser)
    if normalized == "necto":
        return NectoPolicy(parser, project_root=project_root)
    if normalized == "seer":
        return SeerPolicy(parser, project_root=project_root)
    if normalized in {"self_play", "historical_self_play"}:
        return None
    raise ValueError(f"Unsupported opponent: {name}")
