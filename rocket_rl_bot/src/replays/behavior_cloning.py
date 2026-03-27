from __future__ import annotations

import io
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from carball import analyze_replay_file
from torch.nn import functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from src.env.actions import OptimizedDiscreteAction
from src.env.observations import (
    BALL_MAX_SPEED,
    CAR_MAX_ANG_VEL,
    CAR_MAX_SPEED,
    FIELD_DIAGONAL,
    POS_SCALE,
    CompactObservationBuilder,
    common_values,
)
from src.rl.model import ActorCritic
from src.utils.checkpointing import load_checkpoint, save_checkpoint
from src.utils.logging_utils import TrainingLogger, create_run_directories, snapshot_config

OBS_DIM = CompactObservationBuilder.OBSERVATION_DIM
INVERT_VECTOR = np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
BOOST_LOCATIONS = np.asarray(getattr(common_values, "BOOST_LOCATIONS", ()), dtype=np.float32)
BOOST_PAD_COUNT = int(len(BOOST_LOCATIONS))
DEFAULT_BOOST_FRACTION = 0.33
REPLAY_CACHE_SCHEMA_VERSION = 2
GROUND_Z_THRESHOLD = 25.0
GROUND_VELOCITY_THRESHOLD = 150.0
BOOST_CONSUMPTION_PER_SECOND = 1.0 / 3.0
SMALL_PAD_BOOST_FRACTION = 12.0 / 100.0
LARGE_PAD_BOOST_FRACTION = 1.0
SMALL_PAD_RESPAWN_SECONDS = 4.0
LARGE_PAD_RESPAWN_SECONDS = 10.0
PAD_PICKUP_RADIUS = 220.0


def _build_inverted_pad_indices() -> np.ndarray:
    if BOOST_PAD_COUNT == 0:
        return np.zeros((0,), dtype=np.int64)

    mapping = []
    for location in BOOST_LOCATIONS:
        mirrored = location * INVERT_VECTOR
        distances = np.linalg.norm(BOOST_LOCATIONS - mirrored, axis=1)
        mapping.append(int(np.argmin(distances)))
    return np.asarray(mapping, dtype=np.int64)


INVERTED_PAD_INDICES = _build_inverted_pad_indices()
LARGE_PAD_MASK = (
    BOOST_LOCATIONS[:, 2] > 71.0 if BOOST_PAD_COUNT > 0 else np.zeros((0,), dtype=bool)
)


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    frame_count: int
    sample_count: int


@dataclass
class ReplayTrainStats:
    loss_sum: float = 0.0
    correct: int = 0
    samples: int = 0
    replays: int = 0
    skipped_replays: int = 0

    def update(self, loss: float, correct: int, samples: int) -> None:
        self.loss_sum += float(loss) * int(samples)
        self.correct += int(correct)
        self.samples += int(samples)

    def as_metrics(self) -> dict[str, float]:
        average_loss = self.loss_sum / max(self.samples, 1)
        accuracy = self.correct / max(self.samples, 1)
        return {
            "bc_loss": float(average_loss),
            "bc_accuracy": float(accuracy),
            "bc_samples": float(self.samples),
            "bc_replays": float(self.replays),
            "bc_skipped_replays": float(self.skipped_replays),
        }


@dataclass
class ReplayProgressTracker:
    total: int
    width: int = 28

    def __post_init__(self) -> None:
        self._current_completed = 0
        self._tqdm = None
        self._encoding = sys.stdout.encoding or "utf-8"
        if tqdm is not None:
            self._tqdm = tqdm(
                total=max(self.total, 1),
                dynamic_ncols=True,
                leave=True,
                unit="replay",
                smoothing=0.05,
            )

    def _safe_text(self, text: str) -> str:
        try:
            return text.encode(self._encoding, errors="replace").decode(self._encoding)
        except LookupError:
            return text.encode("utf-8", errors="replace").decode("utf-8")

    def render(
        self,
        completed: int,
        *,
        cached_replays: int,
        parsed_replays: int,
        skipped_replays: int,
        samples_seen: int,
        current: str,
        final: bool = False,
    ) -> None:
        if self._tqdm is None:
            ratio = min(max(completed / max(self.total, 1), 0.0), 1.0)
            current_short = current if len(current) <= 44 else current[:41] + "..."
            message = (
                f"[bc][progress] {completed}/{self.total} ({ratio * 100:5.1f}%) "
                f"cache={cached_replays} parsed={parsed_replays} skipped={skipped_replays} "
                f"samples={samples_seen} current={current_short}"
            )
            print(self._safe_text(message), end="\n" if final else "\r", flush=True)
            return

        delta = max(0, int(completed) - int(self._current_completed))
        if delta:
            self._tqdm.update(delta)
            self._current_completed = int(completed)
        current_short = current if len(current) <= 36 else current[:33] + "..."
        self._tqdm.set_description_str(self._safe_text(current_short))
        self._tqdm.set_postfix_str(
            self._safe_text(
                f"cache={cached_replays} parsed={parsed_replays} skipped={skipped_replays} samples={samples_seen}"
            ),
            refresh=False,
        )
        self._tqdm.refresh()
        if final:
            self._tqdm.close()
            self._tqdm = None

    def newline(self) -> None:
        return None

    def write(self, message: str) -> None:
        safe_message = self._safe_text(message)
        if self._tqdm is not None:
            self._tqdm.write(safe_message)
        else:
            print(safe_message)


@contextmanager
def suppress_process_output(enabled: bool):
    if not enabled:
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def load_cached_batch(cache_path: Path) -> ReplayBatch:
    payload = np.load(cache_path)
    schema_version = int(payload.get("schema_version", np.asarray([0], dtype=np.int16))[0])
    if schema_version != REPLAY_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"cache schema {schema_version} incompatible with expected {REPLAY_CACHE_SCHEMA_VERSION}"
        )
    observations = payload["observations"].astype(np.float32)
    actions = payload["actions"].astype(np.int64)
    frame_count = int(payload.get("frame_count", len(actions)))
    return ReplayBatch(
        observations=observations,
        actions=actions,
        frame_count=frame_count,
        sample_count=int(actions.shape[0]),
    )


def save_cached_batch(cache_path: Path, batch: ReplayBatch) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        observations=batch.observations.astype(np.float16),
        actions=batch.actions.astype(np.uint8),
        frame_count=np.asarray([batch.frame_count], dtype=np.int32),
        schema_version=np.asarray([REPLAY_CACHE_SCHEMA_VERSION], dtype=np.int16),
    )


def cache_path_for_replay(cache_dir: Path, replay_path: Path) -> Path:
    return cache_dir / f"{replay_path.stem}_v{REPLAY_CACHE_SCHEMA_VERSION}.npz"


def raw_axis_to_signed(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    signed = (numeric - 128.0) / 127.0
    if abs(signed) < 0.05:
        signed = 0.0
    return float(np.clip(signed, -1.0, 1.0))


def active_flag(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return 1.0 if numeric > 0.0 else 0.0


def scalar_value(row: Any, group: str, column: str, default: float = 0.0) -> float:
    key = (group, column)
    if key not in row.index:
        return float(default)
    try:
        numeric = float(row[key])
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(numeric):
        return float(default)
    return float(numeric)


def extract_xyz(row: Any, group: str, prefix: str) -> np.ndarray:
    return np.asarray(
        [
            float(row[(group, f"{prefix}_x")]),
            float(row[(group, f"{prefix}_y")]),
            float(row[(group, f"{prefix}_z")]),
        ],
        dtype=np.float32,
    )


def is_valid_vector(vector: np.ndarray) -> bool:
    return bool(np.isfinite(vector).all())


def rotation_to_forward_up(rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pitch = float(rotation[0])
    yaw = float(rotation[1])
    roll = float(rotation[2])

    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cr = math.cos(roll)
    sr = math.sin(roll)

    forward = np.asarray([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.asarray(
        [cy * sp * sr - cr * sy, sy * sp * sr + cr * cy, cp * sr],
        dtype=np.float32,
    )
    return forward, up


def estimate_on_ground(position: np.ndarray, velocity: np.ndarray, jump_active: float, dodge_active: float) -> float:
    if jump_active > 0.0 or dodge_active > 0.0:
        return 0.0
    grounded = position[2] <= GROUND_Z_THRESHOLD and abs(float(velocity[2])) <= GROUND_VELOCITY_THRESHOLD
    return 1.0 if grounded else 0.0


def estimate_has_flip(on_ground: float, double_jump_active: float, dodge_active: float) -> float:
    if on_ground > 0.5:
        return 1.0
    if double_jump_active > 0.0 or dodge_active > 0.0:
        return 0.0
    return 1.0


def estimate_controls(row: Any, player_name: str, on_ground: float) -> np.ndarray:
    throttle = raw_axis_to_signed(row[(player_name, "throttle")])
    steer = raw_axis_to_signed(row[(player_name, "steer")])
    boost = active_flag(row[(player_name, "boost_active")])
    handbrake = active_flag(row[(player_name, "handbrake")])
    jump = max(
        active_flag(row[(player_name, "jump_active")]),
        active_flag(row[(player_name, "double_jump_active")]),
        active_flag(row[(player_name, "dodge_active")]),
    )

    pitch = 0.0 if on_ground > 0.5 else -throttle
    yaw = steer
    roll = 0.0
    if handbrake > 0.5 and on_ground < 0.5:
        roll = steer
        yaw = 0.0

    if boost > 0.5 and throttle < 0.0:
        throttle = 0.0

    return np.asarray(
        [throttle, steer, pitch, yaw, roll, jump, boost, handbrake],
        dtype=np.float32,
    )


def maybe_invert(vector: np.ndarray, invert: bool) -> np.ndarray:
    if not invert:
        return vector
    return vector * INVERT_VECTOR


def norm_pos(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32) / POS_SCALE


def norm_vel(vector: np.ndarray, max_norm: float) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32) / float(max_norm)


def norm_distance(vector: np.ndarray) -> np.ndarray:
    return np.asarray([float(np.linalg.norm(vector) / FIELD_DIAGONAL)], dtype=np.float32)


def wall_distances(position: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in position]
    return np.asarray(
        [
            (4096.0 - abs(x)) / 4096.0,
            (5120.0 + y) / 10240.0,
            (5120.0 - y) / 10240.0,
            (2044.0 - z) / 2044.0,
        ],
        dtype=np.float32,
    )


class ReplayBoostStateEstimator:
    def __init__(self, players: list[str]) -> None:
        self.players = list(players)
        self.player_boost = {
            player_name: float(DEFAULT_BOOST_FRACTION)
            for player_name in self.players
        }
        self.pad_cooldowns = np.zeros(BOOST_PAD_COUNT, dtype=np.float32)

    def update(self, row: Any, player_positions: dict[str, np.ndarray], delta_time: float) -> None:
        if BOOST_PAD_COUNT == 0:
            return

        dt = float(np.clip(delta_time, 0.0, 0.25))
        if dt > 0.0:
            self.pad_cooldowns = np.maximum(0.0, self.pad_cooldowns - dt)

        for player_name in self.players:
            boost_active = active_flag(row[(player_name, "boost_active")])
            current = self.player_boost[player_name]
            if boost_active > 0.5 and current > 0.0:
                current -= BOOST_CONSUMPTION_PER_SECOND * dt
            self.player_boost[player_name] = float(np.clip(current, 0.0, 1.0))

        active_pads = self.pad_cooldowns <= 1e-6
        for player_name, position in player_positions.items():
            if position is None or not is_valid_vector(position):
                continue

            deltas = BOOST_LOCATIONS - position.reshape(1, 3)
            distances = np.linalg.norm(deltas, axis=1)
            candidate_indices = np.where(active_pads & (distances <= PAD_PICKUP_RADIUS))[0]
            if candidate_indices.size == 0:
                continue

            nearest_index = int(candidate_indices[np.argmin(distances[candidate_indices])])
            is_large_pad = bool(LARGE_PAD_MASK[nearest_index])
            self.pad_cooldowns[nearest_index] = (
                LARGE_PAD_RESPAWN_SECONDS if is_large_pad else SMALL_PAD_RESPAWN_SECONDS
            )
            gain = LARGE_PAD_BOOST_FRACTION if is_large_pad else SMALL_PAD_BOOST_FRACTION
            self.player_boost[player_name] = float(
                np.clip(self.player_boost[player_name] + gain, 0.0, 1.0)
            )
            active_pads[nearest_index] = False

        for player_name in self.players:
            raw_boost = scalar_value(row, player_name, "boost", default=-1.0)
            if raw_boost <= 0.0:
                continue
            normalized = raw_boost / 100.0 if raw_boost > 1.5 else raw_boost
            if 0.0 <= normalized <= 1.0:
                self.player_boost[player_name] = float(normalized)

    def get_player_boost(self, player_name: str) -> float:
        return float(self.player_boost.get(player_name, DEFAULT_BOOST_FRACTION))

    def get_boost_pad_state(self, is_orange: bool) -> np.ndarray:
        active = (self.pad_cooldowns <= 1e-6).astype(np.float32)
        if is_orange and INVERTED_PAD_INDICES.size == active.size:
            return active[INVERTED_PAD_INDICES].copy()
        return active.copy()


def parse_replay_quietly(replay_path: Path, suppress_parser_output: bool = True):
    with suppress_process_output(suppress_parser_output):
        with warnings.catch_warnings(), io.StringIO() as stdout_buffer, io.StringIO() as stderr_buffer:
            warnings.simplefilter("ignore")
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                analysis = analyze_replay_file(str(replay_path))
                df = analysis.get_data_frame()
                proto = analysis.get_protobuf_data()
    return df, proto


def build_observation_from_frame(
    row: Any,
    *,
    player_name: str,
    opponent_name: str,
    is_orange: bool,
    previous_action: np.ndarray,
    self_boost_amount: float,
    opponent_boost_amount: float,
    boost_pad_state: np.ndarray,
) -> np.ndarray | None:
    ball_position = extract_xyz(row, "ball", "pos")
    ball_velocity = extract_xyz(row, "ball", "vel")
    ball_angular_velocity = extract_xyz(row, "ball", "ang_vel")

    self_position = extract_xyz(row, player_name, "pos")
    self_velocity = extract_xyz(row, player_name, "vel")
    self_angular_velocity = extract_xyz(row, player_name, "ang_vel")
    self_rotation = extract_xyz(row, player_name, "rot")

    opponent_position = extract_xyz(row, opponent_name, "pos")
    opponent_velocity = extract_xyz(row, opponent_name, "vel")
    opponent_angular_velocity = extract_xyz(row, opponent_name, "ang_vel")
    opponent_rotation = extract_xyz(row, opponent_name, "rot")

    vectors = [
        ball_position,
        ball_velocity,
        ball_angular_velocity,
        self_position,
        self_velocity,
        self_angular_velocity,
        opponent_position,
        opponent_velocity,
        opponent_angular_velocity,
    ]
    if not all(is_valid_vector(vector) for vector in vectors):
        return None
    if not (is_valid_vector(self_rotation) and is_valid_vector(opponent_rotation)):
        return None

    self_forward, self_up = rotation_to_forward_up(self_rotation)
    opponent_forward, opponent_up = rotation_to_forward_up(opponent_rotation)

    ball_position = maybe_invert(ball_position, is_orange)
    ball_velocity = maybe_invert(ball_velocity, is_orange)
    ball_angular_velocity = maybe_invert(ball_angular_velocity, is_orange)
    self_position = maybe_invert(self_position, is_orange)
    self_velocity = maybe_invert(self_velocity, is_orange)
    self_angular_velocity = maybe_invert(self_angular_velocity, is_orange)
    self_forward = maybe_invert(self_forward, is_orange)
    self_up = maybe_invert(self_up, is_orange)
    opponent_position = maybe_invert(opponent_position, is_orange)
    opponent_velocity = maybe_invert(opponent_velocity, is_orange)
    opponent_angular_velocity = maybe_invert(opponent_angular_velocity, is_orange)
    opponent_forward = maybe_invert(opponent_forward, is_orange)
    opponent_up = maybe_invert(opponent_up, is_orange)

    jump_active = active_flag(row[(player_name, "jump_active")])
    double_jump_active = active_flag(row[(player_name, "double_jump_active")])
    dodge_active = active_flag(row[(player_name, "dodge_active")])

    on_ground = estimate_on_ground(self_position, self_velocity, jump_active, dodge_active)
    has_flip = estimate_has_flip(on_ground, double_jump_active, dodge_active)
    is_demoed = 0.0

    opponent_jump = active_flag(row[(opponent_name, "jump_active")])
    opponent_double_jump = active_flag(row[(opponent_name, "double_jump_active")])
    opponent_dodge = active_flag(row[(opponent_name, "dodge_active")])
    opponent_on_ground = estimate_on_ground(
        opponent_position,
        opponent_velocity,
        opponent_jump,
        opponent_dodge,
    )
    opponent_has_flip = estimate_has_flip(opponent_on_ground, opponent_double_jump, opponent_dodge)

    # Match the online observation builder: once orange data is inverted, the
    # semantic "attack goal" must still point to ORANGE_GOAL_BACK.
    attack_goal = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
    defend_goal = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)

    relative_ball_position = ball_position - self_position
    relative_ball_velocity = ball_velocity - self_velocity
    attack_goal_vector = attack_goal - self_position
    defend_goal_vector = defend_goal - self_position
    self_to_opponent_position = opponent_position - self_position
    self_to_opponent_velocity = opponent_velocity - self_velocity
    opponent_to_ball_position = ball_position - opponent_position
    opponent_to_ball_velocity = ball_velocity - opponent_velocity
    ball_to_attack_goal = attack_goal - ball_position

    observation = np.concatenate(
        [
            norm_pos(ball_position),
            norm_vel(ball_velocity, BALL_MAX_SPEED),
            norm_vel(ball_angular_velocity, CAR_MAX_ANG_VEL),
            norm_pos(self_position),
            norm_vel(self_velocity, CAR_MAX_SPEED),
            norm_vel(self_angular_velocity, CAR_MAX_ANG_VEL),
            self_forward.astype(np.float32),
            self_up.astype(np.float32),
            np.asarray([self_boost_amount, on_ground, has_flip, is_demoed], dtype=np.float32),
            norm_pos(relative_ball_position),
            norm_vel(relative_ball_velocity, CAR_MAX_SPEED),
            norm_distance(relative_ball_position),
            norm_pos(attack_goal_vector),
            norm_distance(attack_goal_vector),
            norm_pos(defend_goal_vector),
            norm_distance(defend_goal_vector),
            wall_distances(self_position),
            np.asarray(boost_pad_state, dtype=np.float32),
            norm_pos(opponent_position),
            norm_vel(opponent_velocity, CAR_MAX_SPEED),
            norm_vel(opponent_angular_velocity, CAR_MAX_ANG_VEL),
            opponent_forward.astype(np.float32),
            opponent_up.astype(np.float32),
            np.asarray(
                [opponent_boost_amount, opponent_on_ground, opponent_has_flip, 0.0],
                dtype=np.float32,
            ),
            norm_pos(self_to_opponent_position),
            norm_vel(self_to_opponent_velocity, CAR_MAX_SPEED),
            norm_distance(self_to_opponent_position),
            norm_pos(opponent_to_ball_position),
            norm_vel(opponent_to_ball_velocity, CAR_MAX_SPEED),
            norm_distance(opponent_to_ball_position),
            norm_pos(ball_to_attack_goal),
            norm_distance(ball_to_attack_goal),
            np.asarray(previous_action, dtype=np.float32).reshape(-1),
        ],
        dtype=np.float32,
    )
    if observation.shape[0] != OBS_DIM:
        raise ValueError(f"Expected observation dim {OBS_DIM}, got {observation.shape[0]}")
    return observation


def resolve_frame_stride(df: Any, sample_fps: float) -> int:
    if sample_fps <= 0:
        return 1
    delta_key = ("game", "delta")
    if delta_key not in df.columns:
        return 1
    deltas = np.asarray(df[delta_key], dtype=np.float32)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        return 1
    median_delta = float(np.median(deltas))
    target_delta = 1.0 / sample_fps
    return max(1, int(round(target_delta / median_delta)))


def extract_frame_delta(row: Any) -> float:
    return scalar_value(row, "game", "delta", default=1.0 / 120.0)


def extract_replay_batch(
    replay_path: Path,
    *,
    action_parser: OptimizedDiscreteAction,
    sample_fps: float,
    max_samples_per_replay: int | None,
    suppress_parser_output: bool = True,
) -> ReplayBatch:
    df, proto = parse_replay_quietly(replay_path, suppress_parser_output=suppress_parser_output)

    df_players = [
        level_zero
        for level_zero in df.columns.get_level_values(0).unique()
        if level_zero not in ("ball", "game")
    ]
    team_map = {
        player.name: bool(player.is_orange)
        for player in proto.players
        if player.name in df_players
    }
    players = [player_name for player_name in df_players if player_name in team_map]
    if len(players) != 2:
        raise ValueError(f"Replay {replay_path.name} n'est pas un 1v1 exploitable ({players}).")

    frame_stride = resolve_frame_stride(df, sample_fps)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    previous_actions = {
        player_name: np.zeros(8, dtype=np.float32)
        for player_name in players
    }
    boost_state = ReplayBoostStateEstimator(players)

    for frame_idx in range(len(df)):
        row = df.iloc[frame_idx]
        world_positions = {
            player_name: extract_xyz(row, player_name, "pos")
            for player_name in players
        }
        boost_state.update(row, world_positions, extract_frame_delta(row))

        if frame_idx % frame_stride != 0:
            continue

        for player_name in players:
            opponent_name = players[1] if player_name == players[0] else players[0]
            observation = build_observation_from_frame(
                row,
                player_name=player_name,
                opponent_name=opponent_name,
                is_orange=team_map[player_name],
                previous_action=previous_actions[player_name],
                self_boost_amount=boost_state.get_player_boost(player_name),
                opponent_boost_amount=boost_state.get_player_boost(opponent_name),
                boost_pad_state=boost_state.get_boost_pad_state(team_map[player_name]),
            )
            if observation is None:
                continue
            on_ground = estimate_on_ground(
                maybe_invert(extract_xyz(row, player_name, "pos"), team_map[player_name]),
                maybe_invert(extract_xyz(row, player_name, "vel"), team_map[player_name]),
                active_flag(row[(player_name, "jump_active")]),
                active_flag(row[(player_name, "dodge_active")]),
            )
            action_vector = estimate_controls(row, player_name, on_ground)
            action_index = action_parser.nearest_action_index(action_vector)
            previous_actions[player_name] = action_parser.get_action(action_index)
            observations.append(observation)
            actions.append(action_index)

    if not observations:
        return ReplayBatch(
            observations=np.zeros((0, OBS_DIM), dtype=np.float32),
            actions=np.zeros((0,), dtype=np.int64),
            frame_count=len(df),
            sample_count=0,
        )

    observation_array = np.stack(observations, axis=0).astype(np.float32)
    action_array = np.asarray(actions, dtype=np.int64)

    if max_samples_per_replay is not None and action_array.shape[0] > max_samples_per_replay:
        selected = np.linspace(0, action_array.shape[0] - 1, num=max_samples_per_replay, dtype=np.int64)
        observation_array = observation_array[selected]
        action_array = action_array[selected]

    return ReplayBatch(
        observations=observation_array,
        actions=action_array,
        frame_count=len(df),
        sample_count=int(action_array.shape[0]),
    )


def load_or_create_replay_batch(
    replay_path: Path,
    *,
    cache_dir: Path | None,
    action_parser: OptimizedDiscreteAction,
    sample_fps: float,
    max_samples_per_replay: int | None,
    suppress_parser_output: bool = True,
) -> tuple[ReplayBatch, bool]:
    if cache_dir is not None:
        cache_path = cache_path_for_replay(cache_dir, replay_path)
        if cache_path.exists():
            try:
                return load_cached_batch(cache_path), True
            except Exception:
                pass
    batch = extract_replay_batch(
        replay_path,
        action_parser=action_parser,
        sample_fps=sample_fps,
        max_samples_per_replay=max_samples_per_replay,
        suppress_parser_output=suppress_parser_output,
    )
    if cache_dir is not None and batch.sample_count > 0:
        save_cached_batch(cache_path_for_replay(cache_dir, replay_path), batch)
    return batch, False


def evaluate_replay_policy(
    model: ActorCritic,
    replay_files: list[Path],
    *,
    cache_dir: Path | None,
    action_parser: OptimizedDiscreteAction,
    sample_fps: float,
    max_samples_per_replay: int | None,
    batch_size: int,
    device: torch.device,
    suppress_parser_output: bool = True,
    progress: ReplayProgressTracker | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    skipped_replays = 0

    with torch.no_grad():
        for replay_path in replay_files:
            try:
                batch, _ = load_or_create_replay_batch(
                    replay_path,
                    cache_dir=cache_dir,
                    action_parser=action_parser,
                    sample_fps=sample_fps,
                    max_samples_per_replay=max_samples_per_replay,
                    suppress_parser_output=suppress_parser_output,
                )
            except Exception as exc:
                skipped_replays += 1
                if progress is not None:
                    progress.write(f"[bc][val][skip] {replay_path.name}: {exc}")
                continue

            if batch.sample_count == 0:
                skipped_replays += 1
                continue

            indices = np.arange(batch.sample_count)
            for start in range(0, batch.sample_count, batch_size):
                batch_indices = indices[start : start + batch_size]
                observations = torch.as_tensor(batch.observations[batch_indices], dtype=torch.float32, device=device)
                actions = torch.as_tensor(batch.actions[batch_indices], dtype=torch.int64, device=device)
                logits, _ = model(observations)
                loss = F.cross_entropy(logits, actions)
                predictions = torch.argmax(logits, dim=-1)
                total_loss += float(loss.item()) * int(actions.shape[0])
                total_correct += int((predictions == actions).sum().item())
                total_samples += int(actions.shape[0])

    model.train()
    return {
        "validation_loss": float(total_loss / max(total_samples, 1)),
        "validation_accuracy": float(total_correct / max(total_samples, 1)),
        "validation_samples": float(total_samples),
        "validation_skipped_replays": float(skipped_replays),
    }

def run_behavior_cloning(
    *,
    project_root: Path,
    run_config: dict[str, Any],
    replay_files: list[Path],
    validation_files: list[Path],
    cache_dir: Path | None,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    sample_fps: float,
    max_samples_per_replay: int | None,
    log_every_replays: int,
    checkpoint_every_replays: int,
    suppress_parser_output: bool = True,
    resume_checkpoint: Path | None = None,
) -> Path:
    action_parser = OptimizedDiscreteAction()
    model = ActorCritic(OBS_DIM, action_parser.summary.size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if resume_checkpoint is not None:
        load_checkpoint(resume_checkpoint, model, optimizer=optimizer, map_location=device)

    directories = create_run_directories(project_root, run_config)
    resolved_config = {
        **run_config,
        "pretraining": {
            "mode": "behavior_cloning_from_replays",
            "replay_count": len(replay_files),
            "validation_replay_count": len(validation_files),
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "sample_fps": float(sample_fps),
            "max_samples_per_replay": max_samples_per_replay,
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "observation_note": "boost amount and boost pad availability are reconstructed from replay trajectories",
        },
        "metadata": {
            "device": str(device),
            "obs_dim": OBS_DIM,
            "action_dim": int(action_parser.summary.size),
        },
    }
    snapshot_config(directories.config_snapshot, resolved_config)
    logger = TrainingLogger(directories)

    samples_seen = 0
    replays_seen = 0
    global_batch_index = 0

    for epoch_idx in range(epochs):
        epoch_files = list(replay_files)
        rng = np.random.default_rng(seed=epoch_idx + 42)
        rng.shuffle(epoch_files)
        epoch_stats = ReplayTrainStats()
        progress = ReplayProgressTracker(total=len(epoch_files))
        cached_replays = 0
        parsed_replays = 0

        progress.render(
            0,
            cached_replays=cached_replays,
            parsed_replays=parsed_replays,
            skipped_replays=epoch_stats.skipped_replays,
            samples_seen=samples_seen,
            current=f"epoch {epoch_idx + 1}/{epochs} initialisation",
        )

        for replay_idx, replay_path in enumerate(epoch_files, start=1):
            progress.render(
                replay_idx - 1,
                cached_replays=cached_replays,
                parsed_replays=parsed_replays,
                skipped_replays=epoch_stats.skipped_replays,
                samples_seen=samples_seen,
                current=f"parsing {replay_path.name}",
            )

            try:
                batch, from_cache = load_or_create_replay_batch(
                    replay_path,
                    cache_dir=cache_dir,
                    action_parser=action_parser,
                    sample_fps=sample_fps,
                    max_samples_per_replay=max_samples_per_replay,
                    suppress_parser_output=suppress_parser_output,
                )
            except Exception as exc:
                epoch_stats.skipped_replays += 1
                progress.newline()
                progress.write(f"[bc][skip] {replay_path.name}: {exc}")
                progress.render(
                    replay_idx,
                    cached_replays=cached_replays,
                    parsed_replays=parsed_replays,
                    skipped_replays=epoch_stats.skipped_replays,
                    samples_seen=samples_seen,
                    current=f"skip {replay_path.name}",
                )
                continue

            if from_cache:
                cached_replays += 1
            else:
                parsed_replays += 1

            if batch.sample_count == 0:
                epoch_stats.skipped_replays += 1
                progress.render(
                    replay_idx,
                    cached_replays=cached_replays,
                    parsed_replays=parsed_replays,
                    skipped_replays=epoch_stats.skipped_replays,
                    samples_seen=samples_seen,
                    current=f"empty {replay_path.name}",
                )
                continue

            replays_seen += 1
            epoch_stats.replays += 1
            permutation = rng.permutation(batch.sample_count)

            for start in range(0, batch.sample_count, batch_size):
                batch_indices = permutation[start : start + batch_size]
                observations = torch.as_tensor(batch.observations[batch_indices], dtype=torch.float32, device=device)
                actions = torch.as_tensor(batch.actions[batch_indices], dtype=torch.int64, device=device)

                logits, _ = model(observations)
                loss = F.cross_entropy(logits, actions)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                predictions = torch.argmax(logits, dim=-1)
                correct = int((predictions == actions).sum().item())
                batch_samples = int(actions.shape[0])
                samples_seen += batch_samples
                global_batch_index += 1
                epoch_stats.update(float(loss.item()), correct, batch_samples)

            progress.render(
                replay_idx,
                cached_replays=cached_replays,
                parsed_replays=parsed_replays,
                skipped_replays=epoch_stats.skipped_replays,
                samples_seen=samples_seen,
                current=f"done {replay_path.name}",
            )

            if replay_idx % max(log_every_replays, 1) == 0 or replay_idx == len(epoch_files):
                progress.newline()
                metrics = epoch_stats.as_metrics()
                metrics["epoch"] = float(epoch_idx + 1)
                metrics["replay_progress"] = float(replay_idx)
                logger.log_training(samples_seen, metrics)
                progress.write(
                    f"[bc] epoch={epoch_idx + 1}/{epochs} replays={replay_idx}/{len(epoch_files)} "
                    f"loss={metrics['bc_loss']:.4f} acc={metrics['bc_accuracy']:.3f} samples={int(metrics['bc_samples'])}"
                )

            if checkpoint_every_replays > 0 and replay_idx % checkpoint_every_replays == 0:
                progress.newline()
                save_checkpoint(
                    directories.checkpoint_dir / f"step_{samples_seen:012d}.pt",
                    model,
                    optimizer,
                    training_state={
                        "total_steps": 0,
                        "episodes_completed": 0,
                        "last_eval_step": 0,
                        "last_checkpoint_step": 0,
                        "pretraining_samples_seen": int(samples_seen),
                        "pretraining_replays_seen": int(replays_seen),
                        "pretraining_batches_seen": int(global_batch_index),
                    },
                    config=resolved_config,
                    seed=int(run_config["project"]["seed"]),
                )
                progress.write(f"[bc][checkpoint] saved at replay {replay_idx}/{len(epoch_files)}")

        progress.render(
            len(epoch_files),
            cached_replays=cached_replays,
            parsed_replays=parsed_replays,
            skipped_replays=epoch_stats.skipped_replays,
            samples_seen=samples_seen,
            current=f"epoch {epoch_idx + 1}/{epochs} complete",
            final=True,
        )

        if validation_files:
            progress.write(f"[bc][val] running on {len(validation_files)} replay(s)...")
            validation_metrics = evaluate_replay_policy(
                model,
                validation_files,
                cache_dir=cache_dir,
                action_parser=action_parser,
                sample_fps=sample_fps,
                max_samples_per_replay=max_samples_per_replay,
                batch_size=batch_size,
                device=device,
                suppress_parser_output=suppress_parser_output,
            )
            logger.log_evaluation(samples_seen, validation_metrics)
            progress.write(
                f"[bc][val] epoch={epoch_idx + 1}/{epochs} "
                f"loss={validation_metrics['validation_loss']:.4f} "
                f"acc={validation_metrics['validation_accuracy']:.3f}"
            )

    final_checkpoint = directories.checkpoint_dir / "final.pt"
    save_checkpoint(
        final_checkpoint,
        model,
        optimizer,
        training_state={
            "total_steps": 0,
            "episodes_completed": 0,
            "last_eval_step": 0,
            "last_checkpoint_step": 0,
            "pretraining_samples_seen": int(samples_seen),
            "pretraining_replays_seen": int(replays_seen),
            "pretraining_batches_seen": int(global_batch_index),
        },
        config=resolved_config,
        seed=int(run_config["project"]["seed"]),
    )
    logger.close()
    print(f"[bc][done] final checkpoint saved to {final_checkpoint}")
    return final_checkpoint











