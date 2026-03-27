from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from rlgym_sim.utils.common_values import BALL_RADIUS
    from rlgym_sim.utils.state_setters import StateSetter, StateWrapper
except ImportError:  # pragma: no cover
    from rlgym.utils.common_values import BALL_RADIUS
    from rlgym.utils.state_setters import StateSetter, StateWrapper


BLUE_YAW = math.pi / 2.0
ORANGE_YAW = -math.pi / 2.0
BALL_Z = float(BALL_RADIUS)


def _set_car(car, x: float, y: float, z: float, yaw: float, boost: float, velocity=(0.0, 0.0, 0.0), roll: float = 0.0) -> None:
    car.set_pos(float(x), float(y), float(z))
    car.set_rot(yaw=float(yaw), roll=float(roll))
    car.set_lin_vel(*[float(v) for v in velocity])
    car.set_ang_vel(0.0, 0.0, 0.0)
    car.boost = float(np.clip(boost, 0.0, 1.0))


def _set_ball(state_wrapper: StateWrapper, position, velocity=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)) -> None:
    state_wrapper.ball.set_pos(*[float(v) for v in position])
    state_wrapper.ball.set_lin_vel(*[float(v) for v in velocity])
    state_wrapper.ball.set_ang_vel(*[float(v) for v in angular])


def _yaw_to_target(source_xy, target_xy) -> float:
    return math.atan2(float(target_xy[1]) - float(source_xy[1]), float(target_xy[0]) - float(source_xy[0]))


def _velocity_from_yaw(yaw: float, speed: float) -> tuple[float, float, float]:
    return (float(math.cos(yaw) * speed), float(math.sin(yaw) * speed), 0.0)


def _kickoff_slots(scale: float = 1.0, inverted: bool = False) -> dict[int, list[tuple[tuple[float, float, float], float]]]:
    blue = [
        ((-2048.0 * scale, -2560.0 * scale, 17.0), -0.25 * math.pi if inverted else 0.25 * math.pi),
        ((2048.0 * scale, -2560.0 * scale, 17.0), -0.75 * math.pi if inverted else 0.75 * math.pi),
        ((-256.0 * scale, -3840.0 * scale, 17.0), -0.5 * math.pi if inverted else 0.5 * math.pi),
        ((256.0 * scale, -3840.0 * scale, 17.0), -0.5 * math.pi if inverted else 0.5 * math.pi),
        ((0.0, -4608.0 * scale, 17.0), -0.5 * math.pi if inverted else 0.5 * math.pi),
    ]
    orange = [
        ((2048.0 * scale, 2560.0 * scale, 17.0), 0.75 * math.pi if inverted else -0.75 * math.pi),
        ((-2048.0 * scale, 2560.0 * scale, 17.0), 0.25 * math.pi if inverted else -0.25 * math.pi),
        ((256.0 * scale, 3840.0 * scale, 17.0), 0.5 * math.pi if inverted else -0.5 * math.pi),
        ((-256.0 * scale, 3840.0 * scale, 17.0), 0.5 * math.pi if inverted else -0.5 * math.pi),
        ((0.0, 4608.0 * scale, 17.0), 0.5 * math.pi if inverted else -0.5 * math.pi),
    ]
    return {0: blue, 1: orange}


def _fill_support_cars(
    state_wrapper: StateWrapper,
    assigned_indices: set[int],
    *,
    boost: float = 0.33,
    scale: float = 1.0,
    inverted: bool = False,
) -> None:
    slots = _kickoff_slots(scale=scale, inverted=inverted)
    counters = {0: 0, 1: 0}
    for index, car in enumerate(state_wrapper.cars):
        if index in assigned_indices:
            continue
        team = int(getattr(car, "team_num", 0))
        slot_index = min(counters[team], len(slots[team]) - 1)
        position, yaw = slots[team][slot_index]
        _set_car(car, *position, yaw=yaw, boost=boost)
        counters[team] += 1


def _sample_range3(payload: dict[str, Any] | None, defaults: tuple[float, float, float]) -> tuple[float, float, float]:
    if not payload or not bool(payload.get("enabled", False)):
        return defaults
    return (
        float(random.uniform(float(payload.get("min_x", defaults[0])), float(payload.get("max_x", defaults[0])))),
        float(random.uniform(float(payload.get("min_y", defaults[1])), float(payload.get("max_y", defaults[1])))),
        float(random.uniform(float(payload.get("min_z", defaults[2])), float(payload.get("max_z", defaults[2])))),
    )


def _sample_range1(payload: dict[str, Any] | None, default: float) -> float:
    if not payload or not bool(payload.get("enabled", False)):
        return float(payload.get("value", default) if payload else default)
    return float(
        random.uniform(
            float(payload.get("min", payload.get("min_value", default))),
            float(payload.get("max", payload.get("max_value", default))),
        )
    )


class KickoffLikeSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        assigned: set[int] = set()
        _set_ball(state_wrapper, (0.0, 0.0, 93.0))
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, 0.0, -2048.0, 17.0, BLUE_YAW, 0.33)
            else:
                _set_car(car, 0.0, 2048.0, 17.0, ORANGE_YAW, 0.33)
        _fill_support_cars(state_wrapper, assigned, boost=0.33)


class OpenGoalAttackSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        lane_x = random.uniform(-1200.0, 1200.0)
        _set_ball(state_wrapper, (lane_x * 0.30, 2050.0, 93.0), (0.0, 90.0, 0.0))
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, lane_x, 250.0, 17.0, BLUE_YAW, 0.70, velocity=(0.0, 350.0, 0.0))
            else:
                _set_car(car, lane_x * 0.25, 4400.0, 17.0, ORANGE_YAW, 0.10)
        _fill_support_cars(state_wrapper, assigned, boost=0.25)


class SimpleDefenseSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        lane_x = random.uniform(-1800.0, 1800.0)
        ball_velocity_y = random.uniform(550.0, 1100.0)
        _set_ball(state_wrapper, (lane_x * 0.35, -2200.0, 93.0), (0.0, ball_velocity_y, 0.0))
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, lane_x * 0.35, -3900.0, 17.0, BLUE_YAW, 0.55, velocity=(0.0, 250.0, 0.0))
            else:
                _set_car(car, lane_x * 0.25, -900.0, 17.0, ORANGE_YAW, 0.30, velocity=(0.0, 400.0, 0.0))
        _fill_support_cars(state_wrapper, assigned, boost=0.33)


class BallCenterRandomSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        blue_x = random.uniform(-1800.0, 1800.0)
        orange_x = random.uniform(-1800.0, 1800.0)
        ball_velocity = (
            random.uniform(-350.0, 350.0),
            random.uniform(-550.0, 550.0),
            random.uniform(0.0, 160.0),
        )
        _set_ball(state_wrapper, (0.0, 0.0, 93.0), ball_velocity)
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, blue_x, -1800.0, 17.0, BLUE_YAW, 0.45)
            else:
                _set_car(car, orange_x, 1800.0, 17.0, ORANGE_YAW, 0.45)
        _fill_support_cars(state_wrapper, assigned, boost=0.33)


class WrongSideRecoverySetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_x = random.uniform(-1400.0, 1400.0)
        ball_y = random.uniform(-900.0, 900.0)
        ball_velocity = (
            random.uniform(-120.0, 120.0),
            random.uniform(-120.0, 120.0),
            0.0,
        )
        _set_ball(state_wrapper, (ball_x, ball_y, BALL_Z), ball_velocity)
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(
                    car,
                    ball_x + random.uniform(-650.0, 650.0),
                    ball_y + random.uniform(550.0, 1150.0),
                    17.0,
                    BLUE_YAW + random.uniform(-0.45, 0.45),
                    0.55,
                    velocity=(random.uniform(-120.0, 120.0), random.uniform(-80.0, 180.0), 0.0),
                )
            else:
                _set_car(
                    car,
                    ball_x + random.uniform(-650.0, 650.0),
                    ball_y - random.uniform(550.0, 1150.0),
                    17.0,
                    ORANGE_YAW + random.uniform(-0.45, 0.45),
                    0.55,
                    velocity=(random.uniform(-120.0, 120.0), random.uniform(-180.0, 80.0), 0.0),
                )
        _fill_support_cars(state_wrapper, assigned, boost=0.40)


class MisalignedRecoverySetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_x = random.uniform(-1500.0, 1500.0)
        ball_y = random.uniform(-1200.0, 1200.0)
        ball_velocity = (
            random.uniform(-80.0, 80.0),
            random.uniform(-80.0, 80.0),
            0.0,
        )
        _set_ball(state_wrapper, (ball_x, ball_y, BALL_Z), ball_velocity)
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                blue_x = ball_x + random.uniform(-900.0, 900.0)
                blue_y = ball_y - random.uniform(700.0, 1450.0)
                face_ball = _yaw_to_target((blue_x, blue_y), (ball_x, ball_y))
                yaw = face_ball + random.choice([-1.0, 1.0]) * random.uniform(1.35, 2.75)
                _set_car(car, blue_x, blue_y, 17.0, yaw, 0.50, velocity=_velocity_from_yaw(yaw, random.uniform(0.0, 350.0)))
            else:
                orange_x = ball_x + random.uniform(-900.0, 900.0)
                orange_y = ball_y + random.uniform(700.0, 1450.0)
                face_ball = _yaw_to_target((orange_x, orange_y), (ball_x, ball_y))
                yaw = face_ball + random.choice([-1.0, 1.0]) * random.uniform(1.35, 2.75)
                _set_car(car, orange_x, orange_y, 17.0, yaw, 0.50, velocity=_velocity_from_yaw(yaw, random.uniform(0.0, 350.0)))
        _fill_support_cars(state_wrapper, assigned, boost=0.40)


class WallBallSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        side = random.choice([-1.0, 1.0])
        x = 3600.0 * side
        y = random.uniform(-1800.0, 1800.0)
        z = random.uniform(160.0, 450.0)
        _set_ball(state_wrapper, (x, y, z), (0.0, random.uniform(-250.0, 250.0), random.uniform(-80.0, 80.0)))
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, x - 420.0 * side, y - 650.0, 17.0, BLUE_YAW if side < 0 else 0.0, 0.60)
            else:
                _set_car(car, x - 520.0 * side, y + 650.0, 17.0, ORANGE_YAW if side > 0 else math.pi, 0.60)
        _fill_support_cars(state_wrapper, assigned, boost=0.40)


class RandomMatchStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_position = (
            random.uniform(-3200.0, 3200.0),
            random.uniform(-4200.0, 4200.0),
            random.uniform(93.0, 420.0),
        )
        ball_velocity = (
            random.uniform(-1200.0, 1200.0),
            random.uniform(-1400.0, 1400.0),
            random.uniform(-250.0, 450.0),
        )
        ball_angular = (
            random.uniform(-3.0, 3.0),
            random.uniform(-3.0, 3.0),
            random.uniform(-3.0, 3.0),
        )
        _set_ball(state_wrapper, ball_position, ball_velocity, ball_angular)

        for car in state_wrapper.cars:
            team = int(getattr(car, "team_num", 0))
            team_sign = -1.0 if team == 0 else 1.0
            x = random.uniform(-2800.0, 2800.0)
            y = random.uniform(-4200.0, -600.0) if team == 0 else random.uniform(600.0, 4200.0)
            z = 17.0 if random.random() < 0.85 else random.uniform(220.0, 650.0)
            yaw = random.uniform(-math.pi, math.pi)
            boost = random.uniform(0.1, 0.9)
            velocity = (
                random.uniform(-900.0, 900.0),
                random.uniform(-950.0, 950.0) * team_sign,
                0.0 if z <= 20.0 else random.uniform(-300.0, 300.0),
            )
            _set_car(car, x, y, z, yaw, boost, velocity=velocity)


class BetterRandomSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        _set_ball(
            state_wrapper,
            (
                float(np.random.uniform(-3200.0, 3200.0)),
                float(np.random.uniform(-4250.0, 4250.0)),
                float(np.random.triangular(BALL_RADIUS, BALL_RADIUS, 1850.0)),
            ),
        )
        state_wrapper.ball.set_lin_vel(*np.random.normal(0.0, 900.0, size=3).astype(np.float32))
        state_wrapper.ball.set_ang_vel(*np.random.normal(0.0, 2.5, size=3).astype(np.float32))
        for car in state_wrapper.cars:
            _set_car(
                car,
                float(np.random.uniform(-3200.0, 3200.0)),
                float(np.random.uniform(-4250.0, 4250.0)),
                17.0,
                float(np.random.uniform(-math.pi, math.pi)),
                float(np.random.uniform(0.0, 1.0)),
                velocity=tuple(np.random.normal(0.0, 1000.0, size=3).astype(np.float32)),
            )


class LineStateSetter(StateSetter):
    def __init__(self, width: float = 1200.0) -> None:
        super().__init__()
        self.width = float(width)

    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_x = float(random.randint(-3800, 3800))
        ball_y = float(random.randint(-int(self.width), int(self.width)))
        ball_z = float(random.randint(94, 850))
        _set_ball(state_wrapper, (ball_x, ball_y, ball_z))
        for car in state_wrapper.cars:
            team = int(getattr(car, "team_num", 0))
            team_sign = -1.0 if team == 0 else 1.0
            _set_car(
                car,
                float(random.randint(-3800, 3800)),
                float(2300.0 * team_sign + random.randint(-int(self.width), int(self.width))),
                17.0,
                BLUE_YAW if team == 0 else ORANGE_YAW,
                0.33,
            )


class LegacyTrainingPatternSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        car_x = float(random.randint(-500, 500))
        pattern = random.randint(0, 2)
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            team = int(getattr(car, "team_num", 0))
            if pattern == 0:
                if team == 0:
                    _set_car(car, car_x, 0.0, 17.0, BLUE_YAW, 0.25)
                else:
                    _set_car(car, 0.0, 4260.0, 17.0, ORANGE_YAW, 0.25)
                _set_ball(state_wrapper, (car_x, 2816.0, 70.0))
            elif pattern == 1:
                if team == 0:
                    _set_car(car, 0.0, -5120.0, 17.0, BLUE_YAW, 0.50)
                else:
                    _set_car(car, 0.0, -2500.0, 17.0, ORANGE_YAW, 0.50)
                _set_ball(state_wrapper, (0.0, -2816.0, 70.0), (float(random.randint(-200, 200)), float(random.randint(100, 1500)), 0.0))
            else:
                if team == 0:
                    _set_car(car, 0.0, -1024.0, 30.0, BLUE_YAW, 0.50)
                else:
                    _set_car(car, 0.0, 1024.0, 30.0, ORANGE_YAW, 0.50)
                _set_ball(state_wrapper, (0.0, -960.0, 70.0))
        _fill_support_cars(state_wrapper, assigned, boost=0.33)


class BoomerBallSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        direction = random.choice([-1.0, 1.0])
        _set_ball(
            state_wrapper,
            (float(random.randint(-1500, 1500)), float(random.randint(-2500, 2500)), float(random.randint(220, 950))),
            (2600.0 * direction, float(random.randint(-2800, 2800)), float(random.randint(-200, 550))),
            (0.0, 0.0, float(random.randint(-5, 5))),
        )
        assigned: set[int] = set()
        for index, car in enumerate(state_wrapper.cars[:2]):
            assigned.add(index)
            if int(getattr(car, "team_num", 0)) == 0:
                _set_car(car, -1200.0, -3200.0, 17.0, BLUE_YAW, 0.85, velocity=(0.0, 900.0, 0.0))
            else:
                _set_car(car, 1200.0, 3200.0, 17.0, ORANGE_YAW, 0.85, velocity=(0.0, -900.0, 0.0))
        _fill_support_cars(state_wrapper, assigned, boost=0.85)


def _apply_scenario_payload(state_wrapper: StateWrapper, payload: dict[str, Any], mirrored: bool = False) -> None:
    ball = dict(payload.get("ball", {}))
    position_payload = dict(ball.get("position", {}))
    ball_position = _sample_range3(
        ball.get("position_random"),
        (
            float(position_payload.get("x", 0.0)),
            float(position_payload.get("y", 0.0)),
            float(position_payload.get("z", BALL_Z)),
        ),
    )
    if mirrored:
        ball_position = (-ball_position[0], -ball_position[1], ball_position[2])
    velocity_payload = dict(ball.get("velocity", {}))
    angular_payload = dict(ball.get("angular_velocity", {}))
    _set_ball(
        state_wrapper,
        ball_position,
        (
            float(velocity_payload.get("x", 0.0)),
            float(velocity_payload.get("y", 0.0)),
            float(velocity_payload.get("z", 0.0)),
        ),
        (
            float(angular_payload.get("x", 0.0)),
            float(angular_payload.get("y", 0.0)),
            float(angular_payload.get("z", 0.0)),
        ),
    )
    assigned: set[int] = set()
    per_team_targets: dict[int, list[Any]] = {0: [], 1: []}
    for car in state_wrapper.cars:
        per_team_targets[int(getattr(car, "team_num", 0))].append(car)
    for car_payload in payload.get("cars", []):
        team = int(car_payload.get("team", 0))
        if not per_team_targets[team]:
            continue
        car = per_team_targets[team].pop(0)
        base_position_payload = dict(car_payload.get("position", {}))
        car_position = _sample_range3(
            car_payload.get("position_random"),
            (
                float(base_position_payload.get("x", 0.0)),
                float(base_position_payload.get("y", 0.0)),
                float(base_position_payload.get("z", 17.0)),
            ),
        )
        yaw = _sample_range1(dict(car_payload.get("yaw", {})), BLUE_YAW if team == 0 else ORANGE_YAW)
        if mirrored:
            car_position = (-car_position[0], -car_position[1], car_position[2])
            yaw = (yaw + math.pi) % (2.0 * math.pi)
        velocity = dict(car_payload.get("velocity", {}))
        _set_car(
            car,
            car_position[0],
            car_position[1],
            car_position[2],
            yaw,
            _sample_range1(dict(car_payload.get("boost", {})), 0.33),
            velocity=(float(velocity.get("x", 0.0)), float(velocity.get("y", 0.0)), float(velocity.get("z", 0.0))),
        )
        try:
            assigned.add(state_wrapper.cars.index(car))
        except ValueError:
            pass
    _fill_support_cars(state_wrapper, assigned)


class ScenarioPayloadSetter(StateSetter):
    def __init__(self, scenario_payload: dict[str, Any], mirrored: bool = False) -> None:
        super().__init__()
        self.scenario_payload = dict(scenario_payload or {})
        self.mirrored = bool(mirrored)

    def reset(self, state_wrapper: StateWrapper) -> None:
        _apply_scenario_payload(state_wrapper, self.scenario_payload, mirrored=self.mirrored)


class ScenarioFileSetter(StateSetter):
    def __init__(self, scenario_path: str | Path, mirrored: bool = False) -> None:
        super().__init__()
        self.scenario_path = Path(scenario_path)
        self.mirrored = bool(mirrored)

    def reset(self, state_wrapper: StateWrapper) -> None:
        import json

        if not self.scenario_path.exists():
            raise FileNotFoundError(f"Scenario introuvable: {self.scenario_path}")
        payload = json.loads(self.scenario_path.read_text(encoding="utf-8"))
        _apply_scenario_payload(state_wrapper, payload, mirrored=self.mirrored)


class ReplayCSVStateSetter(StateSetter):
    def __init__(self, replay_folder: str | Path, mirrored: bool = False) -> None:
        super().__init__()
        self.replay_folder = Path(replay_folder)
        self.mirrored = bool(mirrored)

    def reset(self, state_wrapper: StateWrapper) -> None:
        csv_files = [path for path in self.replay_folder.rglob("*.csv") if path.is_file()]
        if not csv_files:
            raise FileNotFoundError(f"Aucun CSV replay dans {self.replay_folder}")
        csv_path = random.choice(csv_files)
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]
        if not rows:
            raise ValueError(f"Replay CSV vide: {csv_path}")
        row = random.choice(rows)
        players = sorted({key[:-6] for key in row.keys() if key.endswith("pos_x") and not key.startswith("ball_")})
        if len(players) < 2:
            raise ValueError(f"Replay CSV inexploitable: {csv_path}")

        def _value(key: str, default: float = 0.0) -> float:
            try:
                return float(row.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def _mirror_xy(x: float, y: float) -> tuple[float, float]:
            return (-x, -y) if self.mirrored else (x, y)

        ball_x, ball_y = _mirror_xy(_value("ball_pos_x"), _value("ball_pos_y"))
        ball_vx, ball_vy = _mirror_xy(_value("ball_vel_x"), _value("ball_vel_y"))
        _set_ball(
            state_wrapper,
            (ball_x, ball_y, _value("ball_pos_z", BALL_Z)),
            (ball_vx, ball_vy, _value("ball_vel_z")),
            (_value("ball_ang_vel_x"), _value("ball_ang_vel_y"), _value("ball_ang_vel_z")),
        )
        assigned: set[int] = set()
        for index, player_name in enumerate(players[: len(state_wrapper.cars)]):
            car = state_wrapper.cars[index]
            pos_x, pos_y = _mirror_xy(_value(f"{player_name}_pos_x"), _value(f"{player_name}_pos_y"))
            vel_x, vel_y = _mirror_xy(_value(f"{player_name}_vel_x"), _value(f"{player_name}_vel_y"))
            yaw = _value(f"{player_name}_rot_y")
            if self.mirrored:
                yaw = (yaw + math.pi) % (2.0 * math.pi)
            _set_car(
                car,
                pos_x,
                pos_y,
                _value(f"{player_name}_pos_z", 17.0),
                yaw,
                _value(f"{player_name}_boost", 33.0) / 100.0,
                velocity=(vel_x, vel_y, _value(f"{player_name}_vel_z")),
            )
            car.set_ang_vel(_value(f"{player_name}_ang_vel_x"), _value(f"{player_name}_ang_vel_y"), _value(f"{player_name}_ang_vel_z"))
            assigned.add(index)
        _fill_support_cars(state_wrapper, assigned)


@dataclass
class CurriculumStage:
    name: str
    min_steps: int
    weights: Dict[str, float]


class WeightedSampleSetter(StateSetter):
    def __init__(self, curriculum_config: Dict, project_root: str | Path | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])
        self.setters: Dict[str, StateSetter] = {
            "kickoff_like": KickoffLikeSetter(),
            "open_goal_attack": OpenGoalAttackSetter(),
            "simple_defense": SimpleDefenseSetter(),
            "ball_center_random": BallCenterRandomSetter(),
            "wrong_side_recovery": WrongSideRecoverySetter(),
            "misaligned_recovery": MisalignedRecoverySetter(),
            "wall_ball": WallBallSetter(),
            "random_match_state": RandomMatchStateSetter(),
            "better_random": BetterRandomSetter(),
            "legacy_training_state": LegacyTrainingPatternSetter(),
            "line_state": LineStateSetter(),
            "boomer_ball": BoomerBallSetter(),
        }
        self.setters.update(self._build_custom_setters(curriculum_config))
        self.stages: List[CurriculumStage] = [
            CurriculumStage(
                name=stage["name"],
                min_steps=int(stage["min_steps"]),
                weights={key: float(value) for key, value in stage["weights"].items()},
            )
            for stage in curriculum_config["stages"]
        ]
        self.current_stage = self.stages[0]
        self.current_step = 0
        self._forced_next_state: str | None = None

    def _build_custom_setters(self, curriculum_config: Dict[str, Any]) -> Dict[str, StateSetter]:
        extra: Dict[str, StateSetter] = {}
        definitions = dict(curriculum_config.get("state_definitions", {}))
        for state_id, definition in definitions.items():
            payload = dict(definition or {})
            kind = str(payload.get("kind", "builtin")).strip().lower()
            if kind == "scenario":
                scenario_path = payload.get("path")
                if scenario_path:
                    extra[state_id] = ScenarioFileSetter((self.project_root / scenario_path).resolve(), mirrored=bool(payload.get("mirrored", False)))
            elif kind == "scenario_inline":
                extra[state_id] = ScenarioPayloadSetter(dict(payload.get("definition", {})), mirrored=bool(payload.get("mirrored", False)))
            elif kind == "replay_csv":
                replay_folder = payload.get("replay_folder", "DataState")
                extra[state_id] = ReplayCSVStateSetter((self.project_root / replay_folder).resolve(), mirrored=bool(payload.get("mirrored", False)))
            elif kind == "builtin":
                base_id = str(payload.get("base_id", "")).strip()
                params = dict(payload.get("params", {}))
                if base_id == "line_state":
                    extra[state_id] = LineStateSetter(width=float(params.get("width", 1200.0)))
                elif base_id == "boomer_ball":
                    extra[state_id] = BoomerBallSetter()
                elif base_id == "better_random":
                    extra[state_id] = BetterRandomSetter()
                elif base_id == "legacy_training_state":
                    extra[state_id] = LegacyTrainingPatternSetter()
                elif base_id == "scenario_file":
                    scenario_path = params.get("path") or payload.get("path")
                    if scenario_path:
                        extra[state_id] = ScenarioFileSetter((self.project_root / scenario_path).resolve(), mirrored=bool(params.get("mirrored", False)))
        scenario_dir = self.project_root / "configs" / "scenarios"
        if scenario_dir.exists():
            for scenario_path in sorted(scenario_dir.glob("*.json")):
                state_id = f"scenario:{scenario_path.stem}"
                extra.setdefault(state_id, ScenarioFileSetter(scenario_path))
                extra.setdefault(f"{state_id}:mirrored", ScenarioFileSetter(scenario_path, mirrored=True))
        return extra

    def set_training_step(self, total_steps: int) -> None:
        self.current_step = int(total_steps)
        self.current_stage = self._resolve_stage(self.current_step)

    def force_next_state(self, state_id: str) -> None:
        if state_id in self.setters:
            self._forced_next_state = state_id

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.current_stage.name,
            "min_steps": self.current_stage.min_steps,
            "weights": dict(self.current_stage.weights),
            "available_states": sorted(self.setters.keys()),
            "forced_next_state": self._forced_next_state,
        }

    def reset(self, state_wrapper: StateWrapper) -> None:
        if self._forced_next_state and self._forced_next_state in self.setters:
            selected = self._forced_next_state
            self._forced_next_state = None
            self.setters[selected].reset(state_wrapper)
            return

        weights = {name: value for name, value in self.current_stage.weights.items() if name in self.setters and float(value) > 0.0}
        if not weights:
            self.setters["kickoff_like"].reset(state_wrapper)
            return
        names: Sequence[str] = list(weights.keys())
        probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        selected = str(np.random.choice(names, p=probabilities))
        self.setters[selected].reset(state_wrapper)

    def _resolve_stage(self, total_steps: int) -> CurriculumStage:
        eligible = [stage for stage in self.stages if total_steps >= stage.min_steps]
        return eligible[-1] if eligible else self.stages[0]

