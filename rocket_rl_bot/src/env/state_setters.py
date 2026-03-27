from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

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


def _set_car(car, x: float, y: float, z: float, yaw: float, boost: float, velocity=(0.0, 0.0, 0.0)) -> None:
    car.set_pos(float(x), float(y), float(z))
    car.set_rot(yaw=float(yaw))
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


class KickoffLikeSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (0.0, 0.0, 93.0))
        _set_car(blue_car, 0.0, -2048.0, 17.0, BLUE_YAW, 0.33)
        _set_car(orange_car, 0.0, 2048.0, 17.0, ORANGE_YAW, 0.33)


class OpenGoalAttackSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        lane_x = random.uniform(-1200.0, 1200.0)
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (lane_x * 0.30, 2050.0, 93.0), (0.0, 90.0, 0.0))
        _set_car(blue_car, lane_x, 250.0, 17.0, BLUE_YAW, 0.70, velocity=(0.0, 350.0, 0.0))
        _set_car(orange_car, lane_x * 0.25, 4400.0, 17.0, ORANGE_YAW, 0.10)


class SimpleDefenseSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        lane_x = random.uniform(-1800.0, 1800.0)
        ball_velocity_y = random.uniform(550.0, 1100.0)
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (lane_x * 0.35, -2200.0, 93.0), (0.0, ball_velocity_y, 0.0))
        _set_car(blue_car, lane_x * 0.35, -3900.0, 17.0, BLUE_YAW, 0.55, velocity=(0.0, 250.0, 0.0))
        _set_car(orange_car, lane_x * 0.25, -900.0, 17.0, ORANGE_YAW, 0.30, velocity=(0.0, 400.0, 0.0))


class BallCenterRandomSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        blue_x = random.uniform(-1800.0, 1800.0)
        orange_x = random.uniform(-1800.0, 1800.0)
        ball_velocity = (
            random.uniform(-350.0, 350.0),
            random.uniform(-550.0, 550.0),
            random.uniform(0.0, 160.0),
        )
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (0.0, 0.0, 93.0), ball_velocity)
        _set_car(blue_car, blue_x, -1800.0, 17.0, BLUE_YAW, 0.45)
        _set_car(orange_car, orange_x, 1800.0, 17.0, ORANGE_YAW, 0.45)


class WrongSideRecoverySetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_x = random.uniform(-1400.0, 1400.0)
        ball_y = random.uniform(-900.0, 900.0)
        ball_velocity = (
            random.uniform(-120.0, 120.0),
            random.uniform(-120.0, 120.0),
            0.0,
        )
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (ball_x, ball_y, BALL_Z), ball_velocity)

        blue_x = ball_x + random.uniform(-650.0, 650.0)
        blue_y = ball_y + random.uniform(550.0, 1150.0)
        orange_x = ball_x + random.uniform(-650.0, 650.0)
        orange_y = ball_y - random.uniform(550.0, 1150.0)

        blue_yaw = BLUE_YAW + random.uniform(-0.45, 0.45)
        orange_yaw = ORANGE_YAW + random.uniform(-0.45, 0.45)

        _set_car(
            blue_car,
            blue_x,
            blue_y,
            17.0,
            blue_yaw,
            0.55,
            velocity=(random.uniform(-120.0, 120.0), random.uniform(-80.0, 180.0), 0.0),
        )
        _set_car(
            orange_car,
            orange_x,
            orange_y,
            17.0,
            orange_yaw,
            0.55,
            velocity=(random.uniform(-120.0, 120.0), random.uniform(-180.0, 80.0), 0.0),
        )


class MisalignedRecoverySetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_x = random.uniform(-1500.0, 1500.0)
        ball_y = random.uniform(-1200.0, 1200.0)
        ball_velocity = (
            random.uniform(-80.0, 80.0),
            random.uniform(-80.0, 80.0),
            0.0,
        )
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (ball_x, ball_y, BALL_Z), ball_velocity)

        blue_x = ball_x + random.uniform(-900.0, 900.0)
        blue_y = ball_y - random.uniform(700.0, 1450.0)
        orange_x = ball_x + random.uniform(-900.0, 900.0)
        orange_y = ball_y + random.uniform(700.0, 1450.0)

        blue_face_ball = _yaw_to_target((blue_x, blue_y), (ball_x, ball_y))
        orange_face_ball = _yaw_to_target((orange_x, orange_y), (ball_x, ball_y))
        blue_yaw = blue_face_ball + random.choice([-1.0, 1.0]) * random.uniform(1.35, 2.75)
        orange_yaw = orange_face_ball + random.choice([-1.0, 1.0]) * random.uniform(1.35, 2.75)

        blue_speed = random.uniform(0.0, 350.0)
        orange_speed = random.uniform(0.0, 350.0)

        _set_car(blue_car, blue_x, blue_y, 17.0, blue_yaw, 0.50, velocity=_velocity_from_yaw(blue_yaw, blue_speed))
        _set_car(orange_car, orange_x, orange_y, 17.0, orange_yaw, 0.50, velocity=_velocity_from_yaw(orange_yaw, orange_speed))


class WallBallSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        side = random.choice([-1.0, 1.0])
        x = 3600.0 * side
        y = random.uniform(-1800.0, 1800.0)
        z = random.uniform(160.0, 450.0)
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (x, y, z), (0.0, random.uniform(-250.0, 250.0), random.uniform(-80.0, 80.0)))
        _set_car(blue_car, x - 420.0 * side, y - 650.0, 17.0, BLUE_YAW if side < 0 else 0.0, 0.60)
        _set_car(orange_car, x - 520.0 * side, y + 650.0, 17.0, ORANGE_YAW if side > 0 else math.pi, 0.60)


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

        for index, car in enumerate(state_wrapper.cars[:2]):
            team_sign = -1.0 if index == 0 else 1.0
            x = random.uniform(-2800.0, 2800.0)
            y = random.uniform(-4200.0, -600.0) if index == 0 else random.uniform(600.0, 4200.0)
            z = 17.0 if random.random() < 0.85 else random.uniform(220.0, 650.0)
            yaw = random.uniform(-math.pi, math.pi)
            boost = random.uniform(0.1, 0.9)
            velocity = (
                random.uniform(-900.0, 900.0),
                random.uniform(-950.0, 950.0) * team_sign,
                0.0 if z <= 20.0 else random.uniform(-300.0, 300.0),
            )
            _set_car(car, x, y, z, yaw, boost, velocity=velocity)


@dataclass
class CurriculumStage:
    name: str
    min_steps: int
    weights: Dict[str, float]


class WeightedSampleSetter(StateSetter):
    def __init__(self, curriculum_config: Dict) -> None:
        super().__init__()
        self.setters: Dict[str, StateSetter] = {
            "kickoff_like": KickoffLikeSetter(),
            "open_goal_attack": OpenGoalAttackSetter(),
            "simple_defense": SimpleDefenseSetter(),
            "ball_center_random": BallCenterRandomSetter(),
            "wrong_side_recovery": WrongSideRecoverySetter(),
            "misaligned_recovery": MisalignedRecoverySetter(),
            "wall_ball": WallBallSetter(),
            "random_match_state": RandomMatchStateSetter(),
        }
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

    def set_training_step(self, total_steps: int) -> None:
        self.current_step = int(total_steps)
        self.current_stage = self._resolve_stage(self.current_step)

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.current_stage.name,
            "min_steps": self.current_stage.min_steps,
            "weights": dict(self.current_stage.weights),
        }

    def reset(self, state_wrapper: StateWrapper) -> None:
        weights = self.current_stage.weights
        names: Sequence[str] = list(weights.keys())
        probabilities = np.asarray([weights[name] for name in names], dtype=np.float64)
        probabilities = probabilities / probabilities.sum()
        selected = str(np.random.choice(names, p=probabilities))
        self.setters[selected].reset(state_wrapper)

    def _resolve_stage(self, total_steps: int) -> CurriculumStage:
        eligible = [stage for stage in self.stages if total_steps >= stage.min_steps]
        return eligible[-1] if eligible else self.stages[0]
