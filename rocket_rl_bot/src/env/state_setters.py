from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
        _set_ball(state_wrapper, (lane_x * 0.35, 2450.0, 93.0), (0.0, 180.0, 0.0))
        _set_car(blue_car, lane_x, 800.0, 17.0, BLUE_YAW, 0.55, velocity=(0.0, 300.0, 0.0))
        _set_car(orange_car, lane_x * 0.4, 4300.0, 17.0, ORANGE_YAW, 0.15)


class SimpleDefenseSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        lane_x = random.uniform(-1800.0, 1800.0)
        ball_velocity_y = random.uniform(800.0, 1500.0)
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (lane_x * 0.4, -2700.0, 93.0), (0.0, ball_velocity_y, 0.0))
        _set_car(blue_car, lane_x * 0.5, -4300.0, 17.0, BLUE_YAW, 0.45)
        _set_car(orange_car, lane_x * 0.3, -1200.0, 17.0, ORANGE_YAW, 0.40, velocity=(0.0, 600.0, 0.0))


class BallCenterRandomSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        blue_x = random.uniform(-2200.0, 2200.0)
        orange_x = random.uniform(-2200.0, 2200.0)
        ball_velocity = (
            random.uniform(-600.0, 600.0),
            random.uniform(-900.0, 900.0),
            random.uniform(0.0, 300.0),
        )
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (0.0, 0.0, 93.0), ball_velocity)
        _set_car(blue_car, blue_x, -2200.0, 17.0, BLUE_YAW, 0.35)
        _set_car(orange_car, orange_x, 2200.0, 17.0, ORANGE_YAW, 0.35)


class WallBallSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        side = random.choice([-1.0, 1.0])
        x = 3600.0 * side
        y = random.uniform(-1800.0, 1800.0)
        z = random.uniform(300.0, 900.0)
        blue_car, orange_car = state_wrapper.cars[0], state_wrapper.cars[1]
        _set_ball(state_wrapper, (x, y, z), (0.0, random.uniform(-500.0, 500.0), random.uniform(-150.0, 150.0)))
        _set_car(blue_car, x - 450.0 * side, y - 900.0, 17.0, BLUE_YAW if side < 0 else 0.0, 0.55)
        _set_car(orange_car, x - 600.0 * side, y + 900.0, 17.0, ORANGE_YAW if side > 0 else math.pi, 0.55)


class RandomMatchStateSetter(StateSetter):
    def reset(self, state_wrapper: StateWrapper) -> None:
        ball_position = (
            random.uniform(-3200.0, 3200.0),
            random.uniform(-4200.0, 4200.0),
            random.uniform(93.0, 700.0),
        )
        ball_velocity = (
            random.uniform(-1600.0, 1600.0),
            random.uniform(-1800.0, 1800.0),
            random.uniform(-400.0, 900.0),
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
            z = random.choice([17.0, 17.0, 17.0, random.uniform(250.0, 900.0)])
            yaw = random.uniform(-math.pi, math.pi)
            boost = random.uniform(0.1, 0.9)
            velocity = (
                random.uniform(-1100.0, 1100.0),
                random.uniform(-1100.0, 1100.0) * team_sign,
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
