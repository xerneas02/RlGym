from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np
from rlgym_compat import GameState

from ZZeer import TrainedAgent

from compact_obs_rlbot import CompactObservationBuilder


KICKOFF_CONTROLS = (
        11 * 6 * [SimpleControllerState(throttle=1, boost=True)]
        + 4 * 7 * [SimpleControllerState(throttle=1, boost=True, steer=-1)]
        + 2 * 8 * [SimpleControllerState(throttle=1, jump=True, boost=True)]
        + 1 * 8 * [SimpleControllerState(throttle=1, boost=True)]
        + 1 * 8 * [SimpleControllerState(throttle=1, yaw=0.8, pitch=-0.7, jump=True, boost=True)]
        + 13 * 6 * [SimpleControllerState(throttle=1, pitch=1, boost=True)]
        + 10 * 5 * [SimpleControllerState(throttle=1, roll=1, pitch=0.53)]
)

KICKOFF_NUMPY = np.array([
    [scs.throttle, scs.steer, scs.pitch, scs.yaw, scs.roll, scs.jump, scs.boost, scs.handbrake]
    for scs in KICKOFF_CONTROLS
])


class RLGymExampleBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        self.obs_builder = CompactObservationBuilder()
        self.agent = TrainedAgent()
        self.tick_skip = 8
        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.ticks = 0
        self.prev_time = 0
        self.expected_teammates = 0
        self.expected_opponents = 1
        self.current_obs = None
        self.kickoff_index = -1
        print(f"{self.name} Ready - Index: {index}")

    def initialize_agent(self):
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8, dtype=np.float32)
        self.kickoff_index = -1

    def reshape_state(self, gamestate, player, opponents, allies):
        closest_op = min(opponents, key=lambda p: np.linalg.norm(self.game_state.ball.position - p.car_data.position))
        self.game_state.players = [player, closest_op]

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = packet.game_info.frame_num
        delta = cur_time - self.prev_time
        self.prev_time = cur_time
        ticks_elapsed = self.ticks
        self.ticks += delta

        if ticks_elapsed > self.tick_skip - 1:
            self.game_state.decode(packet, ticks_elapsed)
            player = self.game_state.players[self.index]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]
            allies = [p for p in self.game_state.players if p.team_num == self.team and p.car_id != self.index]

            if len(opponents) != self.expected_opponents or len(allies) != self.expected_teammates:
                self.reshape_state(self.game_state, player, opponents, allies)

            self.current_obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(self.current_obs)
            self.update_controls(self.action)
            self.ticks = 0

        self.maybe_do_kickoff(packet, ticks_elapsed)

        return self.controls

    def maybe_do_kickoff(self, packet, ticks_elapsed):
        if packet.game_info.is_kickoff_pause:
            if self.kickoff_index >= 0:
                self.kickoff_index += round(ticks_elapsed)
            elif self.kickoff_index == -1:
                is_kickoff_taker = False
                ball_pos = np.array([packet.game_ball.physics.location.x, packet.game_ball.physics.location.y])
                positions = np.array([[car.physics.location.x, car.physics.location.y]
                                      for car in packet.game_cars[:packet.num_cars]])
                distances = np.linalg.norm(positions - ball_pos, axis=1)
                if abs(distances.min() - distances[self.index]) <= 10:
                    is_kickoff_taker = True
                    indices = np.argsort(distances)
                    for index in indices:
                        if abs(distances[index] - distances[self.index]) <= 10                                 and packet.game_cars[index].team == self.team                                 and index != self.index:
                            if self.team == 0:
                                is_left = positions[index, 0] < positions[self.index, 0]
                            else:
                                is_left = positions[index, 0] > positions[self.index, 0]
                            if not is_left:
                                is_kickoff_taker = False

                self.kickoff_index = 0 if is_kickoff_taker else -2

            if 0 <= self.kickoff_index < len(KICKOFF_NUMPY) and packet.game_ball.physics.location.y == 0:
                action = KICKOFF_NUMPY[self.kickoff_index]
                self.action = action
                self.update_controls(self.action)
        else:
            self.kickoff_index = -1

    def update_controls(self, action):
        self.controls.throttle = float(action[0])
        self.controls.steer = float(action[1])
        self.controls.pitch = float(action[2])
        self.controls.yaw = float(action[3])
        self.controls.roll = float(action[4])
        self.controls.jump = bool(action[5] > 0)
        self.controls.boost = bool(action[6] > 0)
        self.controls.handbrake = bool(action[7] > 0)
