import rlgym
from rlgym.envs import Match
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import common_conditions
from Observer import *
from State import TrainingStateSetter
from Reward import *
from Terminal import *
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.gym import Gym
from rlgym.gamelaunch import LaunchPreference
from rlgym.utils.reward_functions.combined_reward import CombinedReward
import os


def get_match(game_speed):

    match = Match(
        game_speed          = game_speed,
        reward_function = CombinedReward(
            (
                GoalScoredReward(),
                BoostDifferenceReward(),
                BallTouchReward(),
                DemoReward(),
                DistancePlayerBallReward(),
                DistanceBallGoalReward(),
                FacingBallReward(),
                AlignBallGoalReward(),
                ClosestToBallReward(),
                TouchedLastReward(),
                BehindBallReward(),
                VelocityPlayerBallReward(),
                VelocityReward(),
                BoostAmountReward(),
                ForwardVelocityReward()
            ),
            (
                1.0,  # GoalScoredReward
                0.1,  # BoostDifferenceReward
                0.3,  # BallTouchReward
                0.1,  # DemoReward
                0.2,  # DistancePlayerBallReward
                0.2,  # DistanceBallGoalReward
                0.5,  # FacingBallReward
                0.7,  # AlignBallGoalReward
                0.5,  # ClosestToBallReward
                0.1,  # TouchedLastReward
                0.5,  # BehindBallReward
                0.3,  # VelocityPlayerBallReward
                0.4,  # VelocityReward
                0.3,  # BoostAmountReward
                0.4   # ForwardVelocityReward
            )
        ),
        terminal_conditions = (common_conditions.TimeoutCondition(150), NoTouchOrGoalTimeoutCondition(50)),
        obs_builder         = DefaultObs(),
        state_setter        = TrainingStateSetter(),
        action_parser       = DefaultAction(),
        spawn_opponents     = False
    )
    
    return match

def get_gym(game_speed):
    return Gym(get_match(game_speed), pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC, use_injector=False, force_paging=False, raise_on_crash=False, auto_minimize=False)
    

if __name__ == "__main__":
    #env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=1, wait_time=40, force_paging=True)
    env = get_gym(100)
    
    nbRep = 1

    for i in range(nbRep):
        print(f"{i}/{nbRep}")
        model = PPO.load("ZZeer/rl_model", env=env, verbose=1)
        #model = PPO("MlpPolicy", env=env, verbose=1)
        model.learn(total_timesteps=int(1e5), progress_bar=True)
        model.save("ZZeer/rl_model")

        
