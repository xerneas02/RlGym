import rlgym
from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
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
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym.utils.reward_functions.combined_reward import CombinedReward
import os


def get_match(game_speed=100):

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
                1.25    ,  # GoalScoredReward
                0.1     ,  # BoostDifferenceReward
                0.1     ,  # BallTouchReward
                0.3     ,  # DemoReward
                0.0025  ,  # DistancePlayerBallReward
                0.0025  ,  # DistanceBallGoalReward
                0.000625,  # FacingBallReward
                0.0025  ,  # AlignBallGoalReward
                0.00125 ,  # ClosestToBallReward
                0.00125 ,  # TouchedLastReward
                0.00125 ,  # BehindBallReward
                0.00125 ,  # VelocityPlayerBallReward
                0.000625,  # VelocityReward
                0.00125 ,  # BoostAmountReward
                0.0015     # ForwardVelocityReward
            )
        ),
        spawn_opponents=True,
        terminal_conditions = (common_conditions.TimeoutCondition(150),
                               NoTouchOrGoalTimeoutCondition(50),
                               ),
        obs_builder         = DefaultObs(),
        state_setter        = DefaultState(),
        action_parser       = LookupAction(),
    )
    
    return match

def get_gym(game_speed=100):
    return Gym(get_match(game_speed), 
               pipe_id=os.getpid(), 
               launch_preference=LaunchPreference.EPIC,
               use_injector=False, 
               force_paging=False, 
               raise_on_crash=False, 
               auto_minimize=False
               )
    

if __name__ == "__main__":

    #pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    if(torch.cuda.is_available()):
        print( torch.cuda.current_device())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    
    nbRep = 300

    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=1, wait_time=100, force_paging=True)
    #env = get_gym(100)

    model = PPO("MlpPolicy", env=env, verbose=1, device="cpu")
    for i in range(nbRep):
        print(f"{i}/{nbRep}")
        #model = PPO.load("ZZeer/rl_model", env=env, verbose=1)
        model.learn(total_timesteps=int(1e5), progress_bar=True)
        model.save("ZZeer/rl_model")

        
