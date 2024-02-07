import rlgym
from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import common_conditions
from Observer import *
from State import TrainingStateSetter
from Reward import *
from Terminal import *
from Action import ZeerLookupAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.gym import Gym
from rlgym.gamelaunch import LaunchPreference
from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
import os

FRAME_SKIP = 8

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
                KickoffReward(),
                VelocityReward(),
                BoostAmountReward(),
                ForwardVelocityReward()
            ),
            (
                1.50    ,  # GoalScoredReward (Si le bot marque un but)
                0.1     ,  # BoostDifferenceReward (Si le bot collect ou utilise du boost)
                0.2     ,  # BallTouchReward (Si le bot touche la balle (reward varie en fonction de la hauteur de la balle))
                0.3     ,  # DemoReward (Si le bot démo)
                0.0080  ,  # DistancePlayerBallReward (Si le bot est proche de la balle)
                0.0080  ,  # DistanceBallGoalReward (Si la balle est proche du but adverse)
                0.000625,  # FacingBallReward (Si le bot fait face à la balle)
                0.0025  ,  # AlignBallGoalReward (Si le bot est entre ses buts et la balle [mais il y a une ligne qui relie le bot, la balle et le but])
                0.0015 ,  # ClosestToBallReward (Si plus proche de la balle par rapport aux adversaires)
                0.00125 ,  # TouchedLastReward (Si le bot est le dernier à avoir touché la balle)
                0.00125 ,  # BehindBallReward (Si le bot est entre la balle et son but)
                0.00125 ,  # VelocityPlayerBallReward (Si le bot va dans la même direction de la balle)
                0.2     ,  # KickoffReward (Si le bot gagne le kickoff)
                0.000625,  # VelocityReward (Si le bot bouge)
                0.00125 ,  # BoostAmountReward (Si le bot à du boost)
                0.0015     # ForwardVelocityReward (Si le bot bouge dans la direction de la balle (dans la bonne direction), penalise la marche arrière)
            )
        ),
        terminal_conditions = (common_conditions.TimeoutCondition(150), 
                               common_conditions.GoalScoredCondition()),
        obs_builder         = ZeerObservations(),
        state_setter        = DefaultState(),#DefaultState(),
        action_parser       = ZeerLookupAction(),#LookupAction(),
        spawn_opponents     = True,
        tick_skip          = FRAME_SKIP
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
    else:
        print("Not found")
    
    file_model_name = "kickoff"
    
    nbRep = 50
    
    A = 120 / FRAME_SKIP
    T = 10
    #gamma = lambda x: np.exp(np.log10(0.5)/((T+x)*A))
    gamma = np.exp(np.log10(0.5)/((T)*A))
    
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=1, wait_time=40, force_paging=True)
    env = VecCheckNan(env) # Checks for nans in tensor
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Normalize rewards
    env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard
    #env = get_gym(100)
    
    for i in range(nbRep):
        print(f"{i}/{nbRep}")
        
        try:
            model = PPO.load(f"models/{file_model_name}", env=env, verbose=1, device=torch.device("cuda:0"), custom_objects={"gamma": gamma} ) # gamma(i//(nbRep/10))
        except:
            model = PPO('MlpPolicy', env, n_epochs=10, learning_rate=5e-5, ent_coef=0.01, vf_coef=1., gamma=gamma, clip_range= 0.2, verbose=1, tensorboard_log="logs",  device="cuda" )
        
        model.learn(total_timesteps=int(1e5), progress_bar=True)
        model.save(f"models/{file_model_name}")

        
