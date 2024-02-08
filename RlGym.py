from rlgym.envs import Match
from rlgym.gym import Gym
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.gamelaunch import LaunchPreference

from rlgym_tools.extra_action_parsers.lookup_act import LookupAction
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback, StopTrainingOnNoModelImprovement

from Observer import *
from State import CombinedState, BetterRandom, StateSetterInit, TrainingStateSetter, DefaultStateClose, RandomState, InvertedState 
from Reward import *
from Terminal import *
from Action import ZeerLookupAction
from Callback import HParamCallback
from Constante import *

import os


rewards = CombinedReward(
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
                ForwardVelocityReward(),
                FirstTouchReward(),
                DontTouchPenalityReward(),
                AirPenalityReward(),
            ),
            (
                1.45    ,  # GoalScoredReward
                0.0025  ,  # BoostDifferenceReward 
                1       ,  # BallTouchReward
                0.3     ,  # DemoReward
                0.05    ,  # DistancePlayerBallReward
                0.0025  ,  # DistanceBallGoalReward
                0.000625,  # FacingBallReward
                0.0025  ,  # AlignBallGoalReward
                0.00125 ,  # ClosestToBallReward
                0.00125 ,  # TouchedLastReward
                0.00125 ,  # BehindBallReward
                0.00125 ,  # VelocityPlayerBallReward
                0.0025  ,  # KickoffReward (0.1)
                0.0025  ,  # VelocityReward (0.000625)
                0.00125 ,  # BoostAmountReward
                0.005   ,  # ForwardVelocityReward
                1       ,  # FirstTouchReward
                0.3     ,  # DontTouchPenalityReward
                0       ,  # AirPenality
            ),
            verbose=0
        )

def get_match(game_speed=GAME_SPEED):

    match = Match(
        game_speed          = game_speed,
        reward_function     = rewards,
        terminal_conditions = (common_conditions.TimeoutCondition(2000), NoGoalTimeoutCondition(300, 1)), #NoTouchFirstTimeoutCondition(50)² #common_conditions.GoalScoredCondition(), common_conditions.NoTouchTimeoutCondition(80)
        obs_builder         = ZeerObservations(),
        state_setter        = CombinedState( 
                                rewards,   
                                (                   #42 Garde coef par defaut
                                    (DefaultStateClose(),   (0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 3, 42, 42)),
                                    (TrainingStateSetter(), (42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42)),
                                    (RandomState(),         (0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 42)),
                                    (InvertedState(),       (0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 3, 42, 42)),
                                    (GoaliePracticeState(), (0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42)), 
                                    (HoopsLikeSetter(),     (42, 3, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42)),
                                    (BetterRandom(),        (42, 3, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42)),
                                    (KickoffLikeSetter(),   (0, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 3, 42, 42)),
                                    (WallPracticeState(),   (42, 3, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 0, 0, 42)),
                                ),
                                (
                                    0.11, #DefaultStateClose
                                    0.12, #TrainingStateSetter
                                    0.11, #RandomState
                                    0.11, #InvertedState
                                    0.11, #GoaliePracticeState
                                    0.11, #HoopsLikeSetter
                                    0.11, #BetterRandom
                                    0.11, #KickoffLikeSetter
                                    0.11, #WallPracticeState
                                )
                             ),
        action_parser       = ZeerLookupAction(),#LookupAction(),
        spawn_opponents     = True,
        tick_skip           = FRAME_SKIP
    )
    
    return match

def get_gym(game_speed=GAME_SPEED):
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
    
    file_model_name = "rl_model"
    
    nbRep = 1000
    
    save_periode = 1e5
    
    fps = 120 / FRAME_SKIP
    T = 20
    #gamma = lambda x: np.exp(np.log10(0.5)/((T+x)*A))
    gamma = np.exp(np.log10(0.5)/(T*fps))
    
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=NUM_INSTANCE, wait_time=40, force_paging=True)
    env = VecCheckNan(env) # Checks for nans in tensor
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Normalize rewards
    env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard
    #env = get_gym(100)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_periode/2,
        save_path=f"./models/{file_model_name}",
        name_prefix=file_model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    
    stopTraining = StopTrainingOnNoModelImprovement(10, verbose=1)
    
    eval_callback = EvalCallback(env, callback_after_eval=stopTraining, best_model_save_path=f"./models/{file_model_name}/best_model", log_path=f"./logs/{file_model_name}/results", eval_freq=save_periode/2)
    
    callback = CallbackList([checkpoint_callback, HParamCallback(), ProgressBarCallback(), eval_callback])
    
    best_model = f"models/{file_model_name}/best_model/best_model"
    
    n = 1800000
    model_n = f"models/{file_model_name}/{file_model_name}_{n}_steps"
    
    while True:
        try:
            model = PPO.load(best_model, env=env, verbose=1, device=torch.device("cuda:0"), custom_objects={"gamma": gamma}) # gamma(i//(nbRep/10))
            print("Load model")
        except:
            model = PPO(
                'MlpPolicy', 
                env, 
                n_epochs=10, 
                learning_rate=5e-5, 
                ent_coef=0.01, 
                vf_coef=1., 
                gamma=gamma, 
                clip_range= 0.2, 
                verbose=1, 
                tensorboard_log="logs",  
                device="cuda:0" 
                )
            print("Model created")
        

        model.learn(total_timesteps=int(save_periode*nbRep), progress_bar=False, callback=callback)

        
