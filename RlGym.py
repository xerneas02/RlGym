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
from rlgym_tools.extra_rewards.diff_reward import DiffReward

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, ProgressBarCallback, StopTrainingOnNoModelImprovement

from Observer import *
from State import CombinedState, BetterRandom, TrainingStateSetter, DefaultStateClose, RandomState, InvertedState, LineState, DefaultStateCloseOrange, InvertedStateOrange, RandomStateOrange, Attaque, ChaosState
from Reward import *
from Terminal import *
from Action import ZeerLookupAction
from Callback import HParamCallback
from Constante import *
from CustomTerminal import CustomTerminalCondition

import os
import datetime

DiffDistanceBallGoalReward = DiffReward(DistanceBallGoalReward(z_axe=False))

rewards = CombinedReward(
            (
                GoalScoredReward(),
                SaveReward(),
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
                DontGoalPenalityReward(),
                AirPenalityReward(),
                DiffDistanceBallGoalReward,
                BehindTheBallPenalityReward()
            ),
            (
                0         ,  # GoalScoredReward                    #1
                0         ,  # SaveReward                          #2
                0.1       ,  # BoostDifferenceReward               #3
                5         ,  # BallTouchReward                     #4
                0.3       ,  # DemoReward                          #5
                0.01      ,  # DistancePlayerBallReward            #6
                0.0025    ,  # DistanceBallGoalReward              #7
                0.000625  ,  # FacingBallReward                    #8
                0.0025    ,  # AlignBallGoalReward                 #9
                0.00125   ,  # ClosestToBallReward                 #10
                0.00125   ,  # TouchedLastReward                   #11
                0.00125   ,  # BehindBallReward                    #12
                0.005     ,  # VelocityPlayerBallReward            #13
                0.1       ,  # KickoffReward (0.1)                 #14
                0.005     ,  # VelocityReward (0.000625)           #15
                0.05      ,  # BoostAmountReward                   #16
                0.005     ,  # ForwardVelocityReward               #17
                1         ,  # FirstTouchReward                    #18
                0.003     ,  # DontTouchPenalityReward             #19
                0.00      ,  # DontGoalPenalityReward              #20   
                0         ,  # AirPenality                         #21
                0.00      ,  # DiffDistanceBallGoalReward          #22
                0.00      ,  # BehindTheBallPenalityReward
             ),
            verbose=1
        )

def get_match(game_speed=GAME_SPEED):

    match = Match(
        game_speed          = game_speed,
        reward_function     = rewards,
        terminal_conditions = (common_conditions.TimeoutCondition(250), AfterTouchTimeoutCondition(10)),# ,#NoGoalTimeoutCondition(300, 1) #NoTouchFirstTimeoutCondition(50) #common_conditions.GoalScoredCondition(), common_conditions.NoTouchTimeoutCondition(80)
        obs_builder         = ZeerObservations(),
        state_setter        = CombinedState( 
                                rewards,
                                (                   #42 Garde coef par defaut
                                    (DefaultState(),              ()),
                                    (DefaultStateClose(),         ()),
                                    (DefaultStateCloseOrange(),   ()),
                                    (TrainingStateSetter(),       ()),
                                    (RandomState(),               ()),
                                    (RandomStateOrange(),         ()),
                                    (InvertedState(),             ()),
                                    (InvertedStateOrange(),       ()),
                                    (GoaliePracticeState(),       ()), 
                                    (HoopsLikeSetter(),           ()),
                                    (BetterRandom(),              ()),
                                    (KickoffLikeSetter(),         ()),
                                    (WallPracticeState(),         ()),
                                    (Attaque(),                   ()),
                                    (ChaosState(),                ()),
                                ),
                                (
                                    0.00, #DefaultState
                                    0.50, #DefaultStateClose
                                    0.00, #DefaultStateCloseOrange
                                    0.00, #TrainingStateSetter
                                    0.00, #RandomState
                                    0.00, #RandomStateOrange
                                    0.00, #InvertedState
                                    0.00, #InvertedStateOrange
                                    0.00, #GoaliePracticeState
                                    0.00, #HoopsLikeSetter
                                    0.00, #BetterRandom
                                    0.25, #KickoffLikeSetter
                                    0.00, #WallPracticeState
                                    0.00, #Attaque
                                    0.25, #ChaosState
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
        
    file = open("log.txt", "w")
    file.write("")
    file.close(  )
    
    file = open("log_rew.txt", "w")
    file.write("")
    file.close(  )
    
    file = open("log_error.txt", "w")
    file.write("")
    file.close(  )
    
    
    file_model_name = "model_ZZeerV1"
    
    nbRep = 1000
    
    save_periode = 1e5
    
    fps = 120 / FRAME_SKIP
    T = 20
    gamma = lambda x: np.exp(np.log10(0.5)/((T+x)*fps))
    #gamma = np.exp(np.log10(0.5)/(T*fps))
    
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=NUM_INSTANCE, wait_time=40, force_paging=True)
    #env = get_gym(100)
    env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard
    
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_periode/(2),
        save_path=f"./models/{file_model_name}",
        name_prefix=file_model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    
    
    
    best_model = f"models/{file_model_name}/best_model/best_model"
    
    n = 3000000
    model_n = f"models/{file_model_name}/{file_model_name}_{n}_steps"
    
    total_steps = 0
    
    i = 0
    while True:
        stopTraining = StopTrainingOnNoModelImprovement(10, verbose=1)
    
        eval_callback = EvalCallback(env, callback_after_eval=stopTraining, best_model_save_path=f"./models/{file_model_name}/best_model", log_path=f"./logs/{file_model_name}/results", eval_freq=save_periode/(2))
        
        progressBard = ProgressBarCallback()
        
        callback = CallbackList([checkpoint_callback, HParamCallback(), progressBard, eval_callback])
        
        try:
            model = PPO.load(
                best_model, 
                env=env, 
                verbose=1, 
                device=torch.device("cuda:0"), 
                custom_objects={"gamma": gamma(i//(nbRep/10))}
                ) # gamma(i//(nbRep/10))
            print("Load model")
        except:
            model = PPO(
                'MlpPolicy', 
                env, 
                n_epochs=32, 
                batch_size=64,
                learning_rate=5e-5, 
                ent_coef=0.01, 
                vf_coef=1., 
                gamma=gamma(i//(nbRep/10)), 
                clip_range= 0.2, 
                verbose=1, 
                #policy_kwargs={"optimizer_class" : 0},
                tensorboard_log=f"{file_model_name}_{i}/logs",  
                device="cuda:0" 
                )
            print("Model created")
        
        try:
            model.learn(total_timesteps=int(save_periode*nbRep), progress_bar=False, callback=callback)
            total_steps += progressBard.locals["total_timesteps"] - progressBard.model.num_timesteps
            file = open("log.txt", "a")
            file.write(f"{datetime.datetime.now()} Reload simu timesteps : {total_steps}\n")
            file.close()
        except Exception as e:
            file = open("log_error.txt", "a")
            file.write(f"Error {datetime.datetime.now()} :\n{e}\n")
            file.close()
            
        i += 1
        
        