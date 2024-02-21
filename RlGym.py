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
from State import CombinedState, BetterRandom, TrainingStateSetter, DefaultStateClose, RandomState, InvertedState, LineState, DefaultStateCloseOrange, InvertedStateOrange, RandomStateOrange, Attaque, Defense, ChaosState, AirBallAD, DefenseRapide, Mur, Alea, ReplayState
from Reward import *
from Terminal import *
from Action import ZeerLookupAction
from Callback import HParamCallback
from Constante import *
from CustomTerminal import CustomTerminalCondition
from Extracor import CustomFeatureExtractor
from CustomPolicy import CustomActorCriticPolicy

from sb3_contrib import RecurrentPPO
from schedule import linear_schedule,constant_schedule

import os
import datetime
import subprocess

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
                7        ,  # GoalScoredReward                    #1
                5        ,    # SaveReward
                0.0025     ,  # BoostDifferenceReward               #2
                6        ,  # BallTouchReward                     #3
                0.3      ,  # DemoReward                          #4
                0.005   ,  # DistancePlayerBallReward            #5
                0.005   ,  # DistanceBallGoalReward              #6
                0.000625   ,  # FacingBallReward                    #7
                0.002   ,  # AlignBallGoalReward                 #8
                0.00125   ,  # ClosestToBallReward                 #9
                0.00125   ,  # TouchedLastReward                   #10
                0.00300   ,  # BehindBallReward                    #11
                0.00300   ,  # VelocityPlayerBallReward            #12
                0.0025   ,  # KickoffReward (0.1)                 #13
                0.0025   ,  # VelocityReward (0.000625)           #14
                0.00125   ,  # BoostAmountReward                   #15
                0.005   ,  # ForwardVelocityReward               #16
                0        ,  # FirstTouchReward                    #17
                0.003     ,  # DontTouchPenalityReward             #18
                0.002     ,  # DontGoalPenalityReward              #19   
                0        ,  # AirPenality                         #20
                5      ,  # DiffDistanceBallGoalReward          #21
                0.003   ,  # BehindTheBallPenalityReward         #22
             ),
            verbose=1
        )

def get_match(game_speed=GAME_SPEED):

    match = Match(
        game_speed          = game_speed,
        reward_function     = rewards,
        terminal_conditions = (common_conditions.TimeoutCondition(150), 
                               common_conditions.GoalScoredCondition()) ,#NoGoalTimeoutCondition(300, 1) #NoTouchFirstTimeoutCondition(50) #common_conditions.GoalScoredCondition(), common_conditions.NoTouchTimeoutCondition(80)
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
                                    (LineState(100),             ()),
                                    (Attaque(),                   ()),
                                    (Defense(),                   ()),
                                    (AirBallAD(),                 ()),
                                    (DefenseRapide(),             ()),
                                    (Mur(500),                    ()),
                                    (Alea (True, False),          ()),
                                    (ChaosState(),                ()),
                                    (ReplayState(),               ())
                                ),
                                (
                                    0.00, #DefaultState
                                    0.00, #DefaultStateClose
                                    0.00, #DefaultStateCloseOrange
                                    0.00, #TrainingStateSetter
                                    0.00, #RandomState
                                    0.00, #RandomStateOrange
                                    0.00, #InvertedState
                                    0.00, #InvertedStateOrange
                                    0.00, #GoaliePracticeState
                                    0.00, #HoopsLikeSetter
                                    0.00, #BetterRandom
                                    0.00, #KickoffLikeSetter
                                    0.00, #WallPracticeState
                                    1.00, #LineState
                                    0.00, #Attaque
                                    0.00, #Defense
                                    0.00, #AirBallAD
                                    0.00, #DefenseRapide
                                    0.00, #Mur
                                    0.00, #Alea
                                    0.00, #ChaosState
                                    0.00, #ReplayState
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
    

def modifier_resolution(nouveau_res_x, nouveau_res_y):
    result = subprocess.run(["powershell", "[Environment]::GetFolderPath('MyDocuments')"], capture_output=True, text=True)
    documents_folder = result.stdout.strip()

    chemin_fichier = f'{documents_folder}/My Games/Rocket League/TAGame/Config/TASystemSettings.ini'
    nouvelle_ligne_res_x = f"ResX={nouveau_res_x}\n"
    nouvelle_ligne_res_y = f"ResY={nouveau_res_y}\n"
    nouvelle_ligne_borderless = "Borderless=True\n"
    
    lignes_modifiees = []
    res_x_modifie = False
    res_y_modifie = False
    borderless_modifie = False
    with open(chemin_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        for ligne in lignes:
            if ligne.startswith("ResX=") and not res_x_modifie:
                lignes_modifiees.append(nouvelle_ligne_res_x)
                res_x_modifie = True
            elif ligne.startswith("ResY=") and not res_y_modifie:
                lignes_modifiees.append(nouvelle_ligne_res_y)
                res_y_modifie = True
            elif ligne.startswith("Borderless=") and not borderless_modifie:
                lignes_modifiees.append(nouvelle_ligne_borderless)
                borderless_modifie = True
            else:
                lignes_modifiees.append(ligne)
    
    with open(chemin_fichier, 'w') as fichier:
        fichier.writelines(lignes_modifiees)


if __name__ == "__main__":


    if(torch.cuda.is_available()):
        print( torch.cuda.current_device())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
    else:
        print("Not found")
    
    input("Attention les logs précédant vont être suprimé. (press entre to continue)")
    file = open("log.txt", "w")
    file.write("")
    file.close(  )
    
    file = open("stats_bot.txt", "w")
    file.write("")
    file.close(  )
    
    file = open("log_rew.txt", "w")
    file.write("")
    file.close(  )
    
    file = open("log_error.txt", "w")
    file.write("")
    file.close(  )
    
    modifier_resolution(ResX, ResY)
    
    file_model_name = "ZZeerWillTryHard"
    
    nbRep = 100000
    
    save_periode = 1e5
    
    fps = 120 / FRAME_SKIP
    T = 20
    #gamma = lambda x: np.exp(np.log10(0.5)/((T+x)*fps))
    gamma = np.exp(np.log10(0.5)/(T*fps))
    
    env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=NUM_INSTANCE, wait_time=40, force_paging=True)
    env = VecMonitor(env) # Logs mean reward and ep_len to Tensorboard
    
    
    checkpoint_callback = CheckpointCallback(
        save_freq=save_periode/(2),
        save_path=f"./models/{file_model_name}",
        name_prefix=file_model_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    
    
    
    best_model = f"models/{file_model_name}/best_model/best_model"
    
    n = 1500000
    model_n = f"models/{file_model_name}/{file_model_name}_{n}_steps"
    
    total_steps = 0
    
    i = 0
    while True:
        stopTraining = StopTrainingOnNoModelImprovement(10, verbose=1)

        #                                 , callback_after_eval=stopTraining
        eval_callback = EvalCallback(env, best_model_save_path=f"./models/{file_model_name}/best_model", log_path=f"./logs/{file_model_name}/results", eval_freq=save_periode/(2))
        
        progressBard = ProgressBarCallback()
        
        callback = CallbackList([checkpoint_callback, HParamCallback(), progressBard, eval_callback])
        
        try:
             model = RecurrentPPO.load(
                 best_model, 
                 env=env, 
                 verbose=1, 
                 device=torch.device("cuda:0"), 
                 custom_objects={  
                                 "gamma": gamma,
                                 "n_epochs": 10, 
                                 "learning_rate": constant_schedule(5e-5),
                                }
                 )
             print("Load model")
        except:
            model = RecurrentPPO(
                    policy=CustomActorCriticPolicy, 
                    env=env, 
                    n_epochs=10, 
                    n_steps=50000,
                    batch_size=1728,
                    learning_rate=constant_schedule(5e-5), 
                    ent_coef=0.1, 
                    vf_coef=1., 
                    gamma=gamma, 
                    clip_range= 0.2, 
                    verbose=1, 
                    policy_kwargs=dict(
                        features_extractor_class=CustomFeatureExtractor,
                        features_extractor_kwargs=dict(features_dim=256),
                        lstm_hidden_size=256,
                        n_lstm_layers=1,
                        shared_lstm=True,
                        enable_critic_lstm=False
                    ),
                    tensorboard_log=f"{file_model_name}_{i}/logs",  
                    device="cuda:0" 
                    )
            print("Model created")
        
        #try:
        model.learn(total_timesteps=int(save_periode*nbRep), progress_bar=False, callback=callback)
        total_steps += progressBard.locals["total_timesteps"] - progressBard.model.num_timesteps
        file = open("log.txt", "a")
        file.write(f"{datetime.datetime.now()} Reload simu timesteps : {total_steps}\n")
        file.close()
        #except Exception as e:
        #     file = open("log_error.txt", "a")
        #     file.write(f"Error {datetime.datetime.now()} :\n{e}\n")
        #     file.close()
            
        i += 1
        
        