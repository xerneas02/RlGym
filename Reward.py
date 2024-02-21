import numpy as np
import matplotlib.pyplot as plt
import rlgym
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BALL_MAX_SPEED, ORANGE_GOAL_CENTER, BACK_WALL_Y, BLUE_GOAL_CENTER
from numpy.linalg import norm
from abc import ABC, abstractmethod

from scipy.spatial.distance import cosine
from collections import defaultdict


from typing import Any, Optional, Tuple, overload, Union,Dict

from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

from Constante import *

import datetime

TOUCH_VERIF = False
NUMBER_SIMULATION = 0
NUMBER_GOAL = 0
NUMBER_TOUCH = 0
BEHIND_BALL_TIME = 0
NUMBER_TICK = 0

def lire_fichier(nom_fichier):
    data = defaultdict(list)

    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
        for i in range(0, len(lignes), 5):  # Incrément de 5 pour chaque bloc de données
            simulation = int(lignes[i].strip())
            data[simulation].append(int(lignes[i+1].strip()))  # Buts marqués
            data[simulation].append(int(lignes[i+2].strip()))  # Balles touchées
            data[simulation].append(float(lignes[i+3].strip()))  # % Temps du bot

    simulations = []
    buts_marques_avg = []
    balles_touchees_avg = []
    pourcentage_temps_bot_avg = []

    for simulation, values in data.items():
        simulations.append(simulation)
        buts_marques_avg.append(sum(values[::3]) / len(values[::3]))
        balles_touchees_avg.append(sum(values[1::3]) / len(values[1::3]))
        pourcentage_temps_bot_avg.append(sum(values[2::3]) / len(values[2::3]))

    return simulations, buts_marques_avg, balles_touchees_avg, pourcentage_temps_bot_avg

def plot_courbe(nom_fichier):
    simulations, buts_marques, balles_touchees, pourcentage_temps_bot = lire_fichier(nom_fichier)

    plt.figure(figsize=(10, 6))

    plt.plot(simulations, buts_marques, marker='o', color='b', label='Buts marqués')
    plt.plot(simulations, balles_touchees, marker='o', color='r', label='Balles touchées')
    plt.plot(simulations, pourcentage_temps_bot, marker='o', color='g', label='% Temps du bot')
    plt.title('Comparaison des paramètres')
    plt.xlabel('Nombre de simulations')
    plt.ylabel('Valeurs')
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)  # Permet à la fenêtre de répondre

def plot_statistics():
    # Fermer toutes les fenêtres existantes
    plt.close('all')

    # Lecture des données depuis le fichier stats_bot.txt
    with open("stats_bot.txt", "r") as file:
        lines = file.readlines()

    # Extraction des données en groupes de 5 lignes (4 paramètres + "ff")
    data_groups = [list(map(float, lines[i:i+4])) for i in range(0, len(lines), 5)]
    num_simulations = len(data_groups)

    # Dernier ensemble de données
    last_data = data_groups[-1]

    # Calcul de la moyenne des données
    num_groups = len(data_groups)
    avg_data = [sum(x) / num_groups for x in zip(*data_groups)]

    # Création des histogrammes
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Taille en pouces pour une résolution de 1920x1080

    # Histogramme pour les dernières valeurs
    labels = ['NUMBER_GOAL', 'NUMBER_TOUCH', 'BEHIND_BALL_TIME']
    bars = axs[0].bar(labels, last_data[1:], color='skyblue')

    # Affichage des valeurs sur les barres
    for bar, value in zip(bars, last_data[1:]):
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2, height, round(value, 2), ha='center', va='bottom')

    axs[0].set_title('Dernières valeurs des paramètres')
    axs[0].set_ylabel('Valeur')

    # Histogramme pour les moyennes
    bars = axs[1].bar(labels, avg_data[1:], color='lightgreen')

    # Affichage des valeurs sur les barres
    for bar, value in zip(bars, avg_data[1:]):
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2, height, round(value, 2), ha='center', va='bottom')

    axs[1].set_title('Moyenne des paramètres')
    axs[1].set_ylabel('Moyenne')

    # Affichage du nombre de simulations
    plt.figtext(0.5, 0.002, f"Nombre de simulations : {num_simulations}", ha='center', fontsize=12)

    # Affichage des graphiques sans bloquer l'exécution
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)  # Permet à la fenêtre de répondre


def SaveStatFile():
    global NUMBER_TOUCH, NUMBER_GOAL, BEHIND_BALL_TIME, NUMBER_SIMULATION, NUMBER_TICK
    try:
        with open("stats_bot.txt", "a") as file:
            file.write(str(NUMBER_SIMULATION) + "\n")
            file.write(str(NUMBER_GOAL) + "\n")
            file.write(str(NUMBER_TOUCH) + "\n")
            file.write(str((BEHIND_BALL_TIME / NUMBER_TICK ) * 100) + "\n")
            file.write("ff\n")
        if AFFICHE_SCREEN:
            plot_statistics()
            plot_courbe("stats_bot.txt")
    except Exception as e:
        print("An error occurred while writing to file:", e)
    return 0
    
class CombinedReward(RewardFunction):

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None,
            verbose = 0
    ):
        super().__init__()

        self.count = 0
        self.reward_functions       = reward_functions
        self.reward_weights         = reward_weights or np.ones_like(reward_functions)
        self.default_reward_weights = reward_weights or np.ones_like(reward_functions)
        
        self.reward_names = [str(self.reward_functions[i]).split('.')[1].split(' ')[0] for i in range(len(self.reward_functions))]
        self.track_rewards_on_rollout = {f"{name}":[] for name in self.reward_names}
        
        self.verbose = verbose
        self.total_per_rew = np.zeros_like(reward_functions)
        self.period = 5_000
        self.count_period = 0

        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError(
                ("Reward functions list length ({0}) and reward weights " \
                 "length ({1}) must be equal").format(
                    len(self.reward_functions), len(self.reward_weights)
                )
            )
            
    def get_default_reward_weights(self):
        return self.default_reward_weights
    
    def get_rewards_num(self):
        return len(self.reward_functions)
    
    def set_rewards_weights(self, reward_weights):
        self.reward_weights = reward_weights


    def clean_track(self) -> None:
        for name in self.reward_names:
            self.track_rewards_on_rollout[name][:] = []
            
    def get_mean_rewards(self) -> Dict[str,float]:
        mean_dict = {}
        
        for name in self.reward_names:
            mean_dict[name] = np.mean(self.track_rewards_on_rollout[name])
            
            
            
    @classmethod
    def from_zipped(cls, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]) -> "CombinedReward":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.

        :param rewards_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, initial_state: GameState) -> None:
        self.count = 0
        global TOUCH_VERIF, NUMBER_SIMULATION, NUMBER_GOAL, NUMBER_TOUCH, BEHIND_BALL_TIME, NUMBER_TICK
        TOUCH_VERIF = False
        NUMBER_SIMULATION = NUMBER_SIMULATION +1
        #print(NUMBER_SIMULATION)
        if NUMBER_SIMULATION % SIMULATION_PER_STATS == 0:
            SaveStatFile()
            NUMBER_GOAL = 0
            NUMBER_TOUCH = 0
            BEHIND_BALL_TIME = 0
            NUMBER_TICK = 0
        
        for func in self.reward_functions:
            func.reset(initial_state)

    def get_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
    
        
        total = float(np.dot(self.reward_weights, rewards))
        self.count += 1
        self.count_period += 1
        
        for i in range(len(rewards)):
            self.total_per_rew[i] += rewards[i]

        if GAME_SPEED == 1 and self.verbose == 1:
            for i in range(len(rewards)):
                pondered_reward = rewards[i]*self.reward_weights[i]
                self.track_rewards_on_rollout[self.reward_names[i]].append(pondered_reward)
                
                if rewards[i] != 0 and player.team_num == 0:
                    print(f"reward {self.reward_names[i]}: {pondered_reward}")
            
            
            if total != 0 and player.team_num == 0:
                print(f"Total : {total}\n-----------------------")

        return total

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        if GAME_SPEED == 1 and player.team_num == 0 and self.verbose:
            print(f"---  Time = {self.count}  ---")
        
        self.count_period += 1
        
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        
        for i in range(len(rewards)):
            self.total_per_rew[i] += rewards[i]

        if self.count_period >= self.period:
            file = open("log_rew.txt", "a")
            txt = f"{datetime.datetime.now()} :\n"
            for i in range(len(self.total_per_rew)):
                txt += f"reward {str(self.reward_functions[i]).split('.')[1].split(' ')[0]}: {(self.total_per_rew[i]*self.reward_weights[i])}\n"
            
            txt += "-------------------------------------------------------------------\n\n"
            
            file.write(txt)
            file.close()

            self.count_period = 0
            self.total_per_rew = np.zeros_like(self.reward_functions)

        return float(np.dot(self.reward_weights, rewards))


#Si le bot marque un but
class GoalScoredReward(RewardFunction):
    def __init__(self):
        self.previous_blue_score   = 0
        self.previous_orange_score = 0
        
    def reset(self, initial_state):
        self.previous_blue_score   = initial_state.blue_score
        self.previous_orange_score = initial_state.orange_score

    def get_reward(self, player, state, previous_action):
        blue_scored   = False
        orange_scored = False
        global TOUCH_VERIF, NUMBER_GOAL
        
        if player.team_num == 0 and self.previous_blue_score != state.blue_score:
            blue_scored = True
            self.previous_blue_score = state.blue_score
        
        if player.team_num == 1 and self.previous_orange_score != state.orange_score:
            orange_scored = True
            self.previous_orange_score = state.orange_score
            
        if(player.team_num == 0 and not blue_scored  ) : return 0 
        if(player.team_num == 1 and not orange_scored) : return 0
        
        ball_speed = np.linalg.norm(state.ball.linear_velocity, 2)**2
        if (TOUCH_VERIF):
            NUMBER_GOAL = NUMBER_GOAL + 1
            return 1.0 + 0.5 * ball_speed / (BALL_MAX_SPEED)
        else:
            return 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot collect ou utilise du boost
class BoostDifferenceReward(RewardFunction):
    def __init__(self):
        self.previous_boost = None

    def reset(self, initial_state):
        self.previous_boost = initial_state.players[0].boost_amount

    def get_reward(self, player, state, previous_action):
        current_boost = player.boost_amount
        reward = np.abs(np.sqrt(current_boost/100) - np.sqrt(self.previous_boost/100))
        self.previous_boost = current_boost
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
#Si le bot touche la balle (reward varie en fonction de la hauteur de la balle)
class BallTouchReward(RewardFunction):
    def __init__(self):
        self.last_touch = False
        self.lamb = 0

    def reset(self, initial_state):
        self.last_touch = False
        self.lamb = 0

    def get_reward(self, player, state, previous_action):
        global TOUCH_VERIF, NUMBER_TOUCH
        
        if self.last_touch:
            self.lamb = max(0.1, self.lamb * 0.95)
        else:
            self.lamb = min(1.0, self.lamb + 0.013)
        
        if not player.ball_touched : 
            self.last_touch = False
            return 0
            
        self.last_touch = True
            
        pos_ball_z = state.ball.position[2]
        reward = self.lamb * ((pos_ball_z + BALL_RADIUS)/(2*BALL_RADIUS)) ** 0.2836
        TOUCH_VERIF = True
        NUMBER_TOUCH = NUMBER_TOUCH + 1
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
#Si le bot démo
class DemoReward(RewardFunction):
    def __init__(self):
        self.last_state = None
        
    def reset(self, initial_state):
        self.last_state = initial_state

    def get_reward(self, player, state, previous_action):
        
        for p in self.last_state.players:
            if p.car_id == player.car_id and player.match_demolishes > p.match_demolishes:
                return 1
        
        self.last_state = state
        return 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est proche de la balle
class DistancePlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position = player.car_data.position
        ball_position = state.ball.position
        
        distance = np.linalg.norm(car_position - ball_position, 2) - BALL_RADIUS
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si la balle est proche du but adverse
class DistanceBallGoalReward(RewardFunction):
    def __init__(self, x_axe=True, y_axe=True, z_axe=True):
        self.x_axe = x_axe
        self.y_axe = y_axe
        self.z_axe = z_axe 
    
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        net_position = ORANGE_GOAL_CENTER

        c = net_position[1] - BACK_WALL_Y + BALL_RADIUS
        
        distance = 0
        
        if self.x_axe:
            distance += np.abs(ball_position[0] - net_position[0])
        if self.y_axe:
            distance += np.abs(ball_position[1] - net_position[1])
        if self.z_axe:
            distance += np.abs(ball_position[2] - net_position[2])
        
        distance -= c
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
#Si le bot fait face à la balle
class FacingBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_orientation = player.car_data.forward()
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball, 2)
        
        reward = np.dot(car_orientation, direction_to_ball)
        
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est entre ses buts et la balle [mais il y a une ligne qui relie le bot, la balle et le but]
class AlignBallGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position  = player.car_data.position
        ball_position = state.ball.position
        
        opponent_goal_position = ORANGE_GOAL_CENTER
        self_goal_position     = BLUE_GOAL_CENTER
        
        direction_car_to_ball         = ball_position          - car_position
        direction_ball_to_car         = car_position           - ball_position
        direction_self_goal_to_car    = car_position           - self_goal_position
        direction_car_to_oponent_goal = opponent_goal_position - car_position
        
        direction_car_to_ball        /= np.linalg.norm(direction_car_to_ball)
        direction_ball_to_car        /= np.linalg.norm(direction_ball_to_car)
        direction_self_goal_to_car   /= np.linalg.norm(direction_self_goal_to_car)
        direction_car_to_oponent_goal/= np.linalg.norm(direction_car_to_oponent_goal)
        
        
        return -0.5 * np.dot(direction_ball_to_car, direction_car_to_oponent_goal) + 0.5 * np.dot(direction_car_to_ball, direction_self_goal_to_car) 

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si plus proche de la balle par rapport aux adversaires
class ClosestToBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        car_position = player.car_data.position
        distance = np.linalg.norm(ball_position - car_position, 2)
        
        for p in state.players:
            if distance > np.linalg.norm(ball_position - p.car_data.position, 2):
                return 0
        
        return 1

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est le dernier à avoir touché la balle
class TouchedLastReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return 1 if state.last_touch == player.car_id else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est entre la balle et son but
class BehindBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position  = player.car_data.position
        ball_position = state.ball.position
        net_position  = ORANGE_GOAL_CENTER
        
        direction_to_ball = ball_position - car_position
        direction_to_net  = net_position - car_position
        
        return 1 if np.dot(direction_to_ball, direction_to_net) > 0 else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot va dans la même direction de la balle
class VelocityPlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity
        
        if np.linalg.norm(car_velocity, 2) < 0.01 :
            return 0
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        
        car_velocity_direction = np.dot(car_velocity, direction_to_ball)
        
        return car_velocity_direction / np.linalg.norm(car_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot gagne le kickoff
class KickoffReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        
        if ball_position[0] != 0 or ball_position[1] != 0:
            return 0
        
        car_velocity = player.car_data.linear_velocity
        
        if np.linalg.norm(car_velocity, 2) < 0.01 :
            return 0
        
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        
        car_velocity_direction = np.dot(car_velocity, direction_to_ball)
        
        return car_velocity_direction / np.linalg.norm(car_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot bouge
class VelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity

        return np.linalg.norm(car_velocity, 2) / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot à du boost
class BoostAmountReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return np.sqrt(player.boost_amount / 100)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot bouge dans la direction de la balle (dans la bonne direction), penalise la marche arrière
class ForwardVelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity
        car_orientation = player.car_data.forward()
        
        car_forward_velocity = np.dot(car_velocity, car_orientation)

        return car_forward_velocity / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
    
class FirstTouchReward(RewardFunction):
    def __init__(self):
        self.kickoff = True
        
    def reset(self, initial_state):
        self.kickoff = True

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        if ball_position[0] != 0 or ball_position[1] != 0 and self.kickoff:
            self.kickoff = False
            return player.car_id == state.last_touch
        
        return 0

        
        
    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
    
class AirPenalityReward(RewardFunction):
    def reset(self, initial_state):
        pass
    
    def get_reward(self, player, state, previous_action):
        return (not player.on_ground)*-1
    
    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
class DontTouchPenalityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.has_touched_ball = False

    def reset(self, initial_state: GameState):
        self.ticks = 0
        self.has_touched_ball = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:    
        self.ticks += 1
        
        if(player.ball_touched):
            self.has_touched_ball = True
    
        return - (self.ticks * (not self.has_touched_ball) * 0.01)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)
    
class VelocityBallOwnGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_velocity = state.ball.linear_velocity
        
        if np.linalg.norm(ball_velocity, 2) < 0.01 :
            return 0
        
        ball_position = state.ball.position
        
        
        direction_to_goal = ORANGE_GOAL_CENTER - ball_position
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        
        ball_velocity_direction = np.dot(ball_velocity, direction_to_goal)
        
        return ball_velocity_direction / np.linalg.norm(ball_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
class VelocityBallOpponentGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_velocity = state.ball.linear_velocity
        
        if np.linalg.norm(ball_velocity, 2) < 0.01 :
            return 0
        
        ball_position = state.ball.position
        
        
        direction_to_goal = BLUE_GOAL_CENTER - ball_position
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        
        ball_velocity_direction = np.dot(ball_velocity, direction_to_goal)
        
        return ball_velocity_direction / np.linalg.norm(ball_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
class SaveReward(RewardFunction):
    def __init__(self):
        self.past_ball_velocity = 0
    
    def reset(self, initial_state):
        self.past_ball_velocity = 0

    def get_reward(self, player, state, previous_action):
        ball_velocity = state.ball.linear_velocity
        
        if np.linalg.norm(ball_velocity, 2) < 0.01 :
            return 0
        
        ball_position = state.ball.position
        
        
        direction_to_goal = BLUE_GOAL_CENTER - ball_position
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        
        ball_velocity_direction = np.dot(ball_velocity, direction_to_goal)
        ball_velocity_direction/= np.linalg.norm(ball_velocity, 2)
        
        
        result = ball_velocity_direction < 0 and self.past_ball_velocity > 0 and player.ball_touched
        
        self.past_ball_velocity = ball_velocity_direction
        
        return result * -(ball_velocity_direction)

class DontGoalPenalityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.has_goaled = False
        self.previous_blue_score   = 0
        self.previous_orange_score = 0
        
    def reset(self, initial_state: GameState):
        self.ticks = 0
        self.has_goaled = False
        self.previous_blue_score   = initial_state.blue_score
        self.previous_orange_score = initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float: 
        if self.previous_blue_score != state.blue_score or self.previous_orange_score != state.orange_score:
            self.has_goaled = True
        
        self.ticks += 1
    
        return - (self.ticks * (not self.has_goaled) * 0.01)
    
class BehindTheBallPenalityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.ticks = 0
        self.is_behind = False

    def reset(self, initial_state: GameState):
        self.ticks = 0
        self.has_goaled = False


    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float: 
        self.is_behind = player.car_data.position[1] < state.ball.position[1]
        global BEHIND_BALL_TIME, NUMBER_TICK
        NUMBER_TICK = NUMBER_TICK + 1
        
        self.ticks += 1
        if (self.is_behind):
            BEHIND_BALL_TIME = BEHIND_BALL_TIME + 1
    
        return - (self.ticks * (not self.is_behind) * 0.01)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)
