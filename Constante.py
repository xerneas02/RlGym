from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, BALL_MAX_SPEED
import numpy as np

FRAME_SKIP   = 8
GAME_SPEED   = 1
NUM_INSTANCE = 1

BATCH_SIZE = 1728
N_STEPS    = int(200_000/NUM_INSTANCE)

MOVE_BALL      = False
BALL_SPEED     = 1500
AFFICHE_SCREEN = False
SIMULATION_PER_STATS = 100

ResX = 1920
ResY = 1080

LAYERS = 1

MODEL_NAME = "ZZeerWillTryHardV4"
REPLAY_FOLDER = "DataState/P1-D1"

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

BLUE_TEAM   = 0
ORANGE_TEAM = 1