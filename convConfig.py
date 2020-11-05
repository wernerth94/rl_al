import os
import Misc, Data

# general
MODEL_NAME = 'DDQN_IRIS_CONV'
USE_STOPSWITCH = True
PRINT_FREQ = 500

# RL training
BATCH_SIZE = 16
C = 2000
RL_UPDATES_PER_ENV_UPDATE = 1
MEMORY_CAP = 50000 # 50k

# Env config
SAMPLE_SIZE = 1000
DATAPOINTS_TO_AVRG = 1
BUDGET = 500 # MNIST
#BUDGET = 40 #IRIS
GAME_LENGTH = BUDGET
MAX_INTERACTIONS_PER_GAME = BUDGET * 5 / DATAPOINTS_TO_AVRG
REWARD_SCALE = 1#40
REWARD_SHAPING = False
LABEL_COST = 0.01
INIT_POINTS_PER_CLASS = 5

# training loop
MIN_INTERACTIONS = 500000 # 300k
WARMUP = 500 # int(0.03*MIN_INTERACTIONS)
N_EXPLORE, N_CONVERSION = WARMUP + int(0.2*MIN_INTERACTIONS), int(0.4*MIN_INTERACTIONS)
EVAL_ITERATIONS = 20

GREED = Misc.parameterPlan(1, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
LR = Misc.parameterPlan(0.002, 0.001, warmup=N_EXPLORE, conversion=N_CONVERSION)

# Game Length
GL = Misc.asympParameterPlan(100, BUDGET, warmup=WARMUP + int(0.1*MIN_INTERACTIONS), conversion=int(0.6*MIN_INTERACTIONS))

# file paths
OUTPUT_FOLDER = 'out'+MODEL_NAME
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
ckptDir = os.path.join(cacheDir, 'ckpt')