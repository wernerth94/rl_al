import os
import Misc

# general
MODEL_NAME = 'DDQN_MNIST_CONV'
#MODEL_NAME = 'DDQN_IRIS_CONV'
USE_STOPSWITCH = True
PRINT_FREQ = 500

# RL training
BATCH_SIZE = 16
C = 500
RL_UPDATES_PER_ENV_UPDATE = 1
MEMORY_CAP = 10000 # 10k

# Env config
SAMPLE_SIZE = 1000
DATAPOINTS_TO_AVRG = 1
BUDGET = 300 # MNIST
#BUDGET = 40 #IRIS
GAME_LENGTH = BUDGET
MAX_INTERACTIONS_PER_GAME = BUDGET * 5 / DATAPOINTS_TO_AVRG
REWARD_SCALE = 1#40
REWARD_SHAPING = False
LABEL_COST = 0.001
INIT_POINTS_PER_CLASS = 5

# training loop
MIN_INTERACTIONS = 1000 # 150k
WARMUP = 0 #1000
N_EXPLORE, N_CONVERSION = WARMUP + int(0.2*MIN_INTERACTIONS), int(0.4*MIN_INTERACTIONS)
EVAL_ITERATIONS = 10

GREED = Misc.parameterPlan(0.9, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
LR = Misc.parameterPlan(0.02, 0.001, warmup=N_EXPLORE, conversion=N_CONVERSION)

# Game Length
GL = Misc.asympParameterPlan(100, BUDGET, warmup=WARMUP + int(0.1*MIN_INTERACTIONS), conversion=int(0.6*MIN_INTERACTIONS))

# file paths
OUTPUT_FOLDER = 'out'+MODEL_NAME
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
ckptDir = os.path.join(cacheDir, 'ckpt')