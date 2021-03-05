import os
import Misc

# general
DATASET = 'traffic_signs'
EMBEDDING_SIZE = 256
N_STEPS = 3
MODEL_NAME = '_TRAFFIC'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 16
C = 200
RL_UPDATES_PER_ENV_UPDATE = 1
MEMORY_CAP = 100000 # 100k

# Env config
SAMPLE_SIZE = 500
BUDGET = 300
GAME_LENGTH = int(1e10)
MAX_INTERACTIONS_PER_GAME = BUDGET * 5
REWARD_SCALE = 1
REWARD_SHAPING = True
LABEL_COST = 0 # 0.001
INIT_POINTS_PER_CLASS = 5

# training loop
MIN_INTERACTIONS = 600000 # 150k
WARMUP = 1000
N_EXPLORE = WARMUP + int(0.02*MIN_INTERACTIONS)
N_CONVERSION = int(0.7*MIN_INTERACTIONS)
EVAL_ITERATIONS = 10
# GREED = Misc.parameterPlan(0.9, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
# LR = Misc.parameterPlan(0.01, 0.0001, warmup=N_EXPLORE, conversion=N_CONVERSION)

# HANDCRAFTED
##################################################################
N_EXPLORE = 220000
N_CONVERSION = 70000
GREED = Misc.parameterPlan(0.1, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
LR = Misc.parameterPlan(0.001, 0.001, warmup=N_EXPLORE, conversion=N_CONVERSION)
##################################################################

# Game Length
GL = Misc.asympParameterPlan(BUDGET, BUDGET, warmup=WARMUP + int(0.1*MIN_INTERACTIONS), conversion=int(0.6*MIN_INTERACTIONS))

# file paths
OUTPUT_FOLDER = 'out'+MODEL_NAME
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
stateValueDir = os.path.join(cacheDir, 'stateVal')
stateTransDir = os.path.join(cacheDir, 'stateTrans')