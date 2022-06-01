import os
import util

# general
DATASET = 'mnist_mobileNet'
EMBEDDING_SIZE = 1280
N_STEPS = 3
MODEL_NAME = 'MNIST_SCRATCH'
USE_STOPSWITCH = True
PRINT_FREQ = 1
EVAL_ITERATIONS = 10

# RL training
BATCH_SIZE = 16
C = 2000
RL_UPDATES_PER_ENV_UPDATE = 1
MEMORY_CAP = 200000 # 100k
AGENT_GAMMA = 1.0
AGENT_NHIDDEN = 20

# Env config
SAMPLE_SIZE = 1000
BUDGET = 1000 # MNIST
GAME_LENGTH = BUDGET
REWARD_SCALE = 1
REWARD_SHAPING = True
LABEL_COST = 0 # 0.001
INIT_POINTS_PER_CLASS = 5
CLASS_FROM_SCRATCH = True

# training loop
MIN_INTERACTIONS = 800000 # 150k
WARMUP = 2000

# N_EXPLORE = WARMUP + int(0.02*MIN_INTERACTIONS)
# N_CONVERSION = int(0.7*MIN_INTERACTIONS)
# GREED = Misc.parameterPlan(0.9, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
# LR = Misc.parameterPlan(0.01, 0.0001, warmup=N_EXPLORE, conversion=N_CONVERSION)

# HANDCRAFTED
##################################################################
N_EXPLORE = 220000
N_CONVERSION = 70000
GREED = util.parameterPlan(0.1, 0.1, warmup=N_EXPLORE, conversion=N_CONVERSION)
LR = util.parameterPlan(0.001, 0.001, warmup=N_EXPLORE, conversion=N_CONVERSION)
##################################################################

# Game Length
GL = util.asympParameterPlan(BUDGET, BUDGET, warmup=WARMUP + int(0.1*MIN_INTERACTIONS), conversion=int(0.6*MIN_INTERACTIONS))

# file paths
OUTPUT_FOLDER = 'out_'+MODEL_NAME
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
memDir = os.path.join(OUTPUT_FOLDER, 'memory')
cacheDir = os.path.join(OUTPUT_FOLDER, 'cache')
stateValueDir = os.path.join(cacheDir, 'stateVal')
stateTransDir = os.path.join(cacheDir, 'stateTrans')

print('#########################################################')
print(f'LOADED CONFIG: MNIST \t Model name: {MODEL_NAME}')
print(f'DATASET: {DATASET} \t embedding size {EMBEDDING_SIZE}')
print(f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}')
print(f'TRAINING: interactions={MIN_INTERACTIONS}, C={C}, memSize={MEMORY_CAP}, batchSize={BATCH_SIZE}')
print(f'Greed {GREED[0]} - {GREED[-1]} \t LearningRate {LR[0]} - {LR[-1]}')
print('#########################################################')