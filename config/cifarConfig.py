import os
import Misc

# general
DATASET = 'cifar_custom'
N_STEPS = 3
MODEL_NAME = 'CIFAR10'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 32
C = 500 # TODO
RL_UPDATES_PER_ENV_UPDATE = 1
MEMORY_CAP = 200000
AGENT_GAMMA = 0.99
AGENT_NHIDDEN = 200

# Env config
SAMPLE_SIZE = 100 #1000
BUDGET = 2000 # TODO
GAME_LENGTH = BUDGET
REWARD_SCALE = 1
REWARD_SHAPING = True
INIT_POINTS_PER_CLASS = 10
CLASS_FROM_SCRATCH = True

# training loop
MIN_INTERACTIONS = 800000 # 150k
MAX_EPOCHS = MIN_INTERACTIONS / BUDGET # 150k
WARMUP_EPOCHS = 5

CONVERSION_EPOCHS = 5
# CONVERSION_EPOCHS = int(MAX_EPOCHS / 4.0)
GREED = Misc.parameterPlan(0.9, 0.05, warmup=WARMUP_EPOCHS, conversion=CONVERSION_EPOCHS)
LR = Misc.parameterPlan(0.001, 0.0001, warmup=WARMUP_EPOCHS, conversion=CONVERSION_EPOCHS)


# File Paths
#########################################
# baselines
RECORD_AL_PERFORMANCE = False
BASELINE_FILE = f'baselines/cifar10_custom/bvssb_{BUDGET}.npy'
LOWER_BOUND_FILE = f'baselines/cifar10_custom/random_{BUDGET}.npy'
if os.path.exists(BASELINE_FILE) and os.path.exists(LOWER_BOUND_FILE):
    RECORD_AL_PERFORMANCE = True

def get_description():
    desc = ""
    desc += f'LOADED CONFIG: {MODEL_NAME} \tDATASET: {DATASET}\n'
    desc += f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}\n'
    desc += f'AGENT: batch size={BATCH_SIZE}, C={C}\n'
    desc += f'AGENT: greed {GREED[0]} - {GREED[-1]} \t learningRate {LR[0]} - {LR[-1]}\n'
    desc += f'TRAINING: interactions={MIN_INTERACTIONS}, max epochs={MAX_EPOCHS}\n'
    desc += f'TRAINING: warmup={WARMUP_EPOCHS}, conversion={CONVERSION_EPOCHS}\n'
    desc += f'ENV: budget={BUDGET}, sample size={SAMPLE_SIZE}\n'
    return desc

print('#########################################################')
print(get_description())
print('#########################################################')