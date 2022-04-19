import os
import Misc

# general
DATASET = 'cifar_custom'
N_STEPS = 10
MODEL_NAME = 'CIFAR10'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 32
MEMORY_CAP = 200000
AGENT_C = 500
AGENT_GAMMA = 0.99
AGENT_NHIDDEN = 200

# Env config
SAMPLE_SIZE = 20
BUDGET = 2000
MAX_INTERACTIONS = BUDGET * 10
INIT_POINTS_PER_CLASS = 10
CLASS_FROM_SCRATCH = True
REWARD_SCALE = 1
REWARD_SHAPING = True

# training loop
MIN_INTERACTIONS = 2400000 # 2.4M
MAX_EPOCHS = MIN_INTERACTIONS / BUDGET
WARMUP_EPOCHS = 5

CONVERSION_GREED = 50
CONVERSION_LR = 400
GREED = Misc.parameterPlan(0.9, 0.05, warmup=WARMUP_EPOCHS, conversion=CONVERSION_GREED)
LR = Misc.parameterPlan(0.004, 0.0001, warmup=WARMUP_EPOCHS, conversion=CONVERSION_LR)


# File Paths
#########################################
# baselines
RECORD_AL_PERFORMANCE = False
BASELINE_FILE = f'baselines/cifar10_custom/bvssb.npy'
LOWER_BOUND_FILE = f'baselines/cifar10_custom/random.npy'
if os.path.exists(BASELINE_FILE) and os.path.exists(LOWER_BOUND_FILE):
    RECORD_AL_PERFORMANCE = True

def get_description():
    desc  = f'LOADED CONFIG: {MODEL_NAME} \tDATASET: {DATASET}\n'
    desc += f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}\n'
    desc += f'AGENT: batch size={BATCH_SIZE}, C={AGENT_C}\n'
    desc += f'AGENT: learningRate {LR[0]} - {LR[-1]} in {CONVERSION_LR} epochs\n'
    desc += f'AGENT: greed {GREED[0]} - {GREED[-1]} in {CONVERSION_GREED} epochs\n'
    desc += f'TRAINING: interactions={MIN_INTERACTIONS}, max epochs={MAX_EPOCHS}\n'
    desc += f'TRAINING: warmup={WARMUP_EPOCHS}\n'
    desc += f'TRAINING: n-steps={N_STEPS}\n'
    desc += f'ENV: budget={BUDGET}, sample size={SAMPLE_SIZE}\n'
    return desc

print('#########################################################')
print(get_description())
print('#########################################################')