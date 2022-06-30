import os
import util

# general
DATASET = 'cifar_custom'
N_STEPS = 1
MODEL_NAME = 'CIFAR10'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 32
MEMORY_CAP = int(7e+4) # 200000 lower memory cap works in Q experiment, maybe its better?
AGENT_C = 500
AGENT_GAMMA = 0.0 # TODO
AGENT_NHIDDEN = 200
AGENT_REG = 0.00001

# Env config
SAMPLE_SIZE = 20
BUDGET = 2000
MAX_INTERACTIONS = BUDGET * 10
INIT_POINTS_PER_CLASS = 1
CLASS_FROM_SCRATCH = True
REWARD_SCALE = 1
REWARD_SHAPING = True

# training loop
MIN_INTERACTIONS = 1000000 # 1.0M
MAX_EPOCHS = int(MIN_INTERACTIONS / BUDGET)
WARMUP_EPOCHS = 5

CONVERSION_GREED = int(MIN_INTERACTIONS*0.2 / BUDGET)
CONVERSION_LR = int(MIN_INTERACTIONS*0.5 / BUDGET)
GREED = util.parameterPlan(0.9, 0.05, warmup=WARMUP_EPOCHS, conversion=CONVERSION_GREED)
LR = util.parameterPlan(0.01, 0.004, warmup=WARMUP_EPOCHS, conversion=CONVERSION_LR)


# File Paths
#########################################
# baselines
RECORD_AL_PERFORMANCE = True
BASELINE_FILE = f'baselines/cifar10_custom/bvssb.npy'
LOWER_BOUND_FILE = f'baselines/cifar10_custom/random.npy'
if not os.path.exists(BASELINE_FILE) and not os.path.exists(LOWER_BOUND_FILE):
    RECORD_AL_PERFORMANCE = False
    print("Baseline files not found, disabling AL performance")

def get_description():
    desc  = f'LOADED CONFIG: {MODEL_NAME} \tDATASET: {DATASET}\n'
    desc += f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}\n'
    desc += f'AGENT: batch size={BATCH_SIZE}, C={AGENT_C}\n'
    desc += f'AGENT: learningRate {LR[0]} - {LR[-1]} in {CONVERSION_LR} epochs\n'
    desc += f'AGENT: greed {GREED[0]} - {GREED[-1]} in {CONVERSION_GREED} epochs\n'
    desc += f'TRAINING: interactions={MIN_INTERACTIONS}, max epochs={MAX_EPOCHS}\n'
    desc += f'TRAINING: warmup={WARMUP_EPOCHS}\n'
    desc += f'TRAINING: n-steps={N_STEPS}\n'
    desc += f'ENV: budget={BUDGET}, sample size={SAMPLE_SIZE}, Classifier from scratch={CLASS_FROM_SCRATCH}\n'
    return desc

print('#########################################################')
print(get_description())
print('#########################################################')