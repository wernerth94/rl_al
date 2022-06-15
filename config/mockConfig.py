import os
import util

# general
N_STEPS = 1
MODEL_NAME = 'MOCK'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 32
MEMORY_CAP = 200000
MEMORY_ALPHA = 0.6
AGENT_C = 500
AGENT_GAMMA = 0.0 # TODO
AGENT_NHIDDEN = 10
AGENT_REG = 0.0

# Env config
MAX_REWARD = 0.0005
SAMPLE_SIZE = 20
BUDGET = 2000
REWARD_SCALE = 1
REWARD_SHAPING = True

# training loop
MIN_INTERACTIONS = 500000 # 500k
MAX_EPOCHS = int(MIN_INTERACTIONS / BUDGET)
WARMUP_EPOCHS = 5

CONVERSION_GREED = int(MIN_INTERACTIONS*0.2 / BUDGET)
CONVERSION_LR = int(MIN_INTERACTIONS*0.5 / BUDGET)
GREED = util.parameterPlan(0.9, 0.05, warmup=WARMUP_EPOCHS, conversion=CONVERSION_GREED)
                            # TODO
LR = util.parameterPlan(0.01, 0.01, warmup=WARMUP_EPOCHS, conversion=CONVERSION_LR)


# File Paths
#########################################
# baselines
RECORD_AL_PERFORMANCE = False
BASELINE_FILE = f'baselines/mock/bvssb.npy'
LOWER_BOUND_FILE = f'baselines/mock/random.npy'
if os.path.exists(BASELINE_FILE) and os.path.exists(LOWER_BOUND_FILE):
    RECORD_AL_PERFORMANCE = True

def get_description():
    desc  = f'LOADED CONFIG: {MODEL_NAME}\n'
    desc += f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}, weight_decay={AGENT_REG}\n'
    desc += f'AGENT: batch size={BATCH_SIZE}, C={AGENT_C}\n'
    desc += f'TRAINING: interactions={MIN_INTERACTIONS}, max epochs={MAX_EPOCHS}\n'
    desc += f'TRAINING: warmup={WARMUP_EPOCHS}\n'
    desc += f'TRAINING: learningRate {LR[0]} - {LR[-1]} in {CONVERSION_LR} epochs\n'
    desc += f'TRAINING: greed {GREED[0]} - {GREED[-1]} in {CONVERSION_GREED} epochs\n'
    desc += f'TRAINING: n-steps={N_STEPS}\n'
    desc += f'ENV: budget={BUDGET}, sample size={SAMPLE_SIZE}\n'
    return desc

print('#########################################################')
print(get_description())
print('#########################################################')