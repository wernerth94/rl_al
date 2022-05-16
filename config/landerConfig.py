import Misc

# general
N_STEPS = 10
MODEL_NAME = 'lunar_lander'
USE_STOPSWITCH = True
PRINT_FREQ = 1

# RL training
BATCH_SIZE = 32
MEMORY_CAP = 200000
AGENT_C = 500
AGENT_GAMMA = 0.99
AGENT_NHIDDEN = 200

# training loop
MAX_EPOCHS = 10000
WARMUP_EPOCHS = 20

# 50 epochs with 2k budget = 100000
CONVERSION_GREED = 1000
# 400 epochs with 2k budget
CONVERSION_LR = 1000
GREED = Misc.parameterPlan(0.9, 0.05, warmup=WARMUP_EPOCHS, conversion=CONVERSION_GREED)
LR = Misc.parameterPlan(0.004, 0.0001, warmup=WARMUP_EPOCHS, conversion=CONVERSION_LR)


# File Paths
#########################################
# baselines
RECORD_AL_PERFORMANCE = False
BASELINE_FILE = ""
LOWER_BOUND_FILE = ""

def get_description():
    desc  = f'LOADED CONFIG: {MODEL_NAME}\n'
    desc += f'AGENT: gamma={AGENT_GAMMA}, nHidden={AGENT_NHIDDEN}\n'
    desc += f'AGENT: batch size={BATCH_SIZE}, C={AGENT_C}\n'
    desc += f'AGENT: learningRate {LR[0]} - {LR[-1]} in {CONVERSION_LR} epochs\n'
    desc += f'AGENT: greed {GREED[0]} - {GREED[-1]} in {CONVERSION_GREED} epochs\n'
    desc += f'TRAINING: max epochs={MAX_EPOCHS}\n'
    desc += f'TRAINING: warmup={WARMUP_EPOCHS}\n'
    desc += f'TRAINING: n-steps={N_STEPS}\n'
    return desc

print('#########################################################')
print(get_description())
print('#########################################################')