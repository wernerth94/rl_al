import os
import numpy as np
import json

def saveFile(name, file):
    if os.path.exists(name + '.npy'):
        os.remove(name + '.npy')
    np.save(name + '.npy', file)

def parameterPlan(val1, val2, warmup, conversion):
    plan1 = np.full(warmup, val1)
    plan2 = np.linspace(val1, val2, conversion)
    return np.concatenate([plan1, plan2])


def createStopSwitch():
    if not os.path.exists("stopSwitch"):
        open("stopSwitch", 'a').close()


def saveTrainState(config, state:dict):
    file = os.path.join(config.OUTPUT_FOLDER, config.MODEL_NAME+'.json')
    if os.path.exists(file):
        os.remove(file)
    json.dump(state, open(file, 'w'))


def loadTrainState(config, path_prefix=None):
    file = os.path.join(config.OUTPUT_FOLDER, config.MODEL_NAME+'.json')
    if path_prefix is not None:
        file = os.path.join(path_prefix, file)
    if os.path.exists(file):
        return json.load(open(file))
    return None