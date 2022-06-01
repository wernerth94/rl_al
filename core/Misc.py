import os, time
import numpy as np
import json
import torch
from torch import Tensor

def accuracy(yHat, labels):
    yHat = torch.argmax(yHat, dim=1)
    labels = torch.argmax(labels, dim=1)
    correct = yHat == labels
    acc = torch.sum(correct) / len(yHat)
    return acc.cpu().numpy()


def trainTestIDSplit(length, cutoff=0.8):
    ids = np.arange(length)
    split = int(length * cutoff)
    np.random.shuffle(ids)
    return ids[:split], ids[split:]

def avrg(curve, window):
    if len(curve) <= 0:
        return [0]
    if len(curve) < 2:
        return [curve[0]]
    avrgCurve = []

    for i in range(len(curve)):
        avrgCurve.append(np.mean( curve[max(0, i - int(window/2)) : min(len(curve), i + int(window/2))] ))
    return avrgCurve


def saveNumpyFile(name, file):
    if os.path.exists(name + '.npy'):
        os.remove(name + '.npy')
    np.save(name + '.npy', file)


def checkStopSwitch():
    return os.path.exists("config/stopSwitch")

def createStopSwitch():
    if not os.path.exists("config/stopSwitch"):
        open("config/stopSwitch", 'a').close()


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




