import numpy as np
import random

class UniformScaling:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def apply(self, inputs:list)->list:
        scale = random.random() * (self.high - self.low) + self.low
        return [i * scale for i in inputs]


class GaussianNoise:
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def apply(self, inputs:list)->list:
        noise = np.random.default_rng().normal(self.mean, self.std, inputs[0].shape)
        return [i+noise for i in inputs]