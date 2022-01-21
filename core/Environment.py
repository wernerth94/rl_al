import torch

import tensorflow.keras as keras
import gc
import numpy as np
import bottleneck as bn
from PoolManagement import resetALPool, sampleNewBatch, addDatapointToPool
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn

class ALGame:

    stateSpace = 3

    def __init__(self, dataset, modelFunction, config, verbose):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = (torch.from_numpy(dataset[0]),
                   torch.from_numpy(dataset[1]),
                   torch.from_numpy(dataset[2]),
                   torch.from_numpy(dataset[3]))
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.dataset = dataset
        self.nClasses = self.y_test.shape[1]

        self.config = config
        self.budget = config.BUDGET
        self.gameLength = config.GAME_LENGTH
        self.labelCost = config.LABEL_COST
        self.rewardScaling = config.REWARD_SCALE
        self.rewardShaping = config.REWARD_SHAPING
        self.from_scratch = config.CLASS_FROM_SCRATCH
        self.verbose = verbose

        self.modelFunction = modelFunction
        #self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
        self.classifier = modelFunction(inputShape=self.x_test.shape[1:],
                                        numClasses=self.y_test.shape[1])
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.initialF1 = 0
        self.currentTestF1 = 0
        self.reset()
        state = self.createState()
        self.stateSpace = state.shape[1]


    def reset(self):
        # self.initialF1 = 0
        # self.currentTestF1 = 0
        self.nInteractions = 0

        self.classifier = self.modelFunction(inputShape=self.x_test.shape[1:],
                                             numClasses=self.y_test.shape[1])
        self.initialWeights = self.classifier.state_dict()

        self.xLabeled, self.yLabeled, \
        self.xUnlabeled, self.yUnlabeled, self.perClassIntances = resetALPool(self.dataset,
                                                                              self.config.INIT_POINTS_PER_CLASS)
        self.fitClassifier()
        self.initialF1 = self.currentTestF1

        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        return self.createState()


    def createState(self):
        alFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds])
        alFeatures = torch.from_numpy(alFeatures).float()
        poolFeatures = self.getPoolInfo()
        poolFeatures = poolFeatures.unsqueeze(0).repeat(len(alFeatures), 1)
        state = torch.cat([alFeatures, poolFeatures], dim=1)
        return state


    def getClassifierFeatures(self, x):
        eps = 1e-7
        # prediction metrics
        x = x.to(self.device)
        pred = self.classifier(x).detach()
        part = (-bn.partition(-pred, 4, axis=1))[:,:4] # collects the two highest entries
        struct = np.sort(part, axis=1)

        # weightedF1 = np.average(pred * self.perClassF1, axis=1)
        f1 = np.repeat(np.mean(self.currentTestF1), len(x))
        entropy = -np.average(pred * np.log(eps + pred) + (1+eps-pred) * np.log(1+eps-pred), axis=1)
        bVsSB = 1 - (struct[:, -1] - struct[:, -2])

        # state = np.expand_dims(bVsSB, axis=-1)
        state = np.stack([f1, bVsSB, entropy], axis=-1)
        # state = np.concatenate([state, struct], axis=1)
        return state


    def getPoolInfo(self):
        labeled = torch.mean(self.xLabeled, dim=0)
        unlabeled = torch.mean(self.xUnlabeled, dim=0)
        return torch.cat([labeled, unlabeled])


    def step(self, action):
        self.nInteractions += 1
        datapointId = self.stateIds[action]
        self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = addDatapointToPool(self.xLabeled, self.yLabeled,
                                                                                                              self.xUnlabeled, self.yUnlabeled,
                                                                                                              self.perClassIntances, datapointId)
        reward = self.fitClassifier()
        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        statePrime = self.createState()

        done = self.checkDone()
        return statePrime, reward, done, {}


    def fitClassifier(self, epochs=50, batch_size=128):
        if self.from_scratch:
            self.classifier.load_state_dict(self.initialWeights)

        self.xLabeled = self.xLabeled.to(self.device)
        self.yLabeled = self.yLabeled.to(self.device)
        self.x_test = self.x_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        lastLoss = torch.inf
        for e in range(epochs):
            permutation = torch.randperm(len(self.xLabeled))
            for i in range(0, len(self.xLabeled), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = self.xLabeled[indices], self.yLabeled[indices]
                yHat = self.classifier(batch_x)
                loss = self.loss(yHat, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # early stopping on test
            with torch.no_grad():
                yHat_test = self.classifier(self.x_test)
                test_loss = self.loss(yHat_test, self.y_test)
                if test_loss > lastLoss:
                    break
                lastLoss = test_loss

        newTestF1 = f1_score(self.y_test, np.eye(10, dtype='uint8')[torch.argmax(yHat_test, dim=1)], average="samples")
        self.currentTestLoss = test_loss

        if self.rewardShaping:
            reward = (newTestF1 - self.currentTestF1 - self.labelCost) * self.rewardScaling
        else:
            raise NotImplementedError()
        self.currentTestF1 = newTestF1
        return reward


    def checkDone(self):
        done = self.nInteractions >= self.gameLength # max interactions
        if len(self.xLabeled) - (self.config.INIT_POINTS_PER_CLASS * self.yUnlabeled.shape[1]) >= self.budget:
            done = True # budget exhausted
        return done