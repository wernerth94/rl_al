import torch
import numpy as np
from PoolManagement import resetALPool, sampleNewBatch, addDatapointToPool
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from Misc import accuracy
# from tests.class_sanity_check import run_test

class ALGame:

    stateSpace = 3

    def __init__(self, dataset, modelFunction, config, verbose):
        assert all([isinstance(d, torch.Tensor) for d in dataset])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.y_test_cpu = self.y_test.clone().cpu()
        self.dataset = dataset
        self.nClasses = self.y_test.shape[1]

        self.config = config
        self.budget = config.BUDGET
        self.gameLength = config.GAME_LENGTH
        self.rewardScaling = config.REWARD_SCALE
        self.rewardShaping = config.REWARD_SHAPING
        self.from_scratch = config.CLASS_FROM_SCRATCH
        self.verbose = verbose

        self.modelFunction = modelFunction
        #self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
        self.classifier = modelFunction(inputShape=self.x_test.shape[1:],
                                        numClasses=self.y_test.shape[1])
        self.classifier = self.classifier.to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.initialF1 = 0
        self.currentTestF1 = 0
        self.reset()
        state = self.createState()
        self.stateSpace = state.shape[1]


    def reset(self):
        self.nInteractions = 0

        del self.classifier
        self.classifier = self.modelFunction(inputShape=self.x_test.shape[1:],
                                             numClasses=self.y_test.shape[1])
        self.classifier.to(self.device)
        self.initialWeights = self.classifier.state_dict()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

        self.xLabeled, self.yLabeled, \
        self.xUnlabeled, self.yUnlabeled, self.perClassIntances = resetALPool(self.dataset,
                                                                              self.config.INIT_POINTS_PER_CLASS)
        self.fitClassifier() # sets self.currentTestF1
        self.initialF1 = self.currentTestF1

        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        return self.createState()


    def createState(self):
        alFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds])
        return alFeatures
        # alFeatures = torch.from_numpy(alFeatures).float()
        # poolFeatures = self.getPoolInfo()
        # poolFeatures = poolFeatures.unsqueeze(0).repeat(len(alFeatures), 1)
        # state = torch.cat([alFeatures, poolFeatures], dim=1)
        # return state


    def getClassifierFeatures(self, x):
        eps = 1e-7
        # prediction metrics
        # x = x.to(self.device).float()
        pred = self.classifier(x).detach()
        two_highest, _ = pred.topk(2, dim=1)
        #part = (-bn.partition(-pred.cpu().numpy(), 4, axis=1))[:,:4] # collects the two highest entries
        #struct, indices = torch.sort(two_highest, dim=1)

        # weightedF1 = np.average(pred * self.perClassF1, axis=1)
        f1 = torch.repeat_interleave(torch.Tensor([self.currentTestF1]), len(x))
        entropy = -torch.mean(pred * torch.log(eps + pred) + (1+eps-pred) * torch.log(1+eps-pred), dim=1)
        bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])

        f1 = f1.to(self.device)
        # entropy = entropy.to(self.device)
        # bVsSB = bVsSB.to(self.device)
        state = torch.stack([f1, bVsSB, entropy], dim=-1)
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

        #batch_size = min(batch_size, int(len(self.xLabeled)/5))
        self.xLabeled = self.xLabeled
        self.yLabeled = self.yLabeled
        self.x_test = self.x_test
        self.y_test = self.y_test
        train_dataloader = DataLoader(TensorDataset(self.xLabeled, self.yLabeled), batch_size=batch_size)
        test_dataloader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=batch_size)

        # run_test(train_dataloader, test_dataloader, self.classifier, self.loss, self.optimizer)

        lastLoss = torch.inf
        for e in range(epochs):
            for batch_x, batch_y in train_dataloader:
                yHat = self.classifier(batch_x)
                loss_value = self.loss(yHat, batch_y)
                self.optimizer.zero_grad()
                loss_value.backward()
                self.optimizer.step()
            # early stopping on test
            with torch.no_grad():
                yHat_test = self.classifier(self.x_test)
                test_loss = self.loss(yHat_test, self.y_test)
                if test_loss >= lastLoss:
                    pass
                    # break
                lastLoss = test_loss

        one_hot_y_hat = np.eye(10, dtype='uint8')[torch.argmax(yHat_test, dim=1).cpu()]
        newTestF1 = f1_score(self.y_test_cpu, one_hot_y_hat, average="samples")
        self.currentTestLoss = test_loss

        if self.rewardShaping:
            reward = (newTestF1 - self.currentTestF1) * self.rewardScaling
        else:
            raise NotImplementedError()
        self.currentTestF1 = newTestF1
        return reward


    def checkDone(self):
        done = self.nInteractions >= self.gameLength # max interactions
        if len(self.xLabeled) - (self.config.INIT_POINTS_PER_CLASS * self.yUnlabeled.shape[1]) >= self.budget:
            done = True # budget exhausted
        return done