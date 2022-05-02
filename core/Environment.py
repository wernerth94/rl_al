import torch
import numpy as np
from PoolManagement import resetALPool, addDatapointToPool
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MockALGame:
    def __init__(self, config, noise_level=1, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_level = noise_level
        self.config = config
        self.budget = config.BUDGET
        self.sample_size = config.SAMPLE_SIZE
        self.max_reward = config.MAX_REWARD
        self.reset()
        self._set_state_shape()


    def _set_state_shape(self):
        state = self.create_state()
        self.stateSpace = state.shape[1]


    def reset(self):
        self.added_images = 0
        self.currentTestF1 = 0.4
        return self.create_state()


    def create_state(self):
        qualities = np.random.rand(self.sample_size) * 0.9
        bvssb_noise = np.random.normal(0, 0.1, size=self.sample_size)
        entr_noise = np.random.normal(0, 0.5, size=self.sample_size)
        sample = []
        for i in range(len(qualities)):
            dp = [self.currentTestF1]
            dp.append(qualities[i] +
                      self.noise_level * bvssb_noise[i]) # BvsSB
            dp.append(2 + (qualities[i] - 0.6)*2 +
                      self.noise_level * entr_noise[i])  # entropy
            # Hist of class outputs
            # internal state
            sample.append(dp)

        sample = torch.Tensor(sample)
        sample = sample.to(self.device)
        self.current_qualities = qualities
        return sample

    def step(self, action):
        self.added_images += 1
        reward = self.current_qualities[action] * self.max_reward
        noise = np.random.normal(0, 0.5 * self.max_reward)
        reward += noise
        self.currentTestF1 += reward
        done = self.added_images >= self.budget
        return self.create_state(), reward, done, {}






class ALGame:

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
        self.gameLength = config.BUDGET
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

        self.currentTestF1 = 0
        self.reset()
        self._set_state_shape()


    def _set_state_shape(self):
        state = self.createState()
        self.stateSpace = state.shape[1]


    def _sampleIDs(self, num):
        return np.random.choice(len(self.xUnlabeled), num)


    def reset(self):
        self.nInteractions = 0
        self.added_images = 0

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

        self.stateIds = self._sampleIDs(self.config.SAMPLE_SIZE)
        return self.createState()


    def createState(self):
        sample_x = self.xUnlabeled[self.stateIds]
        alFeatures = self.getClassifierFeatures(sample_x)
        mean_labeled = torch.mean(self.xLabeled, dim=0)
        mean_unlabeled = torch.mean(self.xUnlabeled, dim=0)
        mean_labeled = mean_labeled.unsqueeze(0).repeat(len(alFeatures), 1)
        mean_unlabeled = mean_unlabeled.unsqueeze(0).repeat(len(alFeatures), 1)
        # Last working version:
        # no data normalization
        # state = torch.cat([alFeatures, mean_labeled, mean_unlabeled], dim=1)

        # compute difference of the labeled pool and the sample
        diff_labeled = torch.abs(sample_x - mean_labeled)
        diff_unlabeled = torch.abs(sample_x - mean_unlabeled)
        state = torch.cat([alFeatures, diff_labeled, diff_unlabeled], dim=1)
        return state


    def getClassifierFeatures(self, x):
        eps = 1e-7
        # prediction metrics
        with torch.no_grad():
            pred = self.classifier(x).detach()
            two_highest, _ = pred.topk(2, dim=1)

            f1 = torch.repeat_interleave(torch.Tensor([self.currentTestF1]), len(x))
            entropy = -torch.mean(pred * torch.log(eps + pred) + (1+eps-pred) * torch.log(1+eps-pred), dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
            hist_list = [torch.histc(p, bins=10, min=0, max=1) for p in pred]
            hist = torch.stack(hist_list, dim=0) / self.nClasses

            f1 = f1.to(self.device)
            state = torch.cat([
                f1.unsqueeze(1),
                bVsSB.unsqueeze(1),
                entropy.unsqueeze(1),
                hist
            ], dim=1)
        return state



    def step(self, action):
        self.nInteractions += 1
        self.added_images += 1
        datapointId = self.stateIds[action]
        self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = addDatapointToPool(self.xLabeled, self.yLabeled,
                                                                                                              self.xUnlabeled, self.yUnlabeled,
                                                                                                              self.perClassIntances, datapointId)
        reward = self.fitClassifier()
        self.stateIds = self._sampleIDs(self.config.SAMPLE_SIZE)
        statePrime = self.createState()

        done = self.checkDone()
        return statePrime, reward, done, {}


    def fitClassifier(self, epochs=50, batch_size=128):
        if self.from_scratch:
            self.classifier.load_state_dict(self.initialWeights)

        #batch_size = min(batch_size, int(len(self.xLabeled)/5))
        train_dataloader = DataLoader(TensorDataset(self.xLabeled, self.yLabeled), batch_size=batch_size)
        # test_dataloader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=batch_size)

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
                    #print(f"labeled {len(self.xLabeled)}: stopped after {e} epochs")
                    break
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
        if self.added_images >= self.budget:
            done = True # budget exhausted
        return done



class DuelingALGame(ALGame):
    def __init__(self, dataset, modelFunction, config, verbose):
        super().__init__(dataset, modelFunction, config, verbose)


    def _set_state_shape(self):
        cntx, state = self.createState()
        self.stateSpace = (cntx.shape[1], state.shape[1])

    def createState(self):
        classFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds])
        f1 = classFeatures[0, 0]
        alFeatures = classFeatures[:, 1:]
        cntxFeatures = self.getPoolInfo()
        cntxFeatures = torch.cat([f1.unsqueeze(0), cntxFeatures]).unsqueeze(0)
        return cntxFeatures, alFeatures



class PALGame(ALGame):

    def __init__(self, dataset, modelFunction, config, verbose):
        super().__init__(dataset, modelFunction, config, verbose)
        self.actionSpace = 2


    def _sampleIDs(self, num=1):
        return np.random.randint(len(self.xUnlabeled))


    def createState(self):
        alFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds].unsqueeze(0))
        return alFeatures
        # alFeatures = torch.from_numpy(alFeatures).float()
        # poolFeatures = self.getPoolInfo()
        # poolFeatures = poolFeatures.unsqueeze(0).repeat(len(alFeatures), 1)
        # state = torch.cat([alFeatures, poolFeatures], dim=1)
        # return state


    def step(self, action):
        self.nInteractions += 1
        if action == 1:
            self.added_images += 1
            ret = addDatapointToPool(self.xLabeled, self.yLabeled,
                                     self.xUnlabeled, self.yUnlabeled,
                                     self.perClassIntances, self.stateIds)
            self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = ret
            reward = self.fitClassifier()
        else:
            reward = 0

        self.stateIds = self._sampleIDs()
        statePrime = self.createState()

        done = self.checkDone()
        return statePrime, reward, done, {}
