import tensorflow.keras as keras
import gc
import numpy as np
from sklearn.metrics import classification_report
import bottleneck as bn


class ALGameBase:

    def __init__(self, dataset, budget, maxInteractions, modelFunction,  labelCost, rewardShaping, initPointsPerClass, verbose):
        self.x_train = dataset[0]
        self.y_train = dataset[1]
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.nClasses = self.y_train.shape[1]

        self.budget = budget
        self.maxInteractions = maxInteractions
        self.labelCost = labelCost
        self.rewardShaping = rewardShaping
        self.pointsPerClass = initPointsPerClass
        self.verbose = verbose

        self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)

        self.classifier = modelFunction(inputShape=self.x_train.shape[1:],
                                        numClasses=self.y_train.shape[1])
        self.initialWeights = self.classifier.get_weights()
        self.modelFunction = modelFunction


    def _initLabeledDataset(self):
        del self.xLabeled
        del self.yLabeled
        del self.xUnlabeled
        del self.yUnlabeled
        gc.collect()
        self.xLabeled, self.yLabeled = [], []

        ids = np.arange(self.x_train.shape[0], dtype=int)
        np.random.shuffle(ids)
        addedDataPerClass = np.zeros(self.y_train.shape[1])
        usedIds = []
        for i in ids:
            label = np.argmax(self.y_train[i])
            if addedDataPerClass[label] < self.pointsPerClass:
                self.xLabeled.append(self.x_train[i])
                self.yLabeled.append(self.y_train[i])
                usedIds.append(i)
                addedDataPerClass[label] += 1
            if sum(addedDataPerClass) >= self.pointsPerClass * len(addedDataPerClass):
                break
        unusedIds = [i for i in np.arange(self.x_train.shape[0]) if i not in usedIds]
        self.xLabeled = np.array(self.xLabeled)
        self.yLabeled = np.array(self.yLabeled)
        self.xUnlabeled = np.array(self.x_train[unusedIds])
        self.yUnlabeled = np.array(self.y_train[unusedIds])


    def _fitClassifier(self, epochs=50, batch_size=32):
        #self.classifier.set_weights(self.initialWeights)
        train_history = self.classifier.fit(self.xLabeled, self.yLabeled, batch_size=batch_size, epochs=epochs, verbose=0,
                                            callbacks=[self.es], validation_data=(self.x_test, self.y_test))

        yHat = self.classifier.predict(self.x_test)
        yHat = np.argmax(yHat, axis=1)

        self.report = classification_report(np.argmax(self.y_test, axis=1), yHat, output_dict=True)
        self.perClassPrec = [self.report[str(c)]['precision'] for c in range(self.nClasses)]
        self.perClassRec = [self.report[str(c)]['recall'] for c in range(self.nClasses)]
        self.perClassF1 = [self.report[str(c)]['f1-score'] for c in range(self.nClasses)]

        unique, counts = np.unique(np.argmax(self.yLabeled, axis=1), return_counts=True)
        d = dict(zip(unique, counts))
        self.perClassIntances = [0 for _ in range(self.nClasses)]
        for cls, count in d.items():
            self.perClassIntances[int(cls)] = count

        return np.mean(self.perClassF1), np.min(train_history.history['val_loss'])


    def reset(self):
        self.addedImages = 0
        self.initialF1 = self.currentTestF1
        self.currentStateIds = np.random.choice(self.xUnlabeled.shape[0], (self.sampleSize, self.imgsToAvrg))

        if self._budgetExhausted() or self.firstReset:
            self.firstReset = False
            # full reset
            self.numInteractions = 0
            self.currentTestF1 = 0

            self.classifier = self.modelFunction(inputShape=self.x_train.shape[1:],
                                                 numClasses=self.y_train.shape[1])
            self.initialWeights = self.classifier.get_weights()

            self._initLabeledDataset()
            self.currentTestF1, self.currentTestLoss = self._fitClassifier()
            self.initialF1 = self.currentTestF1

        return self._createState()


    def _createState(self):
        raise NotImplementedError()


    def checkDone(self):
        return self.numInteractions >= self.maxInteractions or \
               self.addedImages >= self.gameLength or \
               self._budgetExhausted()


    def _budgetExhausted(self):
        return len(self.xLabeled) - (self.nClasses * self.pointsPerClass) >= self.budget




class ImageClassificationGame(ALGameBase):

    def __init__(self, dataset, modelFunction, config, verbose=True):
        self.config = config
        self.sampleSize = config.SAMPLE_SIZE
        self.imgsToAvrg = config.DATAPOINTS_TO_AVRG
        self.gameLength = config.GAME_LENGTH
        self.rewardScaling = config.REWARD_SCALE

        super(ImageClassificationGame, self).__init__(dataset, config.BUDGET, config.MAX_INTERACTIONS_PER_GAME,
                                                      modelFunction, config.LABEL_COST, config.REWARD_SHAPING,
                                                      config.INIT_POINTS_PER_CLASS, verbose)

        self.stateSpace = self._calcSateSpace()
        self.actionSpace = config.SAMPLE_SIZE + 1

        self.xLabeled = self.yLabeled = self.xUnlabeled = self.yUnlabeled = np.array([])
        self._initLabeledDataset()
        self.currentTestF1, self.currentTestLoss = self._fitClassifier()
        self.initialF1 = self.currentTestF1
        self.firstReset = True


    def _calcSateSpace(self):
        space = 0
        modelMetrics = 0
        space += len(self.classifier.get_weights()) * modelMetrics
        predMetrics = 4
        space += self.sampleSize * predMetrics
        otherMetrics = 2
        space += otherMetrics

        return space


    def _createState(self):
        eps = 1e-5
        # prediction metrics
        bVsSB, entropy, topProb, weightedF1 = [], [], [], []
        for i in range(self.currentStateIds.shape[1]):
            x = self.xUnlabeled[self.currentStateIds[:, i]]
            pred = self.classifier.predict(x)
            part = (-bn.partition(-pred, 2, axis=1))[:,:2] # collects the two highest entries
            struct = np.sort(part, axis=1)

            weightedF1.append(np.sum(pred * self.perClassF1))
            topProb.append(struct[:,-1])
            entropy.append(-np.average(pred * np.log(eps + pred) + (1+eps-pred) * np.log(1+eps-pred), axis=1))
            bVsSB.append(1 - (struct[:, -1] - struct[:, -2]))

        meanWeightedF1 = np.expand_dims( np.mean(np.stack(topProb), axis=0) , axis=0)
        meanTop = np.expand_dims( np.mean(np.stack(topProb), axis=0) , axis=0)
        meanBVsSB = np.expand_dims( np.mean(np.stack(bVsSB), axis=0) , axis=0)
        meanEntropy = np.expand_dims( np.mean(np.stack(entropy), axis=0) , axis=0)

        # model metrics
        # weights = self.classifier.get_weights()
        # modelMetrics = list()
        # for layer in weights:
        #     modelMetrics += [np.mean(layer), np.std(layer), np.linalg.norm(layer)]  # , np.linalg.norm(layer, ord=2)]
        # modelMetrics = np.array(modelMetrics)

        meanF1 = np.array(np.mean(self.perClassF1)).reshape([1, -1])
        imgProgress = np.array(len(self.xLabeled)-(self.y_train.shape[1]*self.pointsPerClass) / float(self.budget)).reshape([1, -1])

        state = np.concatenate([meanF1, imgProgress,
                                #modelMetrics.reshape([1, -1]),
                                meanBVsSB, meanEntropy, meanTop, meanWeightedF1], axis=1)
        return state


    def step(self, action):
        self.numInteractions += 1

        if int(action) >= self.sampleSize:
            # replace random image
            self.currentStateIds[np.random.randint(0, self.sampleSize)] = np.random.choice(len(self.xUnlabeled),
                                                                                           self.imgsToAvrg)
        else:
            indices = self.currentStateIds[int(action)]
            for a in range(len(indices)):
                idx = indices[a]
                self.xLabeled = np.append(self.xLabeled, self.xUnlabeled[idx:idx + 1], axis=0)
                self.yLabeled = np.append(self.yLabeled, self.yUnlabeled[idx:idx + 1], axis=0)
                self.xUnlabeled = np.delete(self.xUnlabeled, idx, axis=0)
                self.yUnlabeled = np.delete(self.yUnlabeled, idx, axis=0)
                self.addedImages += 1
                # adjust indices of current state
                for i in range(self.currentStateIds.shape[0]):
                    if i != int(action):
                        for j in range(self.currentStateIds.shape[1]):
                            if idx < self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] -= 1
                            elif idx == self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] = np.random.randint(len(self.xUnlabeled))
                    else:
                        for j in range(a+1, self.currentStateIds.shape[1]):
                            if idx < self.currentStateIds[i, j]:
                                self.currentStateIds[i, j] -= 1
            # replace missing image
            self.currentStateIds[int(action)] = np.random.choice(len(self.xUnlabeled), self.imgsToAvrg)

        reward = 0
        if int(action) < self.sampleSize:
            # retrain classifier
            newTestF1, self.currentTestLoss = self._fitClassifier()
            if self.rewardShaping:
                reward = (newTestF1 - self.currentTestF1 - self.labelCost * 1) * self.rewardScaling
            self.currentTestF1 = newTestF1

        done = self.checkDone()
        if done and not self.rewardShaping:
            reward = (self.currentTestF1 - self.initialF1 - self.labelCost * self.addedImages) * self.rewardScaling

        return self._createState(), reward, done, {}



class ConvALGame(ALGameBase):

    def __init__(self, dataset, modelFunction, config, verbose=True):
        self.config = config
        self.sampleSize = config.SAMPLE_SIZE
        self.imgsToAvrg = config.DATAPOINTS_TO_AVRG
        self.gameLength = config.GAME_LENGTH
        self.rewardScaling = config.REWARD_SCALE

        super(ConvALGame, self).__init__(dataset, config.BUDGET, config.MAX_INTERACTIONS_PER_GAME,
                                                      modelFunction, config.LABEL_COST, config.REWARD_SHAPING,
                                                      config.INIT_POINTS_PER_CLASS, verbose)

        self.actionSpace = config.SAMPLE_SIZE
        self.stateSpace = self._calcSateSpace()

        self.xLabeled = self.yLabeled = self.xUnlabeled = self.yUnlabeled = np.array([])
        self._initLabeledDataset()
        self.currentTestF1, self.currentTestLoss = self._fitClassifier()
        self.initialF1 = self.currentTestF1
        self.firstReset = True


    def _calcSateSpace(self):
        space = 0
        modelMetrics = 0
        space += len(self.classifier.get_weights()) * modelMetrics
        predMetrics = 4
        space += predMetrics
        otherMetrics = 2
        space += otherMetrics

        return [self.actionSpace, space]


    def _createState(self):
        eps = 1e-5
        # prediction metrics
        bVsSB, entropy, topProb, weightedF1 = [], [], [], []
        for i in range(self.currentStateIds.shape[1]):
            x = self.xUnlabeled[self.currentStateIds[:, i]]
            pred = self.classifier.predict(x)
            part = (-bn.partition(-pred, 2, axis=1))[:,:2] # collects the two highest entries
            struct = np.sort(part, axis=1)

            weightedF1.append(np.average(pred * self.perClassF1))
            topProb.append(struct[:,-1])
            entropy.append(-np.average(pred * np.log(eps + pred) + (1+eps-pred) * np.log(1+eps-pred), axis=1))
            bVsSB.append(1 - (struct[:, -1] - struct[:, -2]))

        meanWeightedF1 = np.expand_dims( np.mean(np.stack(topProb), axis=0) , axis=0)
        meanTop = np.expand_dims( np.mean(np.stack(topProb), axis=0) , axis=0)
        meanBVsSB = np.expand_dims( np.mean(np.stack(bVsSB), axis=0) , axis=0)
        meanEntropy = np.expand_dims( np.mean(np.stack(entropy), axis=0) , axis=0)

        meanF1 = np.full_like(meanTop, np.mean(self.perClassF1))
        imgProgress = np.array(len(self.xLabeled)-(self.y_train.shape[1]*self.pointsPerClass)).reshape([1, -1]) / float(self.budget)

        state = np.stack([meanF1, np.full_like(meanTop, imgProgress),
                                meanBVsSB, meanEntropy, meanTop, meanWeightedF1], axis=-1)
        return state


    def step(self, action):
        self.numInteractions += 1

        # add images
        indices = self.currentStateIds[int(action)]
        for a in range(len(indices)):
            idx = indices[a]
            self.xLabeled = np.append(self.xLabeled, self.xUnlabeled[idx:idx + 1], axis=0)
            self.yLabeled = np.append(self.yLabeled, self.yUnlabeled[idx:idx + 1], axis=0)
            self.xUnlabeled = np.delete(self.xUnlabeled, idx, axis=0)
            self.yUnlabeled = np.delete(self.yUnlabeled, idx, axis=0)
            self.addedImages += 1

        # sample a new batch
        self.currentStateIds = np.random.choice(self.xUnlabeled.shape[0], (self.sampleSize, self.imgsToAvrg))

        reward = 0
        if int(action) < self.sampleSize:
            # retrain classifier
            newTestF1, self.currentTestLoss = self._fitClassifier()
            if self.rewardShaping:
                reward = (newTestF1 - self.currentTestF1 - self.labelCost * 1) * self.rewardScaling
            self.currentTestF1 = newTestF1

        done = self.checkDone()
        if done and not self.rewardShaping:
            reward = (self.currentTestF1 - self.initialF1 - self.labelCost * self.addedImages) * self.rewardScaling

        return self._createState(), reward, done, {}