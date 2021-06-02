import tensorflow.keras as keras
import gc
import numpy as np
import bottleneck as bn
from PoolManagement import resetALPool, sampleNewBatch, addDatapointToPool


class ALGame:

    stateSpace = 3

    def __init__(self, dataset, modelFunction, config, verbose):
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
        self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
        self.classifier = modelFunction(inputShape=self.x_test.shape[1:],
                                        numClasses=self.y_test.shape[1])
        self.initialF1 = 0
        self.currentTestF1 = 0
        self.hardReset = True
        self.reset()
        state = self.createState()
        self.stateSpace = state.shape[1]
        self.hardReset = True

    def reset(self):
        # self.initialF1 = 0
        # self.currentTestF1 = 0
        self.nInteractions = 0

        if self.hardReset:
            self.hardReset = False
            self.classifier = self.modelFunction(inputShape=self.x_test.shape[1:],
                                                 numClasses=self.y_test.shape[1])
            self.initialWeights = self.classifier.get_weights()

            self.xLabeled, self.yLabeled, \
            self.xUnlabeled, self.yUnlabeled, self.perClassIntances = resetALPool(self.dataset,
                                                                                  self.config.INIT_POINTS_PER_CLASS)
            self.fitClassifier()
            self.initialF1 = self.currentTestF1

        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        return self.createState()


    def createState(self):
        alFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds])
        # state = addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures)
        state = alFeatures
        return state


    def getClassifierFeatures(self, x):
        eps = 1e-7
        # prediction metrics
        pred = self.classifier.predict(x)
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


    def step(self, action):
        self.nInteractions += 1
        datapointId = self.stateIds[action]
        self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = addDatapointToPool(self.xLabeled,
                                                                                                              self.yLabeled,
                                                                                                              self.xUnlabeled,
                                                                                                              self.yUnlabeled,
                                                                                                              self.perClassIntances,
                                                                                                              datapointId)
        reward = self.fitClassifier()
        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        statePrime = self.createState()

        done, self.hardReset = self.checkDone()
        return statePrime, reward, done


    def fitClassifier(self, epochs=50, batch_size=32):
        if self.from_scratch:
            self.classifier.set_weights(self.initialWeights)
        train_history = self.classifier.fit(self.xLabeled, self.yLabeled, batch_size=batch_size, epochs=epochs, verbose=0,
                                            callbacks=[self.es], validation_data=(self.x_test, self.y_test))

        self.perClassF1 = train_history.history['val_f1_score'][-1]
        newTestF1, self.currentTestLoss = np.mean(self.perClassF1), train_history.history['val_loss'][-1]

        reward = 0
        if self.rewardShaping:
            reward = (newTestF1 - self.currentTestF1 - self.labelCost) * self.rewardScaling
        else:
            raise NotImplementedError()
        self.currentTestF1 = newTestF1
        return reward


    def checkDone(self):
        done = self.nInteractions >= self.gameLength
        hardReset = False

        if len(self.xLabeled) - (self.config.INIT_POINTS_PER_CLASS * self.yUnlabeled.shape[1]) >= self.budget:
            done = True
            hardReset = True

        return done, hardReset



class ALStreamingGame(ALGame):

    def __init__(self, dataset, modelFunction, config, verbose):
        assert config.SAMPLE_SIZE == 1
        super(ALStreamingGame, self).__init__(dataset, modelFunction, config, verbose)
        self.actionSpace = 2


    def createState(self):
        alFeatures = self.getClassifierFeatures(self.xUnlabeled[self.stateIds])
        # state = addPoolInformation(xUnlabeled, xLabeled, stateIds, alFeatures)
        state = alFeatures
        return state


    def getClassifierFeatures(self, x):
        eps = 1e-5
        # prediction metrics
        pred = self.classifier.predict(x)
        part = (-bn.partition(-pred, 4, axis=1))[:,:4] # collects the two highest entries
        struct = np.sort(part, axis=1)

        # weightedF1 = np.average(pred * self.perClassF1, axis=1)
        entropy = -np.average(pred * np.log(eps + pred) + (1+eps-pred) * np.log(1+eps-pred), axis=1)
        bVsSB = 1 - (struct[:, -1] - struct[:, -2])

        state = np.expand_dims(bVsSB, axis=-1)
        # state = np.stack([weightedF1, bVsSB, entropy], axis=-1)
        # state = np.concatenate([state, struct], axis=1)
        return state


    def step(self, action):
        self.nInteractions += 1
        datapointId = self.stateIds[0]
        if action == 1:
            self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = addDatapointToPool(self.xLabeled,
                                                                                                                  self.yLabeled,
                                                                                                                  self.xUnlabeled,
                                                                                                                  self.yUnlabeled,
                                                                                                                  self.perClassIntances,
                                                                                                                  datapointId)
            reward = self.fitClassifier()
        else:
            reward = 0

        self.stateIds = sampleNewBatch(self.xUnlabeled, self.config.SAMPLE_SIZE)
        statePrime = self.createState()

        done, self.hardReset = self.checkDone()
        return statePrime, reward, done




class CreditAssignmentGame:

    def __init__(self, gameLength, stateSpace=5, sampleSize=20):
        self.gameLength = gameLength
        self.stateSpace = stateSpace
        self.sampleSize = sampleSize
        self.scale = 1 / gameLength
        self.coeffs = np.array([1,-1, 1, -1, 1])
        #self.coeffs = np.random.rand(stateSpace)*self.scale - self.scale/2


    def createState(self):
        self.currentState = np.random.rand(self.sampleSize, self.stateSpace)*self.scale - self.scale/2
        return self.currentState


    def reset(self):
        self.currentState = None
        self.nInteractions = 0
        return self.createState()


    def step(self, action):
        self.nInteractions += 1
        reward = self.currentState[action] @ self.coeffs
        done = self.nInteractions >= self.gameLength
        return reward, self.createState(), done