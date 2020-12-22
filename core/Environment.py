import tensorflow.keras as keras
import gc
import numpy as np
import bottleneck as bn


class ALGame:

    def __init__(self, dataset, modelFunction, config, verbose):
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.nClasses = self.y_test.shape[1]

        self.config = config
        self.labelCost = config.LABEL_COST
        self.rewardScaling = config.REWARD_SCALE
        self.rewardShaping = config.REWARD_SHAPING
        self.verbose = verbose

        self.stateSpace = 3

        self.modelFunction = modelFunction
        self.es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=1)
        self.classifier = modelFunction(inputShape=self.y_test.shape[1:],
                                        numClasses=self.y_test.shape[1])


    def createState(self, x):
        eps = 1e-5
        # prediction metrics
        pred = self.classifier.predict(x)
        part = (-bn.partition(-pred, 2, axis=1))[:,:2] # collects the two highest entries
        struct = np.sort(part, axis=1)

        weightedF1 = np.average(pred * self.perClassF1, axis=1)
        entropy = -np.average(pred * np.log(eps + pred) + (1+eps-pred) * np.log(1+eps-pred), axis=1)
        bVsSB = 1 - (struct[:, -1] - struct[:, -2])

        state = np.stack([bVsSB, entropy, weightedF1], axis=-1)
        return state


    def fitClassifier(self, x_labeled, y_labeled, epochs=50, batch_size=32):
        #self.classifier.set_weights(self.initialWeights)
        train_history = self.classifier.fit(x_labeled, y_labeled, batch_size=batch_size, epochs=epochs, verbose=0,
                                            callbacks=[self.es], validation_data=(self.x_test, self.y_test))

        self.perClassF1 = train_history.history['val_f1_score'][-1]
        newTestF1, self.currentTestLoss = np.mean(self.perClassF1), np.min(train_history.history['val_loss'])

        reward = 0
        if self.rewardShaping:
            reward = (newTestF1 - self.currentTestF1 - self.labelCost * 1) * self.rewardScaling
        self.currentTestF1 = newTestF1
        return reward


    def reset(self, initial_x:np.array, initial_y:np.array):
        self.initialF1 = 0
        self.currentTestF1 = 0

        # full reset
        self.classifier = self.modelFunction(inputShape=initial_x.shape[1:],
                                             numClasses=initial_y.shape[1])
        self.initialWeights = self.classifier.get_weights()
        _ = self.fitClassifier(initial_x, initial_y)
        self.initialF1 = self.currentTestF1
