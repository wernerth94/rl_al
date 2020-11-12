import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
import os


def epsilonGreedy(q, greed, actionSpace):
    rands = np.random.rand(len(q))
    a = np.argmax(q, axis=1)
    for i in range(q.shape[0]):
        if rands[i] < greed:
            a[i] = np.random.randint(actionSpace)
    return a


def softmaxGreedy(q, greed, actionSpace):
    exp = (q / greed) ** 2
    softmax = exp / np.sum(exp, axis=1).reshape(-1, 1)
    a = np.zeros((q.shape[0], 1))
    for i in range(q.shape[0]):
        try:
            a[i] = np.random.choice(actionSpace, 1, p=softmax[i])
        except ValueError:
            print('softmax error with q', q[i], 'softmax', softmax[i])
            exit(100)
            a[i] = np.argmax(q[i])
    return a



class DDQN:

    def __init__(self, env, gamma=0.99, callbacks=[], fromCheckpoints=None, lr=0.01):
        self.gamma = gamma
        self.env = env
        self.callbacks = callbacks
        self.actionStrategy = epsilonGreedy
        self.model1 = self.createModel(fromCheckpoints, lr=lr)
        self.model2 = self.createModel(None)
        self.model2.set_weights(self.model1.get_weights())


    def createModel(self, fromCheckpoint, lr=0.001, l2Reg=0.0):
        raise NotImplementedError()

    def predict(self, inputs, greedParameter=1):
        raise NotImplementedError()


    def fit(self, memoryBatch, lr=None):
        state = memoryBatch[0]
        actions = memoryBatch[1]
        rewards = memoryBatch[2]
        nextState = memoryBatch[3]
        dones = memoryBatch[4]
        _all = range(len(nextState))

        Q1 = self.model1.predict(state)
        qPrime1 = self.model1.predict(nextState)
        qPrime2 = self.model2.predict(nextState)

        nextAction = np.argmax(qPrime1, axis=1).squeeze()
        target = np.minimum(qPrime1[_all, nextAction],
                            qPrime2[_all, nextAction])
        Q1[_all, np.squeeze(actions)] = rewards + (1-dones) * self.gamma * target

        if lr is not None:
            self.model1.optimizer.lr = lr
        hist = self.model1.fit(x=state, y=Q1, epochs=1, verbose=0, callbacks=self.callbacks)

        return hist.history['loss'][0]

    def copyWeights(self):
            self.model2.set_weights(self.model1.get_weights())



class ConvAgent(DDQN):

    def __init__(self, env, gamma=0.99, callbacks=[], fromCheckpoints=None, lr=0.01):
        super(ConvAgent, self).__init__(env, gamma=gamma, callbacks=callbacks, fromCheckpoints=fromCheckpoints, lr=lr)


    def createModel(self, fromCheckpoint, lr=0.001, l2Reg=0.0):
        model = keras.models.Sequential([
            keras.layers.Input(self.env.stateSpace),
            keras.layers.Conv1D(24, 1, activation='tanh', padding='same'),
            keras.layers.Dropout(0.1),
            keras.layers.Conv1D(12, 1, activation='tanh', padding='same'),
            keras.layers.Conv1D(1, 1, activation='linear', padding='same'),
        ])
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                      loss=keras.losses.Huber())

        if fromCheckpoint is not None and os.path.exists(fromCheckpoint):
            print('load model from ', fromCheckpoint)
            model.load_weights(fromCheckpoint)

        return model


    def predict(self, inputs, greedParameter=1):
        q = self.model1.predict(inputs)[:,:,0]
        if greedParameter <= 0:
            return q, np.argmax(q, axis=1)
        a = self.actionStrategy(q, greedParameter, self.env.actionSpace)
        return q, a



class DenseAgent(DDQN):

    def __init__(self, env, gamma=0.99, callbacks=[], fromCheckpoints=None, lr=0.01):
        super(DenseAgent, self).__init__(env, gamma=gamma, callbacks=callbacks, fromCheckpoints=fromCheckpoints, lr=lr)


    def createModel(self, fromCheckpoint, lr=0.001, l2Reg=0.0):
        model = keras.models.Sequential([
                keras.layers.Input(self.env.stateSpace),
                keras.layers.Dense(10, activation=keras.layers.LeakyReLU()),
                keras.layers.Dense(self.env.actionSpace) ])
        opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                      loss=keras.losses.Huber())

        if fromCheckpoint is not None and os.path.exists(fromCheckpoint):
            print('load model from ', fromCheckpoint)
            model.load_weights(fromCheckpoint)

        return model

    def predict(self, inputs, greedParameter=1):
        q = self.model1.predict(inputs)
        if greedParameter <= 0:
            return q, np.argmax(q, axis=1)
        a = self.actionStrategy(q, greedParameter, self.env.actionSpace)
        return q, a



class Baseline_Entropy:
    def predict(self, state, greedParameter=0):
        scores = state[:, :, 3]
        return scores, np.argmax(scores, axis=1)


class Baseline_BvsSB:
    def predict(self, state, greedParameter=0):
        scores = state[:, :, 2]
        return scores, np.argmax(scores, axis=1)


class Baseline_Random:
    def __init__(self, env):
        self.actionSpace = env.actionSpace

    def predict(self, state, greedParameter=0):
        return np.random.rand(len(state), self.actionSpace), np.array([0])