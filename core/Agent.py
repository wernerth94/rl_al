import tensorflow.keras as keras
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
import os
import AutoEncoder



class DDVN:

    def __init__(self, stateSpace, nSteps, clipped=False, gamma=0.99, callbacks=[], fromCheckpoints=None,
                 lr=0.001, nHidden=80, activation='relu'):
        self.nSteps = nSteps
        self.gamma = gamma
        self.stateSpace = stateSpace
        self.callbacks = callbacks
        self.clipped = clipped
        self.model1 = self.createStateValueModel(fromCheckpoints, lr=lr, nHidden=nHidden, activation=activation)
        self.model2 = self.createStateValueModel(None, nHidden=nHidden, activation=activation)
        self.model2.set_weights(self.model1.get_weights())



    def createStateValueModel(self, fromCheckpoint, lr=0.001, nHidden=10, activation='tanh'):
        if fromCheckpoint is not None and os.path.exists(fromCheckpoint):
            print('loaded model from ', fromCheckpoint)
            return keras.models.load_model(fromCheckpoint)
        else:
            model = keras.models.Sequential([
                keras.layers.Input(self.stateSpace),
                keras.layers.Dense(nHidden, activation=activation),
                keras.layers.Dense(1)])
            opt = tfa.optimizers.RectifiedAdam(learning_rate=lr)
            model.compile(optimizer=tfa.optimizers.Lookahead(opt),
                          loss=keras.losses.Huber())

        return model


    def predict(self, inputs, greedParameter=1):
        v = self.model1.predict(inputs)
        if greedParameter <= 0 or np.random.rand() > greedParameter:
            a = np.argmax(v, axis=0)
            return v[:,0], a
        else:
            i = np.random.randint(len(inputs))
            return v[:,0], np.array(i).reshape(-1)


    def fit(self, memoryBatch, lr=None, batchSize=16):
        state = memoryBatch[0]
        rewards = memoryBatch[1]
        nextStates = memoryBatch[2]
        dones = memoryBatch[3]
        _all = range(len(state))

        #V1 = self.model1.predict(state)[:,0]
        vPrime2 = self.model2.predict(nextStates)[:,0]

        #nextAction = np.argmax(vPrime1, axis=1)  # .squeeze()
        if self.clipped:
            vPrime1 = self.model1.predict(nextStates)[:,0]
            target = np.minimum(vPrime1, vPrime2)
        else:
            target = vPrime2

        R = np.zeros(len(state))
        for i in range(len(rewards)):
            R += (self.gamma ** i) * rewards[i]
        # V1 = R + (1 - dones) * self.gamma * target # OLD and wrong
        V1 = R + (1 - dones) * (self.gamma**len(rewards)) * target

        if lr is not None:
            self.model1.optimizer.lr = lr
        hist = self.model1.fit(x=state, y=V1, epochs=1, batch_size=batchSize, verbose=0, callbacks=self.callbacks)

        return sum(hist.history['loss'])


    def copyWeights(self):
            self.model2.set_weights(self.model1.get_weights())


    def getAgentWeights(self):
        return [self.model1.get_weights(), self.model2.get_weights()]


    def setAgentWeights(self, weights:list):
        self.model1.set_weights(weights[0])
        self.model2.set_weights(weights[1])




class DynaV(DDVN):
    def __init__(self, stateSpace, nSteps, latentSpace=10, clipped=False, gamma=0.99, callbacks=[], fromCheckpoints=None, lr=0.01):
        super(DynaV, self).__init__(stateSpace, nSteps, clipped, gamma, callbacks, fromCheckpoints, lr)
        self.latentSpace = latentSpace
        self.stateTransModel = self.createStateTransitionModel(fromCheckpoints=None)


    def createStateTransitionModel(self, fromCheckpoint, lr=0.001):
        if fromCheckpoint is not None:
            pass
        else:
            model = AutoEncoder.VAE(self.stateSpace)

        return model


    def predict(self, inputs, greedParameter=1):
        V, a = super(DynaV, self).predict(inputs, greedParameter)
        # planning

        return a


    def fit(self, memoryBatch, lr=None):
        # direkt RL
        direktRLLoss = super(DynaV, self).fit(memoryBatch, lr)

        states = memoryBatch[0]
        newStates = memoryBatch[2]
        # fit state transition model
        hist = self.stateTransModel.fit(states, newStates, epochs=1, verbose=0)
        transitionLoss = hist.history['mae'][-1]
        # simulate experience
        simStates = self.stateTransModel(states)
        # fit on simulated experience
        simulatedRLLoss = super(DynaV, self).fit((states, memoryBatch[1], simStates, memoryBatch[3]), lr)

        return direktRLLoss, transitionLoss, simulatedRLLoss


    def getAgentWeights(self):
        stateValueWeights = super(DynaV, self).getAgentWeights()
        return [stateValueWeights, self.stateTransModel.get_weights()]

    def setAgentWeights(self, weights: list):
        super(DynaV, self).setAgentWeights(weights[0])
        self.stateTransModel.set_weights(weights[1])



class Baseline_Entropy:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greedParameter=0):
        scores = state[:, 2]
        return scores, np.expand_dims(np.argmax(scores), axis=-1)


class Baseline_BvsSB:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greedParameter=0):
        scores = state[:, 1]
        return scores, np.expand_dims(np.argmax(scores), axis=-1)


class Baseline_Random:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, state, greedParameter=0):
        return None, np.expand_dims(np.random.randint(len(state)), axis=-1)