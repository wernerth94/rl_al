import os, shutil
import numpy as np

def toList(l):
    try:
        l[0]
        return l
    except TypeError:
        return [l]

class Memory:

    def __init__(self, env, maxLength=np.inf, buildFromBacklog=False, dataAugmentors=[]):
        self.env = env
        self.dataAugmentors = dataAugmentors
        self.maxLength = maxLength
        self._createEmptyMemory()
        if buildFromBacklog:
            self._createFromBacklog()


    def _createEmptyMemory(self):
        self.state = np.zeros([0] + toList(self.env.stateSpace))
        self.actions = np.zeros((0, 1), dtype=int)
        self.rewards = np.zeros((0, 1))
        self.newState = np.zeros([0] + toList(self.env.stateSpace))
        self.dones = np.zeros((0, 1), dtype=int)


    def _createFromBacklog(self):
        backlogDir = 'memoryBacklog'
        fullLength = 0
        for file in os.listdir(backlogDir):
            saveFolder = os.path.join(backlogDir, file)
            mem = Memory(self.env, maxLength=np.inf)
            mem.loadFromDisk(saveFolder)
            self._append(mem.state, mem.actions, mem.rewards, mem.newState, mem.dones)
            fullLength += len(mem)

        self.maxLength = fullLength


    def _append(self, state, action, reward, newState, done):
        self.state = np.append(self.state, state.reshape([-1] + toList(self.env.stateSpace)), axis=0)
        self.actions = np.append(self.actions, np.array(action, dtype=int).reshape([-1, 1]), axis=0)
        self.rewards = np.append(self.rewards, np.array(reward).reshape([-1, 1]), axis=0)
        self.newState = np.append(self.newState, newState.reshape([-1] + toList(self.env.stateSpace)), axis=0)
        self.dones = np.append(self.dones, np.array(done, dtype=int).reshape([-1, 1]), axis=0)

        if len(self.actions) > self.maxLength:
            offset = len(self.actions) - self.maxLength
            self.state = self.state[offset:]
            self.actions = self.actions[offset:]
            self.rewards = self.rewards[offset:]
            self.newState = self.newState[offset:]
            self.dones = self.dones[offset:]


    def addMemory(self, state, action, reward, newState, done):
        self._append(state, action, reward, newState, done)
        for da in self.dataAugmentors:
            s, sP = da.apply([state, newState])
            self._append(s, action, reward, sP, done)


    def sampleMemory(self, size):
        idx = np.random.choice(len(self.actions), min(len(self.actions), size))
        return (self.state[idx],
                self.actions[idx], self.rewards[idx],
                self.newState[idx],
                self.dones[idx])

    def writeToDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)

        np.save(os.path.join(saveFolder, 'state.npy'), self.state)
        np.save(os.path.join(saveFolder, 'actions.npy'), self.actions)
        np.save(os.path.join(saveFolder, 'rewards.npy'), self.rewards)
        np.save(os.path.join(saveFolder, 'newState.npy'), self.newState)
        np.save(os.path.join(saveFolder, 'dones.npy'), self.dones)

    def loadFromDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            self.state = np.load(os.path.join(saveFolder, 'state.npy'))
            self.actions = np.load(os.path.join(saveFolder, 'actions.npy'))
            self.rewards = np.load(os.path.join(saveFolder, 'rewards.npy'))
            self.newState = np.load(os.path.join(saveFolder, 'newState.npy'))
            self.dones = np.load(os.path.join(saveFolder, 'dones.npy'))
            return True
        return False

    def __len__(self):
        return len(self.actions)