import os, shutil
import numpy as np

def toList(l):
    try:
        l[0]
        return l
    except TypeError:
        return [l]


class Memory:

    def __init__(self):
        self.nColumns = -1

    def _append(self, row: np.array):
        self.memory = np.append(self.memory, row, axis=0)

        if len(self.memory) > self.maxLength:
            offset = len(self.memory) - self.maxLength
            self.memory = self.memory[offset:]

    def writeToDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)
        np.save(os.path.join(saveFolder, 'memory.npy'), self.memory)


    def loadFromDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            memFile = os.path.join(saveFolder, 'memory.npy')
            if os.path.exists(memFile):
                self.memory = np.load(memFile)
                print('loaded memory from', memFile)
                return True
        return False

    def __len__(self):
        return len(self.memory)

    def clear(self):
        del self.memory
        self.memory = np.zeros((0, self.nColumns))



class NStepVMemory(Memory):

    def __init__(self, stateSpace, nSteps, maxLength=np.inf, dataAugmentors=[]):
        super(NStepVMemory, self).__init__()
        self.nSteps = nSteps
        self.stateSpace = stateSpace
        self.dataAugmentors = dataAugmentors
        self.maxLength = maxLength

        self.nColumns  = 2 * self.stateSpace # state + newState
        self.nColumns += self.nSteps # rewards
        self.nColumns += 1 # done

        self.memory = np.zeros((0, self.nColumns))


    def argsToRow(self, state:np.array, rewardList:list, newState:np.array, done:int):
        row = np.expand_dims(np.copy(state), axis=0)
        for r in rewardList:
            row = np.concatenate([row, np.array(r).reshape(1, -1)], axis=1)
        row = np.concatenate([row, np.expand_dims(newState, axis=0)], axis=1)
        row = np.concatenate([row, np.array(done).reshape(1, -1)], axis=1)
        return row


    def rowsToArgs(self, chunk):
        state = chunk[:, :self.stateSpace]

        rewardList = []
        for i in range(self.nSteps):
            offset = self.stateSpace + i
            reward = chunk[:, offset]
            rewardList.append(reward)

        offset = self.stateSpace + self.nSteps
        newState = chunk[:, offset : offset+self.stateSpace ]

        done = chunk[:, -1]
        return state, rewardList, newState, done


    def addMemory(self, state:np.array, rewardList:list, newState:np.array, done:int):
        row = self.argsToRow(state, rewardList, newState, done)
        self._append(row)
        # for da in self.dataAugmentors:
        #     s, sP = da.apply([state, newState])
        #     self._append(s, action, reward, sP, done)


    def sampleMemory(self, size):
        idx = np.random.choice(len(self.memory), min(len(self.memory), size), replace=False)
        state, rewardList, newState, done = self.rowsToArgs(self.memory[idx])
        return state, rewardList, newState, done




class NStepQMemory(Memory):

    def __init__(self, stateSpace, nSteps, maxLength=np.inf, dataAugmentors=[]):
        super(NStepQMemory, self).__init__()
        self.nSteps = nSteps
        self.stateSpace = stateSpace
        self.dataAugmentors = dataAugmentors
        self.maxLength = maxLength

        self.nColumns  = 2 * self.stateSpace # state + newState
        self.nColumns += self.nSteps # rewards
        self.nColumns += 2 # done + action

        self.memory = np.zeros((0, self.nColumns))


    def argsToRow(self, state:np.array, action:int, rewardList:list, newState:np.array, done:int):
        row = np.expand_dims(np.copy(state), axis=0)
        row = np.concatenate([row, np.array(action).reshape(1, -1)], axis=1)
        for r in rewardList:
            row = np.concatenate([row, np.array(r).reshape(1, -1)], axis=1)
        row = np.concatenate([row, np.expand_dims(newState, axis=0)], axis=1)
        row = np.concatenate([row, np.array(done).reshape(1, -1)], axis=1)
        return row


    def rowsToArgs(self, chunk):
        state = chunk[:, :self.stateSpace]
        action = chunk[:, self.stateSpace]

        rewardList = []
        for i in range(self.nSteps):
            offset = self.stateSpace + 1 + i
            reward = chunk[:, offset]
            rewardList.append(reward)

        offset = self.stateSpace + 1 + self.nSteps
        newState = chunk[:, offset : offset+self.stateSpace ]

        done = chunk[:, -1]
        return state, action, rewardList, newState, done


    def addMemory(self, state:np.array, action:int, rewardList:list, newState:np.array, done:int):
        row = self.argsToRow(state, action, rewardList, newState, done)
        self._append(row)
        # for da in self.dataAugmentors:
        #     s, sP = da.apply([state, newState])
        #     self._append(s, action, reward, sP, done)


    def sampleMemory(self, size):
        idx = np.random.choice(len(self.memory), min(len(self.memory), size), replace=False)
        state, action, rewardList, newState, done = self.rowsToArgs(self.memory[idx])
        return state, action, rewardList, newState, done