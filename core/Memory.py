import os, shutil
import numpy as np
import torch

def toList(l):
    try:
        l[0]
        return l
    except TypeError:
        return [l]


class Memory:

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nColumns = -1

    def _append(self, row:torch.Tensor):
        self.memory = torch.cat([self.memory, row], dim=0)

        if len(self.memory) > self.maxLength:
            offset = len(self.memory) - self.maxLength
            self.memory = self.memory[offset:]

    def writeToDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            shutil.rmtree(saveFolder)
        os.mkdir(saveFolder)
        torch.save(self.memory, os.path.join(saveFolder, 'memory.pt'))


    def loadFromDisk(self, saveFolder='memory'):
        if os.path.exists(saveFolder):
            memFile = os.path.join(saveFolder, 'memory.npy')
            if os.path.exists(memFile):
                self.memory = torch.load(memFile).to(self.device)
                print('loaded memory from', memFile)
                return True
        return False

    def __len__(self):
        return len(self.memory)

    def clear(self):
        del self.memory
        self.memory = torch.zeros((0, self.nColumns)).to(self.device)



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

        self.memory = torch.zeros((0, self.nColumns)).to(self.device)


    def argsToRow(self, state:torch.Tensor, rewardList:list, newState:torch.Tensor, done:int):
        row = torch.unsqueeze(state.clone(), dim=0)
        for r in rewardList:
            row = torch.cat([row, torch.Tensor([r]).reshape(1, -1)], dim=1)
        row = torch.cat([row, torch.unsqueeze(newState, dim=0)], dim=1)
        row = torch.cat([row, torch.Tensor([done]).reshape(1, -1)], dim=1)
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


    def addMemory(self, state:torch.Tensor, rewardList:list, newState:torch.Tensor, done:int):
        row = self.argsToRow(state, rewardList, newState, done)
        self._append(row)
        # for da in self.dataAugmentors:
        #     s, sP = da.apply([state, newState])
        #     self._append(s, action, reward, sP, done)


    def sampleMemory(self, size, returnTable=False):
        idx = np.random.choice(len(self.memory), min(len(self.memory), size), replace=False)
        if returnTable:
            return self.memory[idx]
        state, rewardList, newState, done = self.rowsToArgs(self.memory[idx])
        return state, rewardList, newState, done



#
# class NStepQMemory(Memory):
#
#     def __init__(self, stateSpace, nSteps, maxLength=np.inf, dataAugmentors=[]):
#         super(NStepQMemory, self).__init__()
#         self.nSteps = nSteps
#         self.stateSpace = stateSpace
#         self.dataAugmentors = dataAugmentors
#         self.maxLength = maxLength
#
#         self.nColumns  = 2 * self.stateSpace # state + newState
#         self.nColumns += self.nSteps # rewards
#         self.nColumns += 2 # done + action
#
#         self.memory = np.zeros((0, self.nColumns))
#
#
#     def argsToRow(self, state:np.array, action:int, rewardList:list, newState:np.array, done:int):
#         row = np.expand_dims(np.copy(state), axis=0)
#         row = np.concatenate([row, np.array(action).reshape(1, -1)], axis=1)
#         for r in rewardList:
#             row = np.concatenate([row, np.array(r).reshape(1, -1)], axis=1)
#         row = np.concatenate([row, np.expand_dims(newState, axis=0)], axis=1)
#         row = np.concatenate([row, np.array(done).reshape(1, -1)], axis=1)
#         return row
#
#
#     def rowsToArgs(self, chunk):
#         state = chunk[:, :self.stateSpace]
#         action = chunk[:, self.stateSpace]
#
#         rewardList = []
#         for i in range(self.nSteps):
#             offset = self.stateSpace + 1 + i
#             reward = chunk[:, offset]
#             rewardList.append(reward)
#
#         offset = self.stateSpace + 1 + self.nSteps
#         newState = chunk[:, offset : offset+self.stateSpace ]
#
#         done = chunk[:, -1]
#         return state, action, rewardList, newState, done
#
#
#     def addMemory(self, state:np.array, action:int, rewardList:list, newState:np.array, done:int):
#         row = self.argsToRow(state, action, rewardList, newState, done)
#         self._append(row)
#         # for da in self.dataAugmentors:
#         #     s, sP = da.apply([state, newState])
#         #     self._append(s, action, reward, sP, done)
#
#
#     def sampleMemory(self, size):
#         idx = np.random.choice(len(self.memory), min(len(self.memory), size), replace=False)
#         state, action, rewardList, newState, done = self.rowsToArgs(self.memory[idx])
#         return state, action, rewardList, newState, done