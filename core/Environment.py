import gym
import torch
import numpy as np
from PoolManagement import reset_al_pool, add_datapoint_to_pool
from sklearn.metrics import f1_score
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class MockALGame(gym.Env):

    def __init__(self, config, noise_level=1, reward_noise=1, *args, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.noise_level = noise_level
        self.reward_noise = reward_noise
        self.config = config
        self.budget = config.BUDGET
        self.sample_size = config.SAMPLE_SIZE
        self.max_reward = config.MAX_REWARD
        self.reset()
        self._set_state_shape()
        self.action_space = gym.spaces.Discrete(config.SAMPLE_SIZE)
        self.spec = gym.envs.registration.EnvSpec("MockAl-v0", reward_threshold=10)


    def _set_state_shape(self):
        state = self.create_state()
        self.state_space = state.shape[1]
        self.observation_space = gym.spaces.Box(0, np.inf, shape=(state.shape[1],))


    def reset(self, *args, **kwargs):
        self.added_images = 0
        self.current_test_f1 = 0.4
        return self.create_state()


    def create_state(self):
        qualities = np.random.rand(self.sample_size) * 0.9
        bvssb_noise = np.random.normal(0, 0.2, size=self.sample_size)
        entr_noise = np.random.normal(0, 1.0, size=self.sample_size)
        sample = []
        for i in range(len(qualities)):
            dp = [self.current_test_f1]
            dp.append(qualities[i] +
                      self.noise_level * bvssb_noise[i]) # BvsSB
            dp.append(2 + (qualities[i] - 0.6)*2 +
                      self.noise_level * entr_noise[i])  # entropy
            # dp.append(qualities[i])
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
        if self.reward_noise:
            noise = np.random.normal(0, 0.5 * self.max_reward)
            reward += noise
        self.current_test_f1 += reward
        done = self.added_images >= self.budget
        return self.create_state(), reward, done, {}


    def render(self, mode="human"):
        pass


######################################################
######################################################
class ALGame(gym.Env):

    def __init__(self, dataset, classifier_function, config, sample_size_in_state=False):
        assert all([isinstance(d, torch.Tensor) for d in dataset])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x_test = dataset[2]
        self.y_test = dataset[3]
        self.y_test_cpu = self.y_test.clone().cpu()
        self.dataset = dataset
        self.n_classes = self.y_test.shape[1]

        self.config = config
        self.budget = config.BUDGET
        self.game_length = config.BUDGET
        self.reward_scaling = config.REWARD_SCALE
        self.rewardShaping = config.REWARD_SHAPING
        self.from_scratch = config.CLASS_FROM_SCRATCH

        self.classifier_function = classifier_function
        self.classifier = classifier_function(inputShape=self.x_test.shape[1:],
                                              numClasses=self.y_test.shape[1])
        self.classifier = self.classifier.to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.loss = nn.CrossEntropyLoss()

        self.current_test_f1 = 0
        self.reset()
        self.action_space = gym.spaces.Discrete(config.SAMPLE_SIZE)
        self._set_state_shape(sample_size_in_state)
        self.spec = gym.envs.registration.EnvSpec("RlAl-v0", reward_threshold=10)


    def _set_state_shape(self, sample_size_in_state):
        state = self.create_state()
        if sample_size_in_state:
            self.state_space = np.multiply(*state.shape)
        else:
            self.state_space = state.shape[1]
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=[state.shape[1],])


    def _sample_ids(self, num):
        return np.random.choice(len(self.x_unlabeled), num)


    def reset(self, *args, **kwargs):
        with torch.no_grad():
            self.n_interactions = 0
            self.added_images = 0

            del self.classifier
            self.classifier = self.classifier_function(inputShape=self.x_test.shape[1:],
                                                       numClasses=self.y_test.shape[1])
            self.classifier.to(self.device)
            self.initial_weights = self.classifier.state_dict()
            self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

            self.x_labeled, self.y_labeled, \
            self.x_unlabeled, self.y_unlabeled, self.per_class_intances = reset_al_pool(self.dataset,
                                                                                        self.config.INIT_POINTS_PER_CLASS)
        self.fit_classifier() # sets self.currentTestF1
        self.initial_f1 = self.current_test_f1

        self.state_ids = self._sample_ids(self.config.SAMPLE_SIZE)
        return self.create_state()


    def create_state(self):
        with torch.no_grad():
            sample_x = self.x_unlabeled[self.state_ids]
            interal_features = self._get_internal_features(sample_x)
            sample_features = self._get_sample_features(sample_x)
            state = torch.cat([sample_features, interal_features], dim=1)
        return state.cpu()


    def _get_internal_features(self, x):
        with torch.no_grad():
            f1 = torch.repeat_interleave(torch.Tensor([self.current_test_f1]), len(x))
            f1 = f1.unsqueeze(1).to(self.device)
            progress = torch.repeat_interleave(torch.Tensor([self.added_images / float(self.budget)]), len(x))
            progress = progress.unsqueeze(1).to(self.device)

            mean_labeled = torch.mean(self.x_labeled, dim=0)
            mean_unlabeled = torch.mean(self.x_unlabeled, dim=0)
            mean_labeled = mean_labeled.unsqueeze(0).repeat(len(x), 1)
            mean_unlabeled = mean_unlabeled.unsqueeze(0).repeat(len(x), 1)

            # compute difference of the labeled pool and the sample
            # didn't work last time
            # diff_labeled = torch.abs(sample_x - mean_labeled)
            # diff_unlabeled = torch.abs(sample_x - mean_unlabeled)
            # state = torch.cat([alFeatures, diff_labeled, diff_unlabeled], dim=1)

        return torch.cat([f1, progress, mean_labeled, mean_unlabeled], dim=1)


    def _get_sample_features(self, x):
        eps = 1e-7
        # prediction metrics
        with torch.no_grad():
            pred = self.classifier(x).detach()
            two_highest, _ = pred.topk(2, dim=1)

            entropy = -torch.mean(pred * torch.log(eps + pred) + (1+eps-pred) * torch.log(1+eps-pred), dim=1)
            bVsSB = 1 - (two_highest[:, -2] - two_highest[:, -1])
            hist_list = [torch.histc(p, bins=10, min=0, max=1) for p in pred]
            hist = torch.stack(hist_list, dim=0) / self.n_classes

            state = torch.cat([
                bVsSB.unsqueeze(1),
                entropy.unsqueeze(1),
                hist
            ], dim=1)
        return state


    def step(self, action):
        with torch.no_grad():
            self.n_interactions += 1
            self.added_images += 1
            datapoint_id = self.state_ids[action]
            self.x_labeled, self.y_labeled, \
            self.x_unlabeled, self.y_unlabeled, perClassIntances = add_datapoint_to_pool(self.x_labeled, self.y_labeled,
                                                                                         self.x_unlabeled, self.y_unlabeled,
                                                                                         self.per_class_intances, datapoint_id)
        reward = self.fit_classifier()
        self.state_ids = self._sample_ids(self.config.SAMPLE_SIZE)
        statePrime = self.create_state()

        done = self.checkDone()
        return statePrime, reward, done, {}


    def fit_classifier(self, epochs=50, batch_size=128):
        if self.from_scratch:
            self.classifier.load_state_dict(self.initial_weights)

        #batch_size = min(batch_size, int(len(self.xLabeled)/5))
        train_dataloader = DataLoader(TensorDataset(self.x_labeled, self.y_labeled), batch_size=batch_size)
        test_dataloader = DataLoader(TensorDataset(self.x_test, self.y_test), batch_size=100)

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
                loss_sum = 0.0
                total = 0.0
                correct = 0.0
                for batch_x, batch_y in test_dataloader:
                    yHat = self.classifier(batch_x)
                    predicted = torch.argmax(yHat, dim=1)
                    # _, predicted = torch.max(yHat.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
                    class_loss = self.loss(yHat, torch.argmax(batch_y.long(), dim=1))
                    loss_sum += class_loss.detach().cpu().numpy()
                if loss_sum >= lastLoss:
                    #print(f"labeled {len(self.xLabeled)}: stopped after {e} epochs")
                    break
                lastLoss = loss_sum
        accuracy = correct / total
        # one_hot_y_hat = np.eye(10, dtype='uint8')[torch.argmax(self.classifier(self.x_test), dim=1).cpu()]
        # new_test_f1 = f1_score(self.y_test_cpu, one_hot_y_hat, average="samples")
        self.current_test_loss = loss_sum

        if self.rewardShaping:
            reward = (accuracy - self.current_test_f1) * self.reward_scaling
        else:
            raise NotImplementedError()
        self.current_test_f1 = accuracy
        return reward


    def checkDone(self):
        done = self.n_interactions >= self.game_length # max interactions
        if self.added_images >= self.budget:
            done = True # budget exhausted
        return done




######################################################
######################################################
class DuelingALGame(ALGame):
    def __init__(self, dataset, classifier_function, config):
        super().__init__(dataset, classifier_function, config)


    def _set_state_shape(self):
        cntx, state = self.create_state()
        self.stateSpace = (cntx.shape[1], state.shape[1])

    def create_state(self):
        classFeatures = self._get_sample_features(self.x_unlabeled[self.state_ids])
        f1 = classFeatures[0, 0]
        alFeatures = classFeatures[:, 1:]
        cntxFeatures = self.getPoolInfo()
        cntxFeatures = torch.cat([f1.unsqueeze(0), cntxFeatures]).unsqueeze(0)
        return cntxFeatures, alFeatures



######################################################
######################################################
class PALGame(ALGame):

    def __init__(self, dataset, classifier_function, config):
        super().__init__(dataset, classifier_function, config)
        self.actionSpace = 2


    def _sample_ids(self, num=1):
        return np.random.randint(len(self.xUnlabeled))


    def create_state(self):
        alFeatures = self._get_sample_features(self.xUnlabeled[self.stateIds].unsqueeze(0))
        return alFeatures
        # alFeatures = torch.from_numpy(alFeatures).float()
        # poolFeatures = self.getPoolInfo()
        # poolFeatures = poolFeatures.unsqueeze(0).repeat(len(alFeatures), 1)
        # state = torch.cat([alFeatures, poolFeatures], dim=1)
        # return state


    def step(self, action):
        self.n_interactions += 1
        if action == 1:
            self.added_images += 1
            ret = add_datapoint_to_pool(self.x_labeled, self.y_labeled,
                                        self.x_unlabeled, self.y_unlabeled,
                                        self.per_class_intances, self.state_ids)
            self.xLabeled, self.yLabeled, self.xUnlabeled, self.yUnlabeled, perClassIntances = ret
            reward = self.fit_classifier()
        else:
            reward = 0

        self.stateIds = self._sample_ids()
        statePrime = self.create_state()

        done = self.checkDone()
        return statePrime, reward, done, {}
