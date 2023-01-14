from turtle import distance
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicy
import numpy as np
from sklearn.model_selection import train_test_split
from cs285.infrastructure import utils
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, k, obs, acs):
        super(NeuralNetwork, self).__init__()
        self.obs = obs
        self.acs = acs
        self.k = k
        
        # Sequential model
        hidden_layer_features = int(1.66*k)
        # inputs layer(s)
        self.input = nn.Linear(in_features=k, out_features=hidden_layer_features)
        # 2 hidden layer
        self.layer_2 = nn.Linear(in_features=hidden_layer_features, out_features=hidden_layer_features)
        self.layer_3 = nn.Linear(in_features=hidden_layer_features, out_features=hidden_layer_features)
        # output layer(s)
        self.output = nn.Linear(in_features=hidden_layer_features, out_features=k)

    def forward(self, x):
        cos_dist = 1 - utils.cos_sim(self.obs, x)
        if x in self.obs:
            min_indexes = np.argpartition(cos_dist, self.k+1)[1:self.k+1]
        else:
            min_indexes = np.argpartition(cos_dist, self.k)[:self.k]
        k_actions = self.acs[min_indexes]
        k_cos_dist = torch.from_numpy(cos_dist[min_indexes]).float()#.float()

        x = nn.functional.relu(self.input(k_cos_dist))
        x = nn.functional.relu(self.layer_2(x))
        x = nn.functional.relu(self.layer_3(x))
        weights = nn.functional.softmax(self.output(x))
        np_weights = weights.detach().numpy()
        weighted = []

        for i in range(np_weights.shape[-1]):
            weighted.append(np_weights[i]*k_actions[i])

        x = np.array([np.sum(weighted, axis = 0)])

        return torch.from_numpy(x)

class KNNAgent():
    def __init__(self, env, agent_params):
        super(KNNAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # Actor policy
        self.actor = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
        )

        # prediction params
        self.k = self.agent_params['k']
        self.pred_type = self.agent_params['pred_type']

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])
    def set_k(self, k_val):
        self.k = k_val
    def train(self):
        ''' train a KNN algorithm to identify the optimal label of an observation given expert data '''
        # Since KNN doesn't "learn" much, the only step here is to get the training+testing data
        #assert(len(self.replay_buffer.obs)==len(self.replay_buffer.acs))
        # train-test split
        self.obs_train, self.acs_train = self.replay_buffer.obs, self.replay_buffer.acs
        #print(self.obs_train.shape, self.obs_train[0].shape, self.acs_train.shape, self.acs_train[0].shape)

        if self.pred_type==5:

            model = NeuralNetwork(self.k, self.obs_train, self.acs_train)
            #print(model.parameters)

            lr = 0.1
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            num_epochs = 100
            loss_vals = []

            for _ in range(num_epochs):
                for X, y in zip(torch.Tensor(self.obs_train), torch.Tensor(self.acs_train)):
                    optimizer.zero_grad()
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    loss = torch.autograd.Variable(loss, requires_grad = True)
                    loss_vals.append(loss.item())
                    loss.backward()
                    optimizer.step()
            
            self.model = model
            print('Training Complete')

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def get_action(self, new_obs):
        k = self.k
        pred_type = self.pred_type
        # L1-distance computation
        if pred_type == 1:
            distances = utils.getL1Distance(self.obs_train, new_obs[0])
            min_indexes = np.argpartition(distances, k)
            # average of top-k labels (actions)
            acs_pred = np.array([np.average(self.acs_train[min_indexes[:k]], axis=0)])
        # L2-distance computation
        elif pred_type == 2:
            distances = utils.getL2Distance(self.obs_train, new_obs[0])
            min_indexes = np.argpartition(distances, k)
            # average of top-k labels (actions)
            acs_pred = np.array([np.average(self.acs_train[min_indexes[:k]], axis=0)])
        elif pred_type == 3:
            distances = utils.getEuclideanDistance(self.obs_train, new_obs[0])
            min_indexes = np.argpartition(distances, k)[:k]
            # multiply top-k labels by softmax probability distribution
            top_k_distances = distances[min_indexes]
            top_k_acs = self.acs_train[min_indexes]
            weighted = np.zeros(shape = top_k_acs.shape)
            prob_dist = utils.softmax_inv(top_k_distances)
            # Multiply values of probability distribution to action values
            for x in range(len(prob_dist)):
                weighted[x] = prob_dist[x]*top_k_acs[x]
            # Take sum of weighted actions
            acs_pred = np.array([np.sum(weighted, axis = 0)])
        elif pred_type==4:
            distances = utils.getEuclideanDistance(self.obs_train, new_obs[0])
            min_indexes = np.argpartition(distances, k)[:k]
            # multiply top-k labels by softmax probability distribution
            top_k_distances = distances[min_indexes]
            top_k_acs = self.acs_train[min_indexes]
            weighted = np.zeros(shape = top_k_acs.shape)
            weights = utils.logistic(top_k_distances)
            # Multiply values of probability distribution to action values
            for x in range(len(weights)):
                weighted[x] = weights[x]*top_k_acs[x]
            # Take sum of weighted actions
            acs_pred = np.array([np.sum(weighted, axis = 0)])
        elif pred_type==5:
            acs_pred = self.model(new_obs).numpy()

        return acs_pred
