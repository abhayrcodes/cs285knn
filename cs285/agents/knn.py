from turtle import distance
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.policies.MLP_policy import MLPPolicy
import numpy as np
from sklearn.model_selection import train_test_split
from cs285.infrastructure import utils

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
        self.distance_type = self.agent_params['distance_type']

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self):
        ''' train a KNN algorithm to identify the optimal label of an observation given expert data '''
        # Since KNN doesn't "learn" much, the only step here is to get the training+testing data
        assert(len(self.replay_buffer.obs)==len(self.replay_buffer.acs))
        obs, acs = self.replay_buffer.obs, self.replay_buffer.acs
        # train-test split
        self.obs_train, self.obs_test, self.acs_train, self.acs_test = train_test_split(obs, acs, test_size=0.3, random_state=42)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def get_action(self, new_obs):
        k = self.k
        distance_type = self.distance_type
        # L1-distance computation
        if distance_type == 1 or distance_type!=2:
            distances = utils.getL1Distance(self.obs_train, new_obs)
        # L2-distance computation
        elif distance_type == 2:
            distances = utils.getL2Distance(self.obs_train, new_obs)
        min_indexes = np.argpartition(distances, k)

        # average of top-k labels (actions)
        acs_pred = np.average(self.acs_train[min_indexes[:k]], axis=0)

        return acs_pred