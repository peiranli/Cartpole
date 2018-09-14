import argparse
import gym
import numpy as np
import random
import math
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='Cartpole ddpg')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 50)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, self.action_space)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(x)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, self.action_space)

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2
        )
        return model(x)