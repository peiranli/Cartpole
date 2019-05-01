import math
import random
import gym
import os
import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

from common.multiprocessing_env import SubprocVecEnv

num_envs = 16
env_name = "Pendulum-v0"

def make_env():
    def make():
        env = gym.make(env_name)
        return env

    return make

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
ACTION_MAX = env.action_space.high[0]
SAMPLE_NUMS = 100

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        mu = self.actor(x)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist

# init actor network
actor_network = Actor(STATE_DIM,ACTION_DIM,256,0)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.001)
eps = np.finfo(np.float32).eps.item()

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = Variable(torch.Tensor(state))
        dist = actor_network(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy())
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def roll_out(sample_nums):
    observation = envs.reset()
    states = []
    actions = []
    rewards = []
    episode_reward = 0
    entropy = 0
    for _ in range(sample_nums):
        #env.render()
        state = np.float32(observation)
        states.append(state)
        dist = actor_network(Variable(torch.Tensor(state)))
        action = dist.sample()
        entropy += dist.entropy().mean()
        action = action.cpu().numpy()
        new_observation,reward,done,_ = envs.step(action)
        episode_reward += reward
        actions.append(action)
        rewards.append(reward)
        observation = new_observation
    #print ('REWARDS :- ', episode_reward)
    return states,actions,rewards,entropy

def discount_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_network(states, actions, rewards, entropy):
    states_var = Variable(FloatTensor(states).view(-1,STATE_DIM))
    actions_var = Variable(FloatTensor(actions).view(-1,ACTION_DIM))
    # train actor network
    actor_network_optim.zero_grad()
    dist = actor_network(states_var)
    log_probs = dist.log_prob(actions_var)
    # calculate qs
    rewards = Variable(torch.Tensor(discount_reward(rewards,0.99))).view(-1,1)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    actor_network_loss = - torch.mean(torch.sum(log_probs * rewards)) - 0.001*entropy
    #print("loss",actor_network_loss)
    actor_network_loss.backward()
    actor_network_optim.step()


MAX_EPISODES = 5000
_ep = 0
early_stop = False
test_rewards = []
threshold_reward = -200

while _ep < MAX_EPISODES and not early_stop:
    observation = env.reset()
    #print ('EPISODE :- ', _ep)
    states,actions,rewards,entropy = roll_out(SAMPLE_NUMS)
    update_network(states,actions,rewards,entropy)

    if _ep % 100 == 0:
        test_reward = np.mean([test_env() for _ in range(10)])
        test_rewards.append(test_reward)
        print ('EPISODE :- ', _ep)
        print("TEST REWARD :- ", test_reward)
        if test_reward > threshold_reward: early_stop = True
    _ep += 1

test_env(True)

envs.close()
env.close()