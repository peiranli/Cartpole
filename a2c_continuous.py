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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        return dist, value

model = ActorCritic(STATE_DIM, ACTION_DIM, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = Variable(torch.Tensor(state))
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy())
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def roll_out(sample_nums):
    observation = envs.reset()
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    is_done = False
    final_r = 0
    episode_reward = 0
    entropy = 0
    for step in range(sample_nums):
        #env.render()
        state = np.float32(observation)
        states.append(state)
        dist, value = model(Variable(torch.Tensor(state)))
        #dist = actor_network(Variable(torch.Tensor(state)))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        action = action.cpu().numpy()
        new_observation,reward,done,_ = envs.step(action)
        episode_reward += reward
        log_probs.append(log_prob)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        new_state = np.float32(new_observation)
        observation = new_observation
    #print ('REWARDS :- ', episode_reward)
    return states,actions,rewards,values,step,final_r,log_probs,entropy

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_network(states, actions, rewards, values, final_r, log_probs, entropy):
        log_probs = torch.cat(log_probs)
        vs = torch.cat(values)
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99, final_r))).view(-1,1)

        advantages = qs - vs
        actor_loss = - torch.mean(log_probs * advantages.detach())

        target_values = qs
        criterion = nn.MSELoss()
        critic_loss = criterion(vs,target_values)

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        #print("loss ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

MAX_EPISODES = 25000
_ep = 0
early_stop = False
test_rewards = []
threshold_reward = -200

while _ep < MAX_EPISODES and not early_stop:
    observation = env.reset()
    states,actions,rewards,values,steps,final_r,log_probs, entropy = roll_out(SAMPLE_NUMS)
    update_network(states,actions,rewards,values,final_r,log_probs, entropy)
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
