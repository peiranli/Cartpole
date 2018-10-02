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

parser = argparse.ArgumentParser(description='a2c')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 50)')
args = parser.parse_args()


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.Softmax(1)
        )
        return model(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.values = []

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3
        )
        return model(state)

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
eps = np.finfo(np.float32).eps.item()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 256

actor = Actor(state_dim, hidden_dim, action_dim)
critic = Critic(state_dim, hidden_dim)

actor_optimizer = optim.Adam(actor.parameters(), lr=1e-2)
critic_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

def select_action(state):
    probs = actor(state)
    m = Categorical(probs)
    action = m.sample()
    actor.saved_log_probs.append(m.log_prob(action))
    return action.item()

def compute_returns(rewards, final_reward):
    R = final_reward
    res = []
    for t in reversed(range(len(rewards))):
        R = R * args.gamma + rewards[t]
        res.insert(0, R)
    return res

def update_net(final_reward):
    actor_loss = []
    rewards = compute_returns(actor.rewards, final_reward)
    rewards = torch.cat(rewards).detach()
    for log_prob, reward, value in zip(actor.saved_log_probs, rewards, critic.values):
        actor_loss.append(-log_prob * (reward - value))
    values = torch.cat(critic.values)
    actor_optimizer.zero_grad()
    actor_loss = torch.cat(actor_loss).sum()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()

    advantages = rewards - values
    critic_loss = advantages.pow(2).mean()
    critic_loss.backward()
    critic_optimizer.step()

    del actor.rewards[:]
    del actor.saved_log_probs[:]
    del critic.values[:]


def main():
    running_reward = 10
    print("reward threshold", env.spec.reward_threshold)
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(FloatTensor([state]))
            value = critic(FloatTensor([state]))
            state, reward, done, _ = env.step(action)
            actor.rewards.append(reward)
            critic.values.append(value)
            if done:
                break
        final_reward = critic()
        running_reward = running_reward * 0.99 + t * 0.01
        update_net(final_reward)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t+1, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t+1))
            break
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            pred = actor(FloatTensor([state]))
            values = pred.detach().numpy()
            action = np.argmax(values)
            state, reward, done, _ = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

if __name__ == '__main__':
    main()