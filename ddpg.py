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

parser = argparse.ArgumentParser(description='ddpg')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 50)')
args = parser.parse_args()



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

class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        
        return action

    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high
        
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        
        return action

#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)
    

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.ReLU(),
            self.l3,
            nn.Tanh()
        )
        return model(state)

    def get_action(self, state):
        action = self.forward(state)
        return action.detach().numpy()[0, 0]

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim+action_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        out = self.l1(state)
        out = F.relu(out)
        out = self.l2(torch.cat([out, action], 1))
        out = F.relu(out)
        out = self.l3(out)
        return out


env = NormalizedActions(gym.make("Pendulum-v0"))
ou_noise = OUNoise(env.action_space)
env.seed(args.seed)
torch.manual_seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim = 256
actor = Actor(state_dim, hidden_dim,  action_dim)
actor_target = Actor(state_dim, hidden_dim, action_dim)
actor_target.load_state_dict(actor.state_dict())

critic = Critic(state_dim, hidden_dim, action_dim)
critic_target = Critic(state_dim, hidden_dim, action_dim)
critic_target.load_state_dict(critic.state_dict())
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-2)
critic_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

memory = ReplayMemory(10000)
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
soft_tau = 1e-2
min_value = -np.inf
max_value = np.inf
steps_done = 0

def update_net():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*transitions)
    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))
    batch_done = Variable(torch.cat(batch_done))

    # estimate the target q with actor_target network and critic_target network
    next_action = actor_target(batch_next_state)
    max_next_q_values = critic_target(batch_next_state, next_action).detach().squeeze(1)
    expected_q_values = batch_reward + (1.0 - batch_done) * args.gamma * max_next_q_values
    expected_q_values = torch.clamp(expected_q_values, min_value, max_value)

    # update critic network
    critic_optimizer.zero_grad()
    current_q_values = critic(batch_state, batch_action)
    critic_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
    critic_loss.backward()
    critic_optimizer.step()

    # update actor network
    actor_optimizer.zero_grad()
    # accurate action prediction
    current_action = actor(batch_state)
    actor_loss = -critic(batch_state,current_action)
    actor_loss = actor_loss.mean()
    actor_loss.backward()
    actor_optimizer.step()

def main():
    running_reward = 10
    print("reward threshold", env.spec.reward_threshold)
    for i_episode in count(1):
        # Initialize the environment and state
        state = env.reset()
        for t in range(10000):
            # Select and perform an action
            action = actor.get_action(FloatTensor([state]))
            action = ou_noise.get_action(action, t)
            next_state, reward, done, _ = env.step(action)
        
            if done:
                reward = -1
            # Store the transition in memory
            transition = (FloatTensor([state]), FloatTensor([action]), FloatTensor([next_state]), FloatTensor([reward]), FloatTensor([done]))
            memory.push(transition)
            state = next_state
            # Perform one step of the optimization (on the target network)
            update_net()
            if done:
                break
        
        running_reward = running_reward * 0.99 + t * 0.01
        
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t+1, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t+1))
            break
        # Soft update the target network
        if i_episode % TARGET_UPDATE == 0:
            for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)

            for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - soft_tau) + param.data * soft_tau)
    
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            action = actor.get_action(FloatTensor([state]))
            state, reward, done, _ = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()


if __name__ == '__main__':
    main()

    
