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
eps = np.finfo(np.float32).eps.item()


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        return model(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2
        )
        return model(state)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
actor = Actor(state_dim, 128, action_dim)
actor_target = Actor(state_dim, 128, action_dim)
actor_target.load_state_dict(actor.state_dict())
critic = Critic(state_dim, 128, action_dim)
critic_target = Critic(state_dim, 128, action_dim)
critic_target.load_state_dict(critic.state_dict())
actor_optimizer = optim.Adam(actor.parameters(), lr=1e-2)
critic_optimizer = optim.Adam(actor.parameters(), lr=1e-3)

memory = ReplayMemory(10000)
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_rate = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_rate:
        with torch.no_grad():
            return actor(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(2)]])

def update_net():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
    non_final_next_states = []
    non_final_mask = []
    for i in range(len(batch_next_state)):
        if batch_next_state[i] is not None:
            non_final_next_states.append(batch_next_state[i])
            non_final_mask.append(i)
    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_reward = (batch_reward - batch_reward.mean()) / (batch_reward.std() + eps)
    batch_next_state = Variable(torch.cat(batch_next_state))
    non_final_next_states = Variable(torch.cat(non_final_next_states))

    # estimate the target q with actor_target network and critic_target network
    next_action = actor_target(batch_next_state).detach().max(1)[1]
    next_action = Variable(next_action)
    max_next_q_values = torch.zeros(BATCH_SIZE, device="cpu")
    max_next_q_values[non_final_mask] = critic_target(non_final_next_states).gather(1, next_action[non_final_mask].unsqueeze(1)).detach().squeeze(1)
    expected_q_values = batch_reward + args.gamma * max_next_q_values

    # update critic network
    critic_optimizer.zero_grad()
    current_q_values = critic(batch_state).gather(1, batch_action)
    critic_loss = F.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
    critic_loss.backward()
    critic_optimizer.step()

    # update actor network
    actor_optimizer.zero_grad()
    # accurate action prediction
    current_action = actor(batch_state).detach().max(1)[1]
    actor_loss = -critic(batch_state).gather(1, current_action.unsqueeze(1))
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
            action = select_action(FloatTensor([state]))
            next_state, reward, done, _ = env.step(action.item())
        
            if done:
                reward = -1

            # Store the transition in memory
            transition = (FloatTensor([state]), action, FloatTensor([next_state]), FloatTensor([reward]))
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
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())
    
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            pred = actor(FloatTensor([state]))
            values = pred.detach().numpy()
            action = np.argmax(values)
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()


if __name__ == '__main__':
    main()

    
