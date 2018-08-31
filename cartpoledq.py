import argparse
import gym
import numpy as np
import random
import math
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parser = argparse.ArgumentParser(description='Cartpole deep q learning')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
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

q_net = DQN()
optimizer = optim.Adam(q_net.parameters(), lr=1e-2)
memory = ReplayMemory(10000)
BATCH_SIZE = 128
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    state = torch.from_numpy(state).float().unsqueeze(0)
    eps_rate = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_rate:
        with torch.no_grad():
            return q_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device="cpu", dtype=torch.long)

def update_net():
    if len(memory) < BATCH_SIZE:
       return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device="cpu", dtype=torch.uint8)
    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final_next_states = torch.cat(non_final_next_states)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = q_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device="cpu")
    next_state_values[non_final_mask] = q_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in q_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    running_reward = 10
    print("reward threshold", env.spec.reward_threshold)
    for i_episode in count(1):
        # Initialize the environment and state
        state = env.reset()
        for t in range(10000):
            # Select and perform an action
            action = select_action(state)
            state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device="cpu")
            if not done:
                next_state = state
            else:
                next_state = None

            # Store the transition in memory
            transition = (state, action, next_state, reward)
            memory.push(transition)

            running_reward = running_reward * 0.99 + t * 0.01
            # Perform one step of the optimization (on the target network)
            update_net()
            if done:
                break

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
    
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            state = torch.from_numpy(state).float().unsqueeze(0)
            pred = q_net(state)
            values = pred.detach().numpy()
            action = np.argmax(values)
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()


if __name__ == '__main__':
    main()