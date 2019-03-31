import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import count
import numpy as np
import math
import random
import os
import gym

## Hyper Parameters
STATE_DIM = 4
ACTION_DIM = 2
SAMPLE_NUMS = 1000

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        return out

# init actor network
actor_network = ActorNetwork(STATE_DIM,64,ACTION_DIM)
actor_network_optim = torch.optim.Adam(actor_network.parameters(),lr = 0.001)
eps = np.finfo(np.float32).eps.item()

env = gym.make("CartPole-v0")

def roll_out():
    state = env.reset()
    states = []
    actions = []
    rewards = []

    for step in range(SAMPLE_NUMS):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = env.step(action)
        actions.append(one_hot_action)
        rewards.append(reward)
        state = next_state
        if done:
            break

    return states,actions,rewards,step

def update_network(states, actions, rewards):
        actions_var = Variable(FloatTensor(actions).view(-1,ACTION_DIM))
        states_var = Variable(FloatTensor(states).view(-1,STATE_DIM))
        # train actor network
        actor_network_optim.zero_grad()
        log_softmax_actions = actor_network(states_var)
        # calculate qs
        rewards = Variable(torch.Tensor(discount_reward(rewards,0.99)))
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* rewards)
        actor_network_loss.backward()
        actor_network_optim.step()


def discount_reward(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def main():
    running_reward = 10
    print("reward threshold", env.spec.reward_threshold)

    for i_episode in count(1):
        states,actions,rewards,steps = roll_out()
        running_reward = running_reward * 0.99 + steps * 0.01
        update_network(states,actions,rewards)
        
        if i_episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, steps+1, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, steps+1))
            break
    # test
    for i_episode in range(10):
        state = env.reset()
        for t in range(1000):
            env.render()
            pred = actor_network(FloatTensor([state]))
            values = pred.detach().numpy()
            action = np.argmax(values)
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

    env.close()

if __name__ == '__main__':
    main()