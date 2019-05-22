import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import numpy as np
import math
import random
import os
import gym

# init a task generator for data fetching
env = gym.make("CartPole-v0")

## Hyper Parameters
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n
SAMPLE_NUMS = 300
TARGET_UPDATE = 10
CLIP_PARAM=0.2

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ActorNetwork(nn.Module):

    def __init__(self,state_dim,action_dim,hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_dim)
        self.apply(init_weights)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out), dim=1)
        dist = Categorical(out)
        return dist

class ValueNetwork(nn.Module):

    def __init__(self,state_dim,action_dim,hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_dim)
        self.apply(init_weights)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

policy_net = ActorNetwork(STATE_DIM,ACTION_DIM,64)
target_policy_net = ActorNetwork(STATE_DIM,ACTION_DIM,64)
target_policy_net.load_state_dict(policy_net.state_dict())
target_policy_net.eval()

value_net = ValueNetwork(STATE_DIM,1,64)
target_value_net = ValueNetwork(STATE_DIM,1,64)
target_value_net.load_state_dict(value_net.state_dict())
target_value_net.eval()


policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        dist = policy_net(FloatTensor([state]))
        action = dist.sample()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def roll_out(sample_nums):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    for step in range(sample_nums):
        states.append(state)
        dist = policy_net(Variable(torch.Tensor([state])))
        action = dist.sample()
        actions.append(action)
        action = action.cpu().numpy()
        next_state,reward,done,_ = env.step(action[0])
        rewards.append(reward)
        state = next_state
        if done:
            is_done = True
            break
    if not is_done:
        final_r = value_net(Variable(torch.Tensor([state])))
    return states,actions,rewards,step,final_r

def discount_reward(r, gamma, final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def update_network(states, actions, rewards, final_r):
        actions_var = torch.cat(actions)
        states_var = Variable(FloatTensor(states).view(-1,STATE_DIM))
        # train actor network
        policy_optimizer.zero_grad()
        
        vs = target_value_net(states_var).detach()
        # calculate qs
        qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r)))
        advantages = qs - vs

        dist = policy_net(states_var)
        log_probs = dist.log_prob(actions_var)
        old_dist = target_policy_net(states_var)
        old_log_probs = old_dist.log_prob(actions_var)

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advantages

        actor_network_loss = - torch.mean(torch.min(surr1, surr2))
        actor_network_loss.backward()
        policy_optimizer.step()

        # train value network
        value_optimizer.zero_grad()
        target_values = qs
        values = value_net(states_var)
        criterion = nn.MSELoss()
        value_network_loss = criterion(values,target_values.unsqueeze(1))
        value_network_loss.backward()
        value_optimizer.step()


def main():
    running_reward = 10
    i_episode = 0
    MAX_EPISODES = 5000
    early_stop = False
    test_rewards = []
    threshold_reward = env.spec.reward_threshold
    while i_episode < MAX_EPISODES and not early_stop:
        states,actions,rewards,steps,final_r = roll_out(SAMPLE_NUMS)
        running_reward = running_reward * 0.99 + steps * 0.01
        update_network(states,actions,rewards,final_r)
        
        if i_episode % 50 == 0:
            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            print ('EPISODE :- ', i_episode)
            print("TEST REWARD :- ", test_reward)
            if test_reward > threshold_reward: early_stop = True
        # Update the target networks
        if i_episode % TARGET_UPDATE == 0:
            target_value_net.load_state_dict(value_net.state_dict())
            target_policy_net.load_state_dict(policy_net.state_dict())
        i_episode += 1
    test_env(True)

    env.close()

if __name__ == '__main__':
    main()