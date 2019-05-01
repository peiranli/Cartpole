from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import gym
import numpy as np
import random
import math
from itertools import count

BATCH_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor 
ByteTensor = torch.ByteTensor 
Tensor = FloatTensor


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
	

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:

	def __init__(self, action_dim, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return self.X

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of input action (int)
        :return:
        """
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs2 = nn.Linear(256,128)

        self.fca1 = nn.Linear(action_dim,128)

        self.fc2 = nn.Linear(256,128)

        self.fc3 = nn.Linear(128,1)
        self.apply(init_weights)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        output = torch.cat((s2,a1),dim=1)

        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, action_lim):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim,256)

        self.fc2 = nn.Linear(256,128)

        self.fc3 = nn.Linear(128,64)

        self.fc4 = nn.Linear(64,action_dim)
        self.apply(init_weights)

    def forward(self, state):
        """
        returns policy function Pi(s) obtained from actor network
        this function is a gaussian prob distribution for all actions
        with mean lying in (-1,1) and sigma lying in (0,1)
        The sampled action can , then later be rescaled
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        action = F.tanh(self.fc4(output))

        action = action * float(self.action_lim)

        return action

class Trainer:

    def __init__(self, state_dim, action_dim, action_lim, memory):
        """
        :param state_dim: Dimensions of state (int)
        :param action_dim: Dimension of action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :param memory: replay memory buffer object
        :return:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.memory = memory
        self.iter = 0
        self.noise = OrnsteinUhlenbeckActionNoise(self.action_dim)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_lim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):

        action = self.target_actor.forward(state).detach()
        return action.data.numpy()

    def get_exploration_action(self, state):

        action = self.actor.forward(state).detach()
        new_action = action.data.numpy() + (self.noise.sample() * self.action_lim)
        return new_action

    def optimize(self):
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)
        batch_state = Variable(torch.cat(batch_state)).view(-1,self.state_dim)
        batch_action = Variable(torch.cat(batch_action)).view(-1,self.action_dim)
        batch_reward = Variable(torch.cat(batch_reward)).view(-1, 1)
        batch_next_state = Variable(torch.cat(batch_next_state)).view(-1,self.state_dim)

        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        next_action = self.target_actor(batch_next_state).detach()
        max_next_q_values = self.target_critic(batch_next_state, next_action.detach())
        expected_q_values = batch_reward + GAMMA * max_next_q_values

         # update critic network
        self.critic_optimizer.zero_grad()
        current_q_values = self.critic(batch_state, batch_action)
        critic_loss = F.smooth_l1_loss(current_q_values, expected_q_values.detach())
        critic_loss.backward()
        #print("critic loss",critic_loss)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        # update actor network
        self.actor_optimizer.zero_grad()
        # accurate action prediction
        current_action = self.actor(batch_state)
        actor_loss = -torch.sum(self.critic(batch_state,current_action))
        actor_loss.backward()
        self.actor_optimizer.step()
        #print("actor loss",actor_loss)

        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)

    def save_models(self, episode_count):
        """
        saves the target actor and critic models
        :param episode_count: the count of episodes iterated
        :return:
        """
        torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
        print ('Models saved successfully')

    def load_models(self, episode):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param episode: the count of episodes iterated (used to find the file name)
        :return:
        """
        self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print ('Models loaded succesfully')


env = gym.make('Pendulum-v0')

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 300
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]

memory = ReplayMemory(MAX_BUFFER)
trainer = Trainer(S_DIM, A_DIM, A_MAX, memory)

for _ep in range(MAX_EPISODES):
    observation = env.reset()
    print ('EPISODE :- ', _ep)
    episode_reward = 0
    for r in range(MAX_STEPS):
        env.render()
        state = np.float32(observation)

        action = trainer.get_exploration_action(Variable(FloatTensor(state)))

        new_observation, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
            transition = (FloatTensor(state), FloatTensor(action), FloatTensor(new_state), FloatTensor([reward]))
            memory.push(transition)

        observation = new_observation

        # perform optimization
        trainer.optimize()
        if done:
            break
    print ('REWARDS :- ', episode_reward)

    if _ep%100 == 0:
        trainer.save_models(_ep)
