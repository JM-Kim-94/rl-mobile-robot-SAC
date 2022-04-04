#!/usr/bin/env python3


import rospy
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from collections import namedtuple
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

from DDPG.env_mobile_robot_DDPG_2 import Env

from torch.utils.tensorboard import SummaryWriter


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=0.2, dt=0.01, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()
         # OUNoise 입력은 np.array([float mu]), 출력은 ndarray([float n])

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class NormalNoise:        
    def ep_to_sigma_angular(self, episode):
        if        episode<200:  return round(-0.003*episode  + 0.7, 5)
        else:                   return 0.05

    def ep_to_sigma_linear(self, episode):
        if        episode<100:  return round(-0.003*episode  + 0.5, 5)
        elif 100<=episode<200:  return round(-0.0015*episode + 0.35, 5)
        else:                   return 0.05
    
    def Noise(self, episode):
        sigma_angular = self.ep_to_sigma_angular(episode)
        sigma_linear  = self.ep_to_sigma_linear (episode)
        
        noise_angular = np.random.normal(0, sigma_angular)
        noise_linear  = np.random.normal(0, sigma_linear)
              
        return [noise_angular, noise_linear]
        

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.next_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, next_states, terminals




class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, name, chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.angular_bound = 1.5
        self.linear_bound  = 0.5
        self.chkpt_file = os.path.join(chkpt_dir, name + '_ddpg')

        self.fc1 = nn.Linear(self.state_size, 128)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.mu_angular = nn.Linear(64, 1)
        torch.nn.init.xavier_normal_(self.mu_angular.weight) 

        self.mu_linear = nn.Linear(64, 1)
        torch.nn.init.xavier_normal_(self.mu_linear.weight)       

        self.optimzer = optim.Adam(self.parameters(), lr=learning_rate)

        self.to(DEVICE)

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        
        angular = torch.tanh(self.mu_angular(x))*self.angular_bound
        linear  = torch.softplus(self.mu_linear(x))*self.linear_bound 

        action = torch.cat([angular, linear], 1)
        
        return action

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.chkpt_file))




class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, name, chkpt_dir='/save_DDPG_models/'):
        super(CriticNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.chkpt_file = os.path.join(chkpt_dir, name + '_ddpg')
        
        self.fc1_s_dim = 128
        self.fc1_a_dim = 32
        self.fc2_dim = self.fc1_s_dim + self.fc1_a_dim

        self.fc1_state = nn.Linear(self.state_size, self.fc1_s_dim)
        nn.init.kaiming_normal_(self.fc1_state.weight)
        self.bn1_state = nn.BatchNorm1d(self.fc1_s_dim)
        
        self.fc1_action = nn.Linear(self.action_size, self.fc1_a_dim)
        nn.init.kaiming_normal_(self.fc1_action.weight)
        self.bn1_action = nn.BatchNorm1d(self.fc1_a_dim)

        self.fc2 = nn.Linear(self.fc2_dim, 200)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(200)
        
        self.fc3 = nn.Linear(200, 128)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(64)

        self.q = nn.Linear(64, self.action_size)
        nn.init.kaiming_normal_(self.q.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.to(DEVICE)

    def forward(self, state, action):
        state  = F.leaky_relu(self.bn1_state(self.fc1_state(state)))
        action = F.leaky_relu(self.bn1_action(self.fc1_action(action)))
        
        x = torch.cat([state, action], dim=-1)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        
        q_out = self.q(x)

        return q_out

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.chkpt_file))



class Agent(object):
    def __init__(self):
        self.EPISODE = 1500
        self.MAX_STEP_SIZE = 3000
        self.state_size = 4
        self.action_size = 2
        self.lr_actor  = 0.0001
        self.lr_critic = 0.0001
        self.max_buffer_size = 500000
        self.batch_size = 100
        self.gamma = 0.99
        self.tau = 0.0005
        self.memory = ReplayBuffer(self.max_buffer_size, self.state_size, self.action_size)
        
        self.actor = ActorNetwork(self.state_size, self.action_size, self.lr_actor, name='Actor')
        self.critic = CriticNetwork(self.state_size, self.action_size, self.lr_critic, name='Critic')
        self.target_actor = ActorNetwork(self.state_size, self.action_size, self.lr_actor, name='TargetActor')        
        self.target_critic = CriticNetwork(self.state_size, self.action_size, self.lr_critic, name='TargetCritic')

        #self.noise = OUActionNoise(mu=np.zeros(self.action_size))
        self.normal_noise = NormalNoise()

        self.soft_update_target(self.target_actor, self.actor, 1)
        self.soft_update_target(self.target_critic, self.critic, 1)

    def choose_action(self, observation, episode):
        self.actor.eval()

        observation = torch.FloatTensor(observation).to(DEVICE)
        mu = self.actor(observation).to(DEVICE)
        
        #mu_prime = mu + torch.FloatTensor(self.noise()).to(DEVICE)  # OU_Noise
        #mu_prime = mu_prime.cpu().detach().numpy()
        
        mu_prime = mu + torch.FloatTensor(self.normal_noise.Noise(episode)).to(DEVICE)
        mu_prime = mu_prime.cpu().detach().numpy()
        
        mu_prime[0][0] = np.clip(mu_prime[0][0], -1.5, 1.5)
        mu_prime[0][1] = np.clip(mu_prime[0][1], 0, 0.51)
        print("mu_prime:",mu_prime)
        self.actor.train()
        
        return mu_prime

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def train(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        
        # print("학습")

        state      = torch.FloatTensor(state).to(DEVICE)
        action     = torch.FloatTensor(action).to(DEVICE)
        reward     = torch.FloatTensor(reward).to(DEVICE)
        next_state = torch.FloatTensor(next_state).to(DEVICE)
        done       = torch.Tensor(done).to(DEVICE)
        #print("state",state)
        #print("action",action)
        #print("reward",reward)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(next_state)
        target_critic_value = self.target_critic.forward(next_state, target_actions)
        critic_value = self.critic.forward(state, action)

        
        target = torch.Tensor([]).to(DEVICE)
        for j in range(self.batch_size):
            target = torch.cat([target, reward[j] + self.gamma*target_critic_value[j]*done[j]])
            
        target = target.view(self.batch_size, 2)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimzer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimzer.step()

        self.soft_update_target(self.target_actor,  self.actor,  self.tau)
        self.soft_update_target(self.target_critic, self.critic, self.tau)
        
    def soft_update_target(self, target, orign, tau):
        for target_param, orign_param in zip(target.parameters(), orign.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * orign_param.data)


    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        #self.target_actor.save_checkpoint()
        #self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()



if __name__ == "__main__":
    rospy.init_node('mobile_robot_DDPG')
    
    date = '1008_1'
    save_dir = '/home/jm-kim/catkin_ws/src/myrobot/src/DDPG/save_DDPG_models/'
    writer = SummaryWriter('DDPG_log/1008/2_action/vel_reward_vel')

    #DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DEVICE = torch.device('cpu')
    print("DEVICE : ", DEVICE)
    
    agent = Agent()
    env = Env(agent.action_size)
    time.sleep(1.5)
    
    for episode in range(agent.EPISODE): 
    
        done = False
        episode_score = 0.0
        s = env.reset(episode)
        
        for t in range(agent.MAX_STEP_SIZE):
        
            a = agent.choose_action([s], episode) 
            s_next, r, done = env.step(a, episode, t)           
            
            agent.remember(s, a, r, s_next, int(done))
            
            agent.train()
            
            episode_score += r
            s = s_next
                        
            if done:
                s = env.reset(episode)
        
        if episode % 2 == 0:            
            actor_chkpt  = save_dir + date + '/DDPG_Actor_EP'  + str(episode) + '.pt'
            critic_chkpt = save_dir + date + '/DDPG_Critic_EP' + str(episode) + '.pt'
            torch.save(agent.actor.state_dict(),  actor_chkpt)
            torch.save(agent.critic.state_dict(), critic_chkpt)
        
        print("EP:",episode, " SCORE:%0.3f"%episode_score)
        writer.add_scalar("Score", episode_score, episode)
        
    print("종료")




















