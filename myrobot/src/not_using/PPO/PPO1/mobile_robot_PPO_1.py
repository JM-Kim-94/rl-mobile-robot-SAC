#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import os
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

from PPO1.env_mobile_robot_PPO import Env

from torch.utils.tensorboard import SummaryWriter

from std_srvs.srv import Empty


DEVICE = torch.device('cpu')
print("DEVICE : ", DEVICE)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, ratio_clipping):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.ratio_clipping = ratio_clipping

        self.std_bound = [1e-2, 1.0]        
        
        self.fc1 = nn.Linear(self.state_dim, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, 32)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.fc_mu = nn.Linear(32, action_dim)
        nn.init.xavier_normal_(self.fc_mu.weight) 

        self.fc_std= nn.Linear(32, action_dim)
        nn.init.xavier_normal_(self.fc_std.weight) 
        
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()      
        

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)        


    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))

        out_mu = self.tanh(self.fc_mu(x))
        std_output = self.softplus(self.fc_std(x))

        return out_mu * self.action_bound, std_output

    def log_pdf(self, mu, std, action):
        std = std.clamp(self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = - 0.5 * (((action - mu) ** 2 / var) + (torch.log(var * 2 * np.pi)))  
        return log_policy_pdf

    def get_policy_action(self, state):
        self.eval()
        mu_a, std_a = self.forward(state)
        mu_a = mu_a.item()
        std_a = std_a.item()
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)

        return mu_a, std_a, action

    def predict(self, state):
        self.eval()
        mu_a, std_a = self.forward(state)
        return mu_a

    def Learn(self, log_old_policy_pdf, states, actions, advantages):
        self.train()
        log_old_policy_pdf = torch.FloatTensor(log_old_policy_pdf)
        actions = torch.FloatTensor(actions).view(states.shape[0], 1)
        advantages = torch.FloatTensor(advantages).view(states.shape[0], 1).detach()

        mu, std = self.forward(states)
        log_policy_pdf = self.log_pdf(mu, std, actions)

        ratio = torch.exp(log_policy_pdf - log_old_policy_pdf)
        clipped_ratio = ratio.clamp(1.0 - self.ratio_clipping, 1.0 + self.ratio_clipping)

        surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages)
        loss = surrogate.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, lr):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = lr

        self.fc1 = nn.Linear(self.state_dim, 64)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 32)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(32, 32)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.fc_v = nn.Linear(32, action_dim)
        torch.nn.init.xavier_normal_(self.fc_v.weight)           

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc_v(x)

        return x

    def predict(self, state):
        self.eval()
        v = self.forward(state)
        return v

    def Learn(self, states, td_target):
        self.train()
        td_target = torch.FloatTensor(td_target).detach()
        predict = self.forward(states)

        loss = torch.mean((predict - td_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))


def convertToTensorInput(input, input_size, batsize=1):
    input = np.reshape(input, [batsize, input_size])
    return torch.FloatTensor(input).to(DEVICE)


class PPOAgent(object):
    def __init__(self, date):   
        self.date = date 
        self.save_dir = '/home/jm-kim/catkin_ws/src/myrobot/src/PPO/PPO1/save_PPO_models/' + self.date
        self.action_size = 1
        self.MAX_STEP_SIZE = 10000
        self.GAMMA = 0.99
        self.BATCH_SIZE = 500
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.0001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 100
        self.GAE_LAMBDA = 0.9  # 0.8
        self.load_model = False
        
        self.state_dim = 4
        self.action_dim = 1
        self.action_bound = 1.5        
        
        self.env = Env(self.action_size)
        self.pause_sim   = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE, self.RATIO_CLIPPING).to(DEVICE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE).to(DEVICE)

        self.save_epi_reward = []

    def gae_target(self, rewards, v_values, next_v_value, done):
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        n_step_targets = torch.zeros_like(rewards).to(DEVICE)
        gae = torch.zeros_like(rewards).to(DEVICE)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack

    def train(self, max_episode_num):
        batch_s, batch_a, batch_r = [], [], []
        batch_log_old_policy_pdf = []

        for ep in range(int(max_episode_num)):
        
            time, episode_reward, done = 0.0, 0.0, False
            s = self.env.reset(ep)

            for t in range(self.MAX_STEP_SIZE):
            
                mu_old, std_old, a = self.actor.get_policy_action(convertToTensorInput(s, self.state_dim))
                a = np.clip(a, -self.action_bound, self.action_bound)

                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (a - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)

                s_next, r, done, goal = self.env.step(a, ep, t)

                s = np.reshape(s, [1, self.state_dim])
                a = np.reshape(a, [1, self.action_dim])
                r = np.reshape(r, [1, 1])
                log_old_policy_pdf = np.reshape(log_old_policy_pdf, [1, 1])

                batch_s.append(s)
                batch_a.append(a)
                batch_r.append(r)

                batch_log_old_policy_pdf.append(log_old_policy_pdf)
                
                if done or goal:
                    s = self.env.reset(ep)

                if len(batch_s) < self.BATCH_SIZE:
                    s = s_next
                    episode_reward += r
                    time += 1
                    continue

                states = self.unpack_batch(batch_s)
                actions = self.unpack_batch(batch_a)
                rewards = self.unpack_batch(batch_r)
                #rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-10)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)

                batch_s, batch_a, batch_r = [], [], []
                batch_log_old_policy_pdf = []

                s_next = np.reshape(s_next, [1, self.state_dim])
                next_v_value = self.critic.predict(convertToTensorInput(s_next, self.state_dim))
                v_values = self.critic.predict(convertToTensorInput(states, self.state_dim, states.shape[0]))
                gaes, y_i = self.gae_target(rewards, v_values, next_v_value, done)
                
                self.pause_sim()
                for _ in range(self.EPOCHS):
                    # train
                    self.critic.Learn(convertToTensorInput(states, self.state_dim, states.shape[0]), y_i)
                    self.actor.Learn(log_old_policy_pdfs, convertToTensorInput(states, self.state_dim, states.shape[0]),actions, gaes)                    
                self.unpause_sim()

                s = s_next
                episode_reward += r[0]

                time += 1

            print('Episode: ', ep + 1, 'Reward: ', episode_reward)
            writer.add_scalar("Score", episode_reward, ep)

            self.save_epi_reward.append(episode_reward)

            if ep % 2 == 0:
                self.actor.save_weights(self.save_dir + 'pendulum_actor_'  + str(ep) + '.pt')
                self.critic.save_weights(self.save_dir + 'pendulum_critic_'+ str(ep) + '.pt')

        np.savetxt('pendulum_epi_reward.txt', self.save_epi_reward)

    def plot_Result(self):
        plt.plot(self.save_epi_reward)
        plt.show()



if __name__ == "__main__":

    rospy.init_node('mobile_robot_ppo')
    
    date = '1111_1/'
    
    writer = SummaryWriter('PPO_log/1111')
    
    
    
    agent = PPOAgent(date)    
    time.sleep(1.5)
    
    max_episode_num = 5000

    agent.train(max_episode_num)

    agent.plot_Result()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
