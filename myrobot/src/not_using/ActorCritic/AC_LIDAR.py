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
from ActorCritic.env_AC_LIDAR import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('AC_log/0913/reduce_state_4')


USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('DEVICE:',device)
#device = torch.device("cpu")

EPISODES = 1500




import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal


def cvt2Tensor(in_put, input_size, batsize=1):
    in_put = np.reshape(in_put, [batsize, input_size])
    in_put = torch.FloatTensor(in_put).to(device)
    return in_put


class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound, learning_rate):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = torch.FloatTensor([action_bound]).to(device)
        self.learning_rate = learning_rate
        self.loss = 0.0

        self.std_bound = [1e-2, 0.99]
        
        self.fc1 = nn.Linear(self.state_size, 128) #300dldjTdma       
        self.bn1 = nn.BatchNorm1d(128)
        self.drp1 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc1.weight)
        #nn.init.xavier_uniform(self.fc1.weight)
        
        self.fc2 = nn.Linear(128, 128)       
        self.bn2 = nn.BatchNorm1d(128) 
        self.drp2 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc2.weight)
        #nn.init.xavier_uniform(self.fc2.weight)
        
        self.fc3 = nn.Linear(128, 64)       
        self.bn3 = nn.BatchNorm1d(64) 
        self.drp3 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc3.weight)
        #nn.init.xavier_uniform(self.fc3.weight)
        
        self.fc4 = nn.Linear(64, 64)       
        self.bn4 = nn.BatchNorm1d(64) 
        self.drp4 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc4.weight)
        #nn.init.xavier_uniform(self.fc4.weight)
        
        self.fc5 = nn.Linear(64, 32)       
        self.bn5 = nn.BatchNorm1d(32) 
        self.drp5 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc5.weight)
        #nn.init.xavier_uniform(self.fc5.weight)
             
        self.fc_mu = nn.Linear(32, self.action_size)
        #nn.init.kaiming_normal_(self.fc_mu.weight) 
        nn.init.xavier_uniform(self.fc_mu.weight)
        
        self.fc_std = nn.Linear(32, self.action_size)
        #nn.init.kaiming_normal_(self.fc_std.weight)
        nn.init.xavier_uniform(self.fc_std.weight)
        
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = self.drp1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.drp2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.drp3(x)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.drp4(x)
        
        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.drp5(x)
        
        mu = self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x))    
        
        return mu * self.action_bound, std





class Critic(nn.Module):
    def __init__(self, state_size, action_size, lr):
        super(Critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = lr
        self.loss = 0.0

        self.fc1 = nn.Linear(self.state_size, 128) #300dldjTdma       
        self.bn1 = nn.BatchNorm1d(128)
        self.drp1 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(128, 128)       
        self.bn2 = nn.BatchNorm1d(128) 
        self.drp2 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(128, 64)       
        self.bn3 = nn.BatchNorm1d(64) 
        self.drp3 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc3.weight)
        
        self.fc4 = nn.Linear(64, 64)       
        self.bn4 = nn.BatchNorm1d(64) 
        self.drp4 = nn.Dropout(0.2) 
        nn.init.kaiming_normal_(self.fc4.weight)
        
        self.fc5 = nn.Linear(64, 64)       
        self.bn5 = nn.BatchNorm1d(64) 
        self.drp5 = nn.Dropout(0.5) 
        nn.init.kaiming_normal_(self.fc5.weight)
             
        self.fc_out = nn.Linear(64, self.action_size)
        nn.init.kaiming_normal_(self.fc_out.weight)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, state):
        x = F.leaky_relu(self.bn1(self.fc1(state)))
        x = self.drp1(x)
        
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = self.drp2(x)
        
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.drp3(x)
        
        x = F.leaky_relu(self.bn4(self.fc4(x)))
        x = self.drp4(x)
        
        x = F.leaky_relu(self.bn5(self.fc5(x)))
        x = self.drp5(x)
        
        v = F.leaky_relu(self.fc_out(x)) 

        return v









class A2C_agent(object):
    def __init__(self):
        self.GAMMA = 0.99
        self.BATCH_SIZE = 20
        self.ACTOR_LEARNING_RATE = 0.001
        self.CRITIC_LEARNING_RATE = 0.0005
        self.MAX_STEP_SIZE = 3000

        self.env = Env(1)
        self.state_size = 4 #109 # 
        self.action_size = 1   # 단 한 개 (continuous)
        self.action_bound = 1.5

        self.actor = Actor(self.state_size, self.action_size, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.actor = self.actor.to(device)
        print(self.actor,'\n')
        self.critic = Critic(self.state_size, self.action_size, self.CRITIC_LEARNING_RATE)
        self.critic = self.critic.to(device)
        print(self.critic,'\n')
        

        self.save_epi_reward = []
    
    def get_action(self, state):
        self.actor.eval()   
        
        with torch.no_grad():
            mu_a, std_a = self.actor(state)
            mu_a = mu_a.cpu().numpy()     #detach()
            std_a = std_a.cpu().numpy()   #detach()
        std_a = np.clip(std_a, self.actor.std_bound[0], self.actor.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=(1 ,self.action_size))

        return action
        
    
    def train_actor(self, states, actions, td_errors):
        print("train_actor!")
        self.actor.train()
        
        
        #actions = torch.FloatTensor(actions).view(states.shape[0], self.action_size)
        #td_errors = torch.FloatTensor(td_errors).view(states.shape[0], self.action_size)
        
        #states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        td_errors = torch.FloatTensor(td_errors).to(device)

        mu, std = self.actor(states)
        dist = Normal(mu, std)
        prob = dist.log_prob(actions)
        self.actor.loss = torch.mean(-prob * td_errors)

        self.actor.optimizer.zero_grad()
        self.actor.loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        self.actor.optimizer.step()

        return self.actor.loss
        
    def predict_v_value(self, state, next_state):
        self.critic.eval()
        with torch.no_grad():
            #v_value = self.critic(cvt2Tensor(state, self.state_size)).detach().numpy()
            #next_v_value = self.critic(cvt2Tensor(next_state, self.state_size)).detach().numpy() 
            v_value = self.critic(cvt2Tensor(state, self.state_size)).cpu().numpy()
            next_v_value = self.critic(cvt2Tensor(next_state, self.state_size)).cpu().numpy()    
    
        return v_value, next_v_value
    
    def train_critic(self, states, td_target):
        
        self.critic.train()
        print("train_critic!")
        
        td_target = torch.FloatTensor(td_target).to(device)
        predict = self.critic(states).to(device)
        self.critic.loss = torch.mean((predict - td_target) **2)  # MSE

        self.critic.optimizer.zero_grad()
        self.critic.loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        self.critic.optimizer.step()

        return self.critic.loss

    def TD_error_TD_target(self, reward, v_value, next_v_value, done):
        if done:
            TD_target = v_value
            TD_error = TD_target - v_value
        else:
            TD_target = reward + self.GAMMA * next_v_value
            TD_error = TD_target - v_value

        return TD_error, TD_target

    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch) - 1):
            unpack = np.append(unpack, batch[idx + 1], axis=0)

        return unpack

    def train(self, max_episode_num):
    
        for ep in range(max_episode_num):
            batch_state, batch_action, batch_td_target, batch_td_error = [], [], [], []

            time, episode_reward, done = 0, 0.0, False

            state = self.env.reset(ep)
            critic_loss = 0
            actor_loss = 0
            
            #state = torch.from_numpy(state).type(torch.FloatTensor)
            #state = torch.unsqueeze(state, 0).to(device)

            for step in range(self.MAX_STEP_SIZE):
                # if ep > 2500: 
                #     self.env.render()

                action = self.get_action(cvt2Tensor(state, self.state_size))
                #action = self.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)[0]
                print("action:%0.3f" % action)

                next_state, reward, done = self.env.step(action, ep, step)

                state = np.reshape(state, [1, self.state_size])
                next_state = np.reshape(next_state, [1, self.state_size])
                action = np.reshape(action, [1, self.action_size])
                reward = np.reshape(reward, [1, 1])
                  
                v_value, next_v_value = self.predict_v_value(state, next_state)

                train_reward = reward / 20
                td_error, td_target = self.TD_error_TD_target(train_reward, v_value, next_v_value, done)

                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(td_target)
                batch_td_error.append(td_error)
                
                if done:
                    state = self.env.reset(ep)

                if len(batch_state) < self.BATCH_SIZE:  # 32개 데이터 누적
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue

                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                td_errors = self.unpack_batch(batch_td_error)

                batch_state, batch_action, batch_td_target, batch_td_error = [], [], [], []

                critic_loss = self.train_critic(cvt2Tensor(states, self.state_size, states.shape[0]), td_targets)
                actor_loss = self.train_actor(cvt2Tensor(states, self.state_size, states.shape[0]), actions, td_errors)

                state = next_state[0]
                episode_reward += reward[0]
                time += 1

            print('Episode: ', ep + 1, 'Time: ', time, 'Reward: ', episode_reward, 'actor loss', actor_loss.item(),
                  'critic loss', critic_loss.item())
            writer.add_scalar("Score", episode_reward, ep)
            # tensorboard --logdir .ros/AC_log
            self.save_epi_reward.append(episode_reward)

            # if ep % 10 == 0:
            #    self.actor.save_weights('Pendulum_A2C/save_model/pendulum_actor.pt')
            #    self.critic.save_weights('Pendulum_A2C/save_model/pendulum_critic.pt')

        np.savetxt('pendulum_epi_reward_1.txt', self.save_epi_reward)







def main():
    max_episode_num = 10000
    
    agent = A2C_agent()
    
    time.sleep(2)

    agent.train(max_episode_num)


if __name__ == "__main__":
    rospy.init_node('mobile_robot_AC')
    main()


















