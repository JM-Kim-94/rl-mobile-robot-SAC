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
        linear  = torch.sigmoid(self.mu_linear(x))*self.linear_bound 

        action = torch.cat([angular, linear], 1)
        
        return action

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
        
        self.actor = ActorNetwork(self.state_size, self.action_size, self.lr_actor, name='Actor')
        
        self.actor.load_state_dict(torch.load("/home/jm-kim/catkin_ws/src/myrobot/src/DDPG/save_DDPG_models/1008_1/DDPG_Actor_EP1232.pt"))
        print("MODEL:")
        print(self.actor)


    def choose_action(self, observation, episode):
        self.actor.eval()
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(DEVICE)
            mu = self.actor(observation).to(DEVICE)
        
            #noise = torch.FloatTensor(self.normal_noise.Noise(episode)).to(DEVICE)
            mu_prime = mu # + noise
            mu_prime = mu_prime.cpu().detach().numpy()
        
            mu_prime[0][0] = np.clip(mu_prime[0][0], -1.5, 1.5)
            mu_prime[0][1] = np.clip(mu_prime[0][1], 0, 0.51)
            #print("mu_prime:",mu_prime,'\n')
                
        return mu_prime




if __name__ == "__main__":
    rospy.init_node('mobile_robot_DDPG_Test')
    
    date = '1008_1'
    save_dir = '/home/jm-kim/catkin_ws/src/myrobot/src/DDPG/save_DDPG_models/'
    DEVICE = torch.device('cpu')
    print("DEVICE : ", DEVICE)
    
    agent = Agent()
    env = Env(agent.action_size)
    time.sleep(3)
    
    TOTAL_TRIAL = 1000
    trial, goal, collision = 0, 0, 0
    s = env.reset(0)
    while True:
        
        done = False
        a = agent.choose_action([s], 0)
        s_next, r, done = env.step(a, 200, 0)
        
        s = s_next
        
        if r > 800:
            print("Current trial : ", trial)
            goal += 1
            trial += 1
        
        if done:                 
            s = env.reset(0)  
            print("Current trial : ", trial)            
            collision += 1
            trial += 1
            time.sleep(0.5)
        
        if trial >= TOTAL_TRIAL:
            break
            
    accuracy = goal/(trial)
    print("Goal:",goal, " Collision:",collision, " Total Trial:", trial)
    print("Accuracy = ", accuracy*100 ,"%")
    
    
    
        





















