#!/usr/bin/env python3


import rospy
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import os
import shutil
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


#import GPUtil
#import psutil
#from threading import Thread
#import time

from env_mobile_robot_SAC_new_test import Env

#from torch.utils.tensorboard import SummaryWriter

from std_srvs.srv import Empty

from collections import namedtuple
import collections, random


        
        
class PolicyNet_old(nn.Module):
    def __init__(self, learning_rate, init_alpha, target_entropy, lr_alpha, DEVICE):
        super(PolicyNet_old, self).__init__()
        
        self.learning_rate  = learning_rate
        self.init_alpha     = init_alpha
        self.target_entropy = target_entropy
        self.lr_alpha       = lr_alpha
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32,2)
        self.fc_std  = nn.Linear(32,2)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(DEVICE)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        
        mu = self.fc_mu(x)
        #std = F.softplus(self.fc_std(x))
        
        real_action = torch.tanh(mu)
        
        return real_action   


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, DEVICE):
        super(PolicyNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 200)
        self.fc_2 = nn.Linear(200, 200)
        self.fc_3 = nn.Linear(200, 100)
        self.fc_4 = nn.Linear(100, 100)
        self.fc_mu = nn.Linear(100, action_dim)
        self.fc_std = nn.Linear(100, action_dim)

        self.lr = actor_lr
        self.dev = DEVICE

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = torch.FloatTensor([1.5, 0.5]).to(self.dev)
        self.min_action = torch.FloatTensor([-1.5, 0]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias = (self.max_action + self.min_action) / 2.0
        
        self.to(self.dev)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        x = F.leaky_relu(self.fc_4(x))
        mu = self.fc_mu(x)
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # # Enforcing Action Bound
        log_prob = reparameter.log_prob(x_t)
        log_prob = log_prob - torch.sum(torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), dim=-1, keepdim=True)

        return action, log_prob




class PolicyNet_1225(nn.Module):
    def __init__(self, state_dim, action_dim, actor_lr, DEVICE):
        super(PolicyNet_1225, self).__init__()
        
        self.fc_1    = nn.Linear(4, 128)
        self.fc_2    = nn.Linear(128, 128)
        self.fc_3    = nn.Linear(128, 64)
        self.fc_mu   = nn.Linear(64,2)
        self.fc_std  = nn.Linear(64,2)
                
        self.lr = actor_lr
        self.dev = DEVICE
        
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.max_action = torch.FloatTensor([1.5, 0.5]).to(self.dev)
        self.min_action = torch.FloatTensor([-1.5, 0]).to(self.dev)
        self.action_scale = (self.max_action - self.min_action) / 2.0
        self.action_bias  = (self.max_action + self.min_action) / 2.0
        
        self.to(self.dev)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))        
        mu      = self.fc_mu(x)
        log_std = self.fc_std(x)        
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mu, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        reparameter = Normal(mean, std)
        x_t = reparameter.rsample()
        y_t = torch.tanh(mean)
        action = self.action_scale * y_t + self.action_bias
        
        return action, y_t



class Agent:

    def __init__(self):    
        self.lr_pi           = 0.0001
        self.lr_q            = 0.0005
        self.init_alpha      = 8
        self.gamma           = 0.98
        self.batch_size      = 128
        self.buffer_limit    = 150000
        self.tau             = 0.001    # for target network soft update
        self.target_entropy  = -1       # for automated alpha update
        self.lr_alpha        = 0.001    # for automated alpha update
        self.DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("DEVICE : ", self.DEVICE)
        
        
        self.pi = PolicyNet_1225(4, 2, 0.001, self.DEVICE)
        
        #self.pi.load_state_dict(torch.load("/home/jm-kim/catkin_ws/src/myrobot/src/SAC/saved_model/1225/sac_actor_1225_EP662.pt"))
        self.pi.load_state_dict(torch.load("/home/jm-kim/catkin_ws/src/myrobot/src/SAC/saved_model/1223_1/sac_actor_1223_1_EP388.pt"))
               
        print("ACTOR:\n", self.pi)

    def choose_action(self, state):
        with torch.no_grad():
            action, log_prob = self.pi.sample(s.to(self.DEVICE))
        return action
    
    



if __name__ == '__main__':
    rospy.init_node('mobile_robot_sac_test')
    
    env = Env()
    agent = Agent()
    
    time.sleep(3)
    
    s = env.reset(0)
    done = False
    
    total_trial = 0
    collision, arrival = 0, 0
    

    while True:     
        s = torch.FloatTensor(s)
        real_action = agent.choose_action(s)
            
        s_prime, r, done, arrv = env.step(real_action, 200)                       
                    
        s = s_prime
            
        if done:
            s = env.reset(0)      
            total_trial += 1
            collision   += 1
            print("Current Trial : ",total_trial)
            time.sleep(0.5)
            
        if arrv:
            total_trial += 1
            arrival     += 1
            print("Current Trial : ",total_trial)
        
        if total_trial == 1000:
             break
     
     
    print("\n", "ARRIVAL=",arrival," COLLISION=",collision," ACCURACY=",(arrival/total_trial)*100,"%","\n")
        
        #sim_rate.sleep()
        










    
    
    
    
    
    
    
    
    
    
    
    
    

