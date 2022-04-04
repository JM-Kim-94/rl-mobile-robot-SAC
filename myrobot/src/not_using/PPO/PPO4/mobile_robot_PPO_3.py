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

from PPO3.env_mobile_robot_PPO import Env

from torch.utils.tensorboard import SummaryWriter

from std_srvs.srv import Empty

#Hyperparameters
learning_rate   = 0.001
gamma           = 0.99
lmbda           = 0.9
eps_clip        = 0.6
K_epoch         = 4
rollout_len     = 500
buffer_size     = 2
minibatch_size  = 16

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.fc1   = nn.Linear(4,128)
        self.fc2   = nn.Linear(128,64)
        
        self.fc3   = nn.Linear(64,64)
        self.fc_mu = nn.Linear(128,1) 
        self.fc_std  = nn.Linear(128,1)
        
        self.fc4   = nn.Linear(64,64)
        self.fc_v = nn.Linear(128,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimization_step = 0
        
        #nn.init.kaiming_normal_(self.fc1.weight)
        #nn.init.kaiming_normal_(self.fc2.weight)
        #nn.init.kaiming_normal_(self.fc3.weight)
        #nn.init.kaiming_normal_(self.fc4.weight)
        #nn.init.kaiming_normal_(self.fc_mu.weight)
        #nn.init.xavier_normal_(self.fc_mu.weight)
        #nn.init.kaiming_normal_(self.fc_std.weight) 
        #nn.init.kaiming_normal_(self.fc_v.weight) 
        
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

    def pi(self, x, softmax_dim = 0):
        x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        mu = 1.5*torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        std = torch.clamp(std, 0.01, 1.0)
        return mu, std
    
    def v(self, x):
        x = F.leaky_relu(self.fc1(x))
        #x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc4(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_batch, a_batch, r_batch, s_prime_batch, prob_a_batch, done_batch = [], [], [], [], [], []
        data = []

        for j in range(buffer_size):
            for i in range(minibatch_size):
                rollout = self.data.pop()
                s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

                for transition in rollout:
                    s, a, r, s_prime, prob_a, done = transition
                    
                    s_lst.append(s)
                    a_lst.append([a])
                    r_lst.append([r])
                    s_prime_lst.append(s_prime)
                    prob_a_lst.append([prob_a])
                    done_mask = 0 if done else 1
                    done_lst.append([done_mask])

                s_batch.append(s_lst)
                a_batch.append(a_lst)
                r_batch.append(r_lst)
                s_prime_batch.append(s_prime_lst)
                prob_a_batch.append(prob_a_lst)
                done_batch.append(done_lst)
                    
            mini_batch = torch.tensor(s_batch, dtype=torch.float).to(DEVICE), torch.tensor(a_batch, dtype=torch.float).to(DEVICE), \
                          torch.tensor(r_batch, dtype=torch.float).to(DEVICE), torch.tensor(s_prime_batch, dtype=torch.float).to(DEVICE), \
                          torch.tensor(done_batch, dtype=torch.float).to(DEVICE), torch.tensor(prob_a_batch, dtype=torch.float).to(DEVICE)
            data.append(mini_batch)

        return data

    def calc_advantage(self, data):
        data_with_adv = []
        for mini_batch in data:
            s, a, r, s_prime, done_mask, old_log_prob = mini_batch
            with torch.no_grad():
                td_target = r + gamma * self.v(s_prime) * done_mask
                delta = td_target - self.v(s)
            delta = delta.cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(DEVICE)
            data_with_adv.append((s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage))

        return data_with_adv

        
    def train_net(self):
        self.pause_proxy()
        #print("일시정지")
        if len(self.data) == minibatch_size * buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)

            for i in range(K_epoch):
                for mini_batch in data:
                    s, a, r, s_prime, done_mask, old_log_prob, td_target, advantage = mini_batch

                    mu, std = self.pi(s, softmax_dim=1)
                    dist = Normal(mu, std)
                    log_prob = dist.log_prob(a)
                    ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target) + dist.entropy().mean()

                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    #nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimization_step += 1
        #print("일시정지 해제")
        self.unpause_proxy()


if __name__ == "__main__":

    rospy.init_node('mobile_robot_ppo_3')
    
    date = '1111_1/'
    
    #writer = SummaryWriter('PPO_log/1111')
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(dev)
    print("DEVICE : ", DEVICE)    
    
    state_size = 4
    action_size = 1
    
    env = Env(action_size)
    model = PPO().to(DEVICE)
    score = 0.0
    print_interval = 1
    rollout = []

    for n_epi in range(10000):
        s = env.reset()
        done = False
        for step in range(int(3000/rollout_len)):
            for t in range(rollout_len):
                
                mu, std = model.pi(torch.FloatTensor(s).to(DEVICE))
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                s_prime, r, done, info = env.step([a.item()])

                rollout.append((s, a, r, s_prime, log_prob.item(), done))
                if len(rollout) == rollout_len:
                    model.put_data(rollout)
                    rollout = []

                s = s_prime
                score += r
                if done:
                    s = env.reset()

            #pause_proxy()
            
            model.train_net()
            #unpause_proxy()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score/print_interval, model.optimization_step))
            score = 0.0

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
