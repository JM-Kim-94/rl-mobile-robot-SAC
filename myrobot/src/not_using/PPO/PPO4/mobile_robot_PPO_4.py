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

from PPO4.env_mobile_robot_PPO import Env

from torch.utils.tensorboard import SummaryWriter

from std_srvs.srv import Empty

class ActorNet(nn.Module):
    def __init__(self, learning_rate):
        super(ActorNet,self).__init__()
        self.learning_rate = learning_rate
        
        self.fc1 = nn.Linear(4, 128)  
        nn.init.kaiming_normal_(self.fc1.weight)            
        self.fc2 = nn.Linear(128, 64)
        nn.init.kaiming_normal_(self.fc2.weight)         
        self.fc3 = nn.Linear(64, 64)      
        nn.init.kaiming_normal_(self.fc3.weight)          
        self.fc_mu = nn.Linear(64, 1)
        nn.init.uniform_(self.fc_mu.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_mu.bias.data, -0.003, 0.003)
        #nn.init.xavier_normal_(self.fc_mu.weight)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        mu = 1.5 * torch.tanh(self.fc_mu(x))
        return mu
        

class CriticNet(nn.Module):
    def __init__(self, learning_rate):
        super(CriticNet,self).__init__()
        self.learning_rate = learning_rate
        
        self.fc1 = nn.Linear(4, 128) 
        nn.init.kaiming_normal_(self.fc1.weight)              
        self.fc2 = nn.Linear(128, 64)  
        nn.init.kaiming_normal_(self.fc2.weight)        
        self.fc3 = nn.Linear(64, 64) 
        nn.init.kaiming_normal_(self.fc3.weight)          
        self.fc_v = nn.Linear(64, 1)
        nn.init.uniform_(self.fc_v.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_v.bias.data, -0.003, 0.003)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        #x = F.leaky_relu(self.fc3(x))
        v = self.fc_v(x)
        return v


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        
        self.a_lr = 0.00001
        self.c_lr = 0.0001
        
        self.pi = ActorNet(self.a_lr).to(DEVICE)
        self.v  = CriticNet(self.c_lr).to(DEVICE)
        self.optimization_step = 0
        
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)


    def put_data(self, transition):
        self.data.append(transition)


    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
                
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s         = torch.FloatTensor(s_lst).to(DEVICE)
        a         = torch.FloatTensor(a_lst).to(DEVICE)
        r         = torch.FloatTensor(r_lst).to(DEVICE)
        s_prime   = torch.FloatTensor(s_prime_lst).to(DEVICE)
        done_mask = torch.FloatTensor(done_lst).to(DEVICE)
        prob_a    = torch.FloatTensor(prob_a_lst).to(DEVICE)         
                                              
        self.data = []
        #r = (r - r.mean()) / (r.std() + 1e-7)
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self, n_epi):
        self.pause_proxy()        
        
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
        
        self.v.train()
        self.pi.train()
        
        for i in range(K_epoch):        
            
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()      
                  
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.FloatTensor(advantage_lst).to(DEVICE)

            mu = self.pi(s)
            std = 0.8 if n_epi<500 else 0.3
            dist = Normal(mu, std)
            log_prob = dist.log_prob(a)          
        
            ratio = torch.exp(log_prob - old_log_prob).to(DEVICE)     
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            
            actor_loss  = (-torch.min(surr1, surr2)).mean()            
            critic_loss = F.smooth_l1_loss(self.v(s), td_target)   
            
            self.v.optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.v.parameters(), 2.0)
            self.v.optimizer.step()            
            
            self.pi.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 2.0)
            self.pi.optimizer.step()
            
            self.optimization_step += 1
            
        self.unpause_proxy()





if __name__ == '__main__':
    rospy.init_node('mobile_robot_ppo_4')
    
    date = '1202_1_2manstep_2'
    save_dir = "/home/jm-kim/catkin_ws/src/myrobot/src/PPO/PPO4/saved_model/" + date 
    os.mkdir(save_dir)
    save_dir += "/"
    
    writer = SummaryWriter('PPO_log/'+date)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    #dev = 'cpu'
    DEVICE = torch.device(dev)
    print("DEVICE : ", DEVICE)  
    
    state_size = 4
    action_size = 1
    
    
     # Hyperparameters
    gamma = 0.999995
    lmbda = 0.99
    eps_clip = 0.35
    K_epoch = 100
    rollout_len = 20000
    EPISODE = 20000
    
    env = Env(action_size)
    
    time.sleep(3)
    agent = PPO()
    score = 0.0
    print_interval = 1
    rollout = []
    
    #sim_rate = rospy.Rate(50)
    
    for n_epi in range(10000):
        s = env.reset()
        done = False
        for step in range(int(EPISODE/rollout_len)):
            for t in range(rollout_len):
                agent.pi.eval()
                with torch.no_grad():
                    mu = agent.pi(torch.FloatTensor(s).to(DEVICE))
                mu = mu.detach().cpu().numpy()[0]
                #print(mu)
                std = 0.8 if n_epi<500 else 0.3
                dist = Normal(mu, std)
                a = dist.sample()
                log_prob = dist.log_prob(a)
                #aa = np.clip(a.item(),-1.5, 1.5)
                s_prime, r, done, arrival = env.step([a])
                
                print("state:",s, " mu from pi:", mu, " stochastic action from mu:", a)
                #sim_rate.sleep()

                agent.put_data((s, a, r, s_prime, log_prob, done))

                s = s_prime
                score += r
                if done or arrival:
                    s = env.reset()

            agent.train_net(n_epi)
            
        if n_epi % 2 == 0: 
            torch.save(agent.pi.state_dict(), save_dir + "ppo_"+str(n_epi)+".pt")
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}, opt step: {}".format(n_epi, score / print_interval,
                                                                              agent.optimization_step))
            writer.add_scalar("Score", score / print_interval, n_epi)
            score = 0.0
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    

