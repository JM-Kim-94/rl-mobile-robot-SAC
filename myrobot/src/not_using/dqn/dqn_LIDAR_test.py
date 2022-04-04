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
from dqn.env.env_dqn_LIDAR_test import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

EPISODES = 1000

#class DQN(nn.Module):
#    def __init__(self, state_size, action_size): 
#        super(DQN, self).__init__() 
#        self.state_size = state_size
#        self.action_size = action_size
#        
#        self.fc1 = nn.Linear(self.state_size, 200) #300dldjTdma       
#        self.bn1 = nn.BatchNorm1d(200)
#        self.drp1 = nn.Dropout(0.3) 
#        #nn.init.kaiming_normal_(self.fc1.weight)        
#        
#        self.fc2 = nn.Linear(200, 128)       
#        self.bn2 = nn.BatchNorm1d(128) 
#        self.drp2 = nn.Dropout(0.3) 
#        #nn.init.kaiming_normal_(self.fc2.weight)
#        
#        self.fc3 = nn.Linear(128, 64)       
#        self.bn3 = nn.BatchNorm1d(64) 
#        self.drp3 = nn.Dropout(0.3) 
#        #nn.init.kaiming_normal_(self.fc3.weight)
#        
#        self.fc4 = nn.Linear(64, 32)       
#        self.bn4 = nn.BatchNorm1d(32) 
#        self.drp4 = nn.Dropout(0.3) 
#        #nn.init.kaiming_normal_(self.fc4.weight)
#           
#        self.fc5 = nn.Linear(32, self.action_size)
#        #nn.init.kaiming_normal_(self.fc5.weight) 
#        
#        #nn.init.kaiming_normal_(self.fc1.weight)
#        #nn.init.kaiming_normal_(self.fc2.weight)
#        #nn.init.kaiming_normal_(self.fc3.weight) 
#            
#    def forward(self, x):
#        x = F.leaky_relu(self.fc1(x))
#        x = self.bn1(x)
#        x = self.drp1(x)
#        
#        x = F.leaky_relu(self.fc2(x))
#        x = self.bn2(x)
#        x = self.drp2(x)
#        
#        x = F.leaky_relu(self.fc3(x))
#        x = self.bn3(x)
#        x = self.drp3(x)
#        
#        x = F.leaky_relu(self.fc4(x))
#        x = self.bn4(x)
#        x = self.drp4(x)
#        
#        x = self.fc5(x)
#        return x

class DQN(nn.Module):
    def __init__(self, state_size, action_size): 
        super(DQN, self).__init__() 
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(self.state_size, 200) #300dldjTdma       
        self.bn1 = nn.BatchNorm1d(200)
        self.drp1 = nn.Dropout(0.3) 
        nn.init.kaiming_normal_(self.fc1.weight)
        
        self.fc2 = nn.Linear(200, 128)       
        self.bn2 = nn.BatchNorm1d(128) 
        self.drp2 = nn.Dropout(0.3) 
        nn.init.kaiming_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(128, 64)       
        self.bn3 = nn.BatchNorm1d(64) 
        self.drp3 = nn.Dropout(0.3) 
        nn.init.kaiming_normal_(self.fc3.weight)
                      
        self.fc4 = nn.Linear(64, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.drp4 = nn.Dropout(0.3)
        nn.init.kaiming_normal_(self.fc4.weight) 
             
        self.fc5 = nn.Linear(64, self.action_size)
        nn.init.kaiming_normal_(self.fc5.weight) 
        
        #nn.init.kaiming_normal_(self.fc1.weight)
        #nn.init.kaiming_normal_(self.fc2.weight)
        #nn.init.kaiming_normal_(self.fc3.weight) 
            
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.drp1(x)
        
        x = F.leaky_relu(self.fc2(x))
        x = self.bn2(x)
        x = self.drp2(x)
        
        x = F.leaky_relu(self.fc3(x))
        x = self.bn3(x)
        x = self.drp3(x)
        
        x = F.leaky_relu(self.fc4(x))
        x = self.bn4(x)
        x = self.drp4(x)
        
        x = self.fc5(x)
        return x

  
class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):
        
        """transition 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, state_next, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
   


class Brain():
    def __init__(self, state_size, action_size):
        #self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.date = '0820'
        self.dirPath = self.dirPath.replace('src/dqn', 'save_model/'+self.date+'_2/dqn_lidar_')
        self.result = Float32MultiArray()
        
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 3000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.epsilon_decay = 0.985
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        self.memory = ReplayMemory(200000)       

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        #self.model = torch.load("/home/jm-kim/catkin_ws/src/myrobot/save_model/0820_2/dqn_lidar_810.pt")
        
        self.model = torch.load("/home/jm-kim/catkin_ws/src/myrobot/save_model/0820_3/dqn_lidar_704.pt")
        print(self.model)
        
        self.loss = 0.0
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, momentum = 0.92)
        
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[300,400], gamma=0.5, verbose=True)
                
    def decide_action(self, state):
    
        #print("모델에 의한 행동선택")
        self.model.eval()
        with torch.no_grad():
            action = self.model(state).max(1)[1].view(1,1)
        
        return action
        
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return 
        
        self.mini_batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        
        self.expected_state_action_values = self.get_expected_state_action_values()
        
        self.update_q_network()
        
        
    def make_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        mini_batch = Transition(*zip(*transitions))
        #print("메모리에서 랜덤 샘플")
        
        state_batch = torch.cat(mini_batch.state)
        action_batch = torch.cat(mini_batch.action)
        reward_batch = torch.cat(mini_batch.reward)
        non_final_next_states = torch.cat([s for s in mini_batch.next_state if s is not None])
        
        return mini_batch, state_batch, action_batch, reward_batch, non_final_next_states
        
        
    def get_expected_state_action_values(self):
    
        self.model.eval()
        self.target_model.eval()
        
        #print(self.state_batch.shape)
        
        self.state_action_values = self.model(self.state_batch).gather(1, self.action_batch)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.mini_batch.next_state)), dtype=torch.bool).to(device)
        
        next_state_values = torch.zeros(self.batch_size).to(device)
        
        a_m = torch.zeros(self.batch_size,  dtype=torch.long).to(device)
        
        a_m[non_final_mask] = self.model(self.non_final_next_states).detach().max(1)[1]
        
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)
        
        next_state_values[non_final_mask] = self.target_model(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        
        expected_state_action_values = self.reward_batch + self.discount_factor*next_state_values
        
        return expected_state_action_values
        
        
    def update_q_network(self):
    
        self.model.train()
        self.loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        #loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1)) #원래 이거였음
        
        #print("모델 훈련")
        self.optimizer.zero_grad()
        self.loss.backward()
        #print("loss:%0.4f" % self.loss)
        #loss.backward() #원래 이거였음
        #for param in self.model.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target_q_network(self):
        #print("타겟모델 업데이트")
        self.target_model.load_state_dict(self.model.state_dict())       
            
        
    
                


class Agent():
    def __init__(self, state_size, action_size):
        self.brain = Brain(state_size, action_size)
        
    def update_q_function(self):
        self.brain.replay()
        
    def get_action(self, state):
        action = self.brain.decide_action(state)
        return action
        
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)        
        
    def update_target_q_function(self):
        self.brain.update_target_q_network()
                



if __name__ == '__main__':
    rospy.init_node('mobile_robot_dqn_test')
    
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    get_action = Float32MultiArray()
    
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_loss_result = rospy.Publisher('loss_result', Float32MultiArray, queue_size=5)
    
    result = Float32MultiArray()
    loss_result = Float32MultiArray()
    
    #210 + 4 = 214 , 105 + 4=109
    state_size = 109 #214  109
    action_size = 7
    
    goal, collision = 0, 0

    env = Env(action_size)
    
    agent = Agent(state_size, action_size)
    
    scores, losses, episodes = [], [], []
    global_step = 0
    start_time = time.time()
    
    time.sleep(0.5)
    for episode in range(agent.brain.load_episode + 1, EPISODES):
    
        done = False
        
        state = env.reset()
        #old_action = 3
        
        # print("Episode:",episode, "state:",state)
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0).to(device)
        
        score = 0
        losses = 0.0
        t = 0
        print("GOAL:",goal,"  COLIISION:",collision)
        if goal + collision >= 100:
            print("\n","ACCURACY:",goal/(goal + collision),"\n")
            break
        while True:
        #for t in range(agent.brain.episode_step):
            t += 1
            action = agent.get_action(state)
            #print("step: ", t, "   episode: ", episode)

            observation_next, reward, done = env.step(action.item())
            #print("Reward: ", reward)
            
            reward = (torch.tensor([reward]).type(torch.FloatTensor)).to(device)
            state_next = observation_next
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
            state_next = torch.unsqueeze(state_next, 0).to(device)
            
            #agent.memorize(state, action, state_next, reward)
            
            #agent.update_q_function()
            
            state = state_next
            old_action = action.item()
            
            score += reward
            losses += agent.brain.loss
            
            
            get_action.data = [action.int(), score, reward.int()]
            pub_get_action.publish(get_action)        
            
            
            if reward > 1000:
                goal += 1
                print("GOAL:",goal,"  COLIISION:",collision)
                
            elif reward < -400:
                collision += 1   
                print("GOAL:",goal,"  COLIISION:",collision)  
                
            if goal + collision >= 100:
                print("\n","ACCURACY:",goal/(goal + collision),"\n")
                break       
            
                
            if done:
                #agent.update_target_q_function()   
                #rospy.loginfo("UPDATE TARGET NETWORK")
        
                state_next = None
                #rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f' % (episode, score, len(agent.brain.memory), agent.brain.epsilon))
                #scores.append(score)
                #episodes.append(episode)
                state = env.reset()
                # print("Episode:",episode, "state:",state)
                state = torch.from_numpy(state).type(torch.FloatTensor)
                state = torch.unsqueeze(state, 0).to(device)
                
                #break
            
            
            #if time_out: 
            #    
            #    state_next = None
            #
            #    #agent.update_target_q_function()
            #    #rospy.loginfo("UPDATE TARGET NETWORK")
            #    
            #    rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f' % (episode, score, len(agent.brain.memory), agent.brain.epsilon))
            #    
            #    scores.append(score)
            #    #losses.append(agent.brain.loss)
            #    episodes.append(episode)
            #    
            #    result.data = [score, episode] 
            #    loss_result.data = [losses/agent.brain.episode_step, episode]
            #    pub_result.publish(result)
            #    pub_loss_result.publish(loss_result)
                
            #    #writer.add_scalar("score", score, episode)
            #    #writer.add_scalar("loss", losses/agent.brain.episode_step, episode)
                
                #state = env.reset()
                ## print("Episode:",episode, "state:",state)
                #state = torch.from_numpy(state).type(torch.FloatTensor)
                #state = torch.unsqueeze(state, 0).to(device)    
                
                
            #    break
            
        
        #agent.update_target_q_function()   
        #rospy.loginfo("UPDATE TARGET NETWORK")
        
        #writer.add_scalar("Score", score, episode)
        #writer.add_scalar("Loss", losses/agent.brain.episode_step, episode)
        
        #if agent.brain.epsilon > agent.brain.epsilon_min:
        #    agent.brain.epsilon *= agent.brain.epsilon_decay
            
        #if episode % 2 == 0:
        #    #agent.update_target_q_function()
        #    #rospy.loginfo("UPDATE TARGET NETWORK")
        #    with torch.no_grad():
        #        torch.save(agent.brain.model, agent.brain.dirPath + str(episode) + '.pt') 
                
        #elif episode % 4 == 0:
        #    agent.update_target_q_function()   
        #    rospy.loginfo("UPDATE TARGET NETWORK")
                
    #with torch.no_grad():        
    #    torch.save(agent.brain.model, agent.brain.dirPath + str(episode) + '.pt')        
    print("종료")           
    #writer.close()
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
