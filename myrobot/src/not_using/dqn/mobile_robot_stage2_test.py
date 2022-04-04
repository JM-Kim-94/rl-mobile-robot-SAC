#!/usr/bin/env python3


import rospy
import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from collections import namedtuple
from std_msgs.msg import Float32MultiArray
from src.turtlebot3_dqn.environment_stage_2 import Env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

EPISODES = 3000


class DQN(nn.Module):
    def __init__(self, state_size, action_size): 
        super(DQN, self).__init__() 
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(self.state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.drp = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, self.action_size)  
            
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drp(x)
        x = self.fc3(x)
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
        self.pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
        self.dirPath = os.path.dirname(os.path.realpath(__file__))
        self.dirPath = self.dirPath.replace('turtlebot3_dqn/nodes', 'turtlebot3_dqn/save_model/stage_1_')
        self.result = Float32MultiArray()
        
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = ReplayMemory(10000)       

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        
        self.model = torch.load("/home/jm-kim/catkin_ws/src/turtlebot3/turtlebot3_machine_learning/turtlebot3_dqn/save_model/0507_stage2_wide_2_trial/stage_2_900.pt")
        
        print(self.model)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        
        if self.load_model:
            self.model.load(self.dirPath+str(self.load_episode)+".pt")

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
                
    def decide_action(self, state, episode):
    
        if self.epsilon <= np.random.rand():
            print("모델에 의한 행동선택")
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
                print("action : ", action.item())
        else:
            action = torch.LongTensor([[random.randrange(self.action_size)]]).to(device)
            print("무작위 행동선택 action : ", action.item())
        
        return action
        
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 
        
        self.mini_batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()
        
        self.expected_state_action_values = self.get_expected_state_action_values()
        
        self.update_q_network()
        
        
    def make_minibatch(self):
        transitions = self.memory.sample(self.batch_size)
        mini_batch = Transition(*zip(*transitions))
        print("메모리에서 랜덤 샘플")
        
        state_batch = torch.cat(mini_batch.state)
        action_batch = torch.cat(mini_batch.action)
        reward_batch = torch.cat(mini_batch.reward)
        non_final_next_states = torch.cat([s for s in mini_batch.next_state if s is not None])
        
        return mini_batch, state_batch, action_batch, reward_batch, non_final_next_states
        
        
    def get_expected_state_action_values(self):
    
        self.model.eval()
        self.target_model.eval()
        
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
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        
        print("모델 훈련")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_q_network(self):
        print("타겟모델 업데이트")
        self.target_model.load_state_dict(self.model.state_dict())       
            
        
    
                

class Agent():
    def __init__(self, state_size, action_size):
        self.brain = Brain(state_size, action_size)
        
    def update_q_function(self):
        self.brain.replay()
        
    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action
        
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)        
        
    def update_target_q_function(self):
        self.brain.update_target_q_network()
                



if __name__ == '__main__':
    rospy.init_node('mobile_robot_dqn_stage2_test')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()

    state_size = 184
    action_size = 5

    env = Env(action_size)
    
    agent = Agent(state_size, action_size)
    
    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    
        
    for episode in range(agent.brain.load_episode + 1, EPISODES):
        print("Episode:",episode)
        done = False
        
        state = env.reset()
        # print("Episode:",episode, "state:",state)
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = torch.unsqueeze(state, 0).to(device)
        
        score = 0
        
        for t in range(agent.brain.episode_step):
            action = agent.get_action(state, episode)
            print("step: ", t, "   action: ", action.item())

            observation_next, reward, done = env.step(action.item())
            reward = torch.tensor([reward]).to(device)  # torch.from_numpy(reward).type(torch.FloatTensor)
            state_next = observation_next
            state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
            state_next = torch.unsqueeze(state_next, 0).to(device)
            
            #agent.memorize(state, action, state_next, reward)
            
            #agent.update_q_function()
            
            state = state_next
            
            score += reward
            
            #if episode % 10 == 0:
            #    torch.save(agent.brain.model, agent.brain.dirPath + str(episode) + '.pt')
            #    with open(agent.brain.dirPath + str(episode) + '.json', 'w') as outfile:
            #        json.dump(param_dictionary, outfile)
                    
            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True
            
            
            if done: 
                #result.data = [score, np.max(agent.q_value)]
                #pub_result.publish(result)
                #agent.update_target_q_function()
                state_next = None
                scores.append(score)
                episodes.append(episode)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d', episode, score, len(agent.brain.memory), agent.brain.epsilon, h, m, s)
                #param_keys = ['epsilon']
                #param_values = [agent.brain.epsilon]
                #param_dictionary = dict(zip(param_keys, param_values))
                break
                
            global_step += 1
            print("글로벌스텝:",global_step)
            
        #    if global_step % agent.brain.target_update == 0:
        #        rospy.loginfo("UPDATE TARGET NETWORK")
            
        #if agent.brain.epsilon > agent.brain.epsilon_min:
        #    agent.brain.epsilon *= agent.brain.epsilon_decay
            
            
            
          

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
