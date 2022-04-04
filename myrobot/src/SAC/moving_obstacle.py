#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState, ModelStates
import random
import numpy as np
from math import *


class Move:
    def __init__(self):
        
        self.sub_models = rospy.Subscriber('gazebo/model_states', ModelStates, self.callback_sub_models)
        self.models = ModelStates()
        
        self.pub_model  = rospy.Publisher('gazebo/set_model_state',  ModelState, queue_size=10)
        
    def callback_sub_models(self, msg):
        self.models = msg

 

if __name__ == '__main__':
    rospy.init_node('moving_obstacles')
    
    moving = Move()
    
    #sim_rate = rospy.Rate(200)
    
    fire_1, fire_2 = ModelState(), ModelState()
    fire_3, fire_4 = ModelState(), ModelState()
    
    time.sleep(3)
    
    dist = 2.5
    vel  = 1
    
    while not rospy.is_shutdown():
    
        models = moving.models
        
        for i in range(len(models.name)):
            if models.name[i] == 'fire_hydrant_set_1':
                fire_1.model_name = 'fire_hydrant_set_1'
                fire_1.pose = models.pose[i]
                
                fire_1.pose.position.x = cos(time.time()/vel) + dist
                fire_1.pose.position.y = sin(time.time()/vel) + dist
                
                moving.pub_model.publish(fire_1)
            
            if models.name[i] == 'fire_hydrant_set_2':
                fire_2.model_name = 'fire_hydrant_set_2'
                fire_2.pose = models.pose[i]
                
                fire_2.pose.position.x = sin(time.time()/vel) + dist
                fire_2.pose.position.y = cos(time.time()/vel) - dist
                
                moving.pub_model.publish(fire_2)
                
            if models.name[i] == 'fire_hydrant_set_3':
                fire_3.model_name = 'fire_hydrant_set_3'
                fire_3.pose = models.pose[i]
                
                fire_3.pose.position.x = cos(time.time()/vel) - dist
                fire_3.pose.position.y = sin(time.time()/vel) - dist
                
                moving.pub_model.publish(fire_3)
                
            if models.name[i] == 'fire_hydrant_set_4':
                fire_4.model_name = 'fire_hydrant_set_4'
                fire_4.pose = models.pose[i]
                
                fire_4.pose.position.x = sin(time.time()/vel) - dist
                fire_4.pose.position.y = cos(time.time()/vel) + dist
                
                moving.pub_model.publish(fire_4)
        
        
        
        
        #sim_rate.sleep()
    
    
    
    
    
    
    
    
    
