#!/usr/bin/env python3

import rospy
import numpy as np
import math
import os
import time
import random
from math import pi as PI
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Point, Pose
from std_msgs.msg import Float32
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        #37.632944285639674, 127.0767255448578 중도
        self.goal_x = 207126.59740132 # 청운관
        self.goal_y = 559308.55532458
        self.current_x = 0.0
        self.current_y = 0.0
        self.heading = 0
        self.action_size = action_size
        self.get_goalbox = False
        self.position = Pose()
        self.line_error = 0
        
        # 퍼블리셔
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        self.sub_GPS = rospy.Subscriber('GPS_pub', Twist, self.GPS_callback)
        self.sub_YAW = rospy.Subscriber('imu_yaw', Float32, self.IMU_callback)
        
        self.gps_data = Twist()
        self.yaw_data = 0.0
        
    def GPS_callback(self, gps_msg):
        self.gps_data = gps_msg
        self.current_x = gps_msg.linear.x
        self.current_y = gps_msg.linear.y
        
    def IMU_callback(self, imu_msg):
        self.yaw_data = imu_msg.data
        

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.current_x, self.goal_y - self.current_y), 2)

        return goal_distance
        

    def getState(self):
        done = False
        
        yaw = -(self.yaw_data)*PI/180
        goal_angle = math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)
        
        heading = goal_angle - yaw + PI/2
        #heading = goal_angle - yaw 
        #if heading > PI:
        #    heading -= 2 * PI
        #elif heading < -PI:
        #    heading += 2 * PI
        heading = round(heading, 2)       
        
        current_distance = round(math.hypot(self.goal_x - self.current_x, self.goal_y - self.current_y), 2)
        print("current_distance:",current_distance, "   yaw:",yaw, "   heading:",heading)
        
        if current_distance < 0.5:
            self.get_goalbox = True
            done = True

        return [heading, current_distance], done

    def setReward(self, state, done, action):
        yaw_reward = []
        current_distance = state[-1]
        heading = state[-2]
        
        distance_reward = -1.3 * (current_distance / self.goal_distance)+1
        angle_reward = (0.5**(abs(heading)) - 0.3) / 0.7
        
        reward = distance_reward + angle_reward
        
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = 207126.59740132, 559308.55532458
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        
        #print("Reward : ", reward)
        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = (((self.action_size - 1)/2 - action) * max_angular_vel) / 3

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.4  # 원래는 0.15였음. 
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        
        #print("Goal_x:",self.goal_x, "   Goal_y:",self.goal_y)
        
        state, done = self.getState()
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
            
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState()        

        return np.asarray(state)
        
        
     
     
     
     
     
     
     
     
     
     
     
     
     
     
