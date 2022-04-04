#!/usr/bin/env python3

import rospy
import numpy as np
import math
import os
import time
import random
from math import pi
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 7
        self.goal_y = 7
        self.heading = 0
        self.action_size = action_size
        #self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.line_error = 0
        self.yaw = 0.0
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        
        # 퍼블리셔
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        # 서브스크라이버
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_lin_err = rospy.Subscriber('error', Int64, self.getLineError)
        self.sub_scan = rospy.Subscriber('scan', LaserScan, self.getScanData) # 내가 만든거
        self.scan = LaserScan()
        
    def getScanData(self, scan_msg): # 내가 만든거 
        self.scan = scan_msg
        
    def getLineError(self, err):
        self.line_error = err.data

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw0 = euler_from_quaternion(orientation_list)
        self.yaw = yaw0
        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - self.yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi            
        

        self.heading = round(heading, 2)

    def getState(self):
        #scan_range = []
        #max_err = 200
        #line_error = 0
        heading = self.heading
        #min_range = 0.25
        done = False

        #for i in range(len(scan.ranges)):
        #    if scan.ranges[i] == float('Inf'):
        #        scan_range.append(3.5)
        #    elif np.isnan(scan.ranges[i]):
        #        scan_range.append(0)
        #    else:
        #        scan_range.append(scan.ranges[i])
                
        #if lin_err>max_err or lin_err<-max_err:
        #    line_error = lin_err # 나중에 적절하게 바꾸기 지금은 그냥 에러값 자체를 상태로. 
            #done = True
        #elif lin_err<=max_err and lin_err>=-max_err:
        #    line_error = lin_err
            
        #obstacle_min_range = round(min(scan_range), 2)
        #obstacle_angle = np.argmin(scan_range)
        #if min_range > min(scan_range) > 0:
        #    done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        print("current_distance:",current_distance, "  yaw:",self.yaw, "  heading:", heading)
        if current_distance < 0.5:
            self.get_goalbox = True
            done = True
        
        return [heading, current_distance], done

    def setReward(self, state, done, action):
        #yaw_reward = []
        #line_reward = 0
        #line_error = state[-5]
        current_distance = state[-1]
        heading = state[-2]

        #for i in range(7):
        #    angle = -pi / 4 + heading + (pi/12 * i) + pi / 2
        #    tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        #    yaw_reward.append(tr)

        #distance_rate = 2 ** (current_distance / self.goal_distance)
        #distance_rate = 2*(0.5 ** (current_distance / self.goal_distance)-0.5)
        #reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)
        
        #distance_reward = 2.5*(0.5 ** (current_distance / self.goal_distance)-0.6)
        distance_reward = -1.3 * (current_distance / self.goal_distance)+1
        angle_reward = -1.5 * (abs(heading)) + 1
        #(0.5**(abs(heading)) - 0.3) / 0.7
                
        #if -20<line_error<20 and -0.3<heading<0.3 : 
        #    line_reward = 1 - 0.05*abs(line_error)
        
        #time_reward = -0.5
        
        #reward = 2*(distance_reward + angle_reward) + line_reward
        #reward = distance_reward + angle_reward + line_reward + time_reward
        reward = distance_reward + angle_reward
        
        #if done:
        #    rospy.loginfo("Collision or out of road")
        #    reward = -10
        #    self.pub_cmd_vel.publish(Twist())

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = 7,7
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        
        #print("Reward : ", reward)
        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = (((self.action_size - 1)/2 - action) * max_angular_vel) / 3

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.5  # 원래는 0.15였음. 
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)
        
        print("Goal_x:",self.goal_x, "   Goal_y:",self.goal_y)

        #lidar_data = None
        ##error_data = None
        #while lidar_data is None:
        #    try:
        #        lidar_data = rospy.wait_for_message('scan', LaserScan, timeout=5)
        #        #error_data = rospy.wait_for_message('error', Int64, timeout=1)
        #        #error_data = (error_data.data)
        #    except:
        #        pass
        
        #error_data = self.line_error
        #state, done = self.getState(lidar_data, error_data)
        #state, done = self.getState(self.scan, error_data)
        state, done = self.getState()
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        
        try:
            self.reset_proxy()
            time.sleep(1)
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        #lidar_data = None
        ##error_data = None
        #while lidar_data is None:
        #    try:
        #        lidar_data = rospy.wait_for_message('scan', LaserScan, timeout=5)
        #        #error_data = rospy.wait_for_message('error', Int64, timeout=5)
        #        #error_data = (error_data).data
        #    except:
        #        pass
                
        error_data = self.line_error
        
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState()
        

        return np.asarray(state)
        
        
        
class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('/src/dqn', '/urdf/goal_box/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()
        self.stage = rospy.get_param('/stage_number')
        self.goal_position = Pose()
        self.init_goal_x = 1
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'
        self.obstacle_1 = 2, 2
        self.obstacle_2 = 2, -2
        self.obstacle_3 = -2, 2
        self.obstacle_4 = -2, -2
        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        self.check_model = False
        self.index = 0

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                
                break
            else:
                pass

    def deleteModel(self):
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass

    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 4:
            while position_check:
                goal_x = random.randrange(-37, 37) / 10.0 # stage_2_wide.world 
                goal_y = random.randrange(-37, 37) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.8 and abs(goal_y - self.obstacle_1[1]) <= 0.8:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.8 and abs(goal_y - self.obstacle_2[1]) <= 0.8:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.8 and abs(goal_y - self.obstacle_3[1]) <= 0.8:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.8 and abs(goal_y - self.obstacle_4[1]) <= 0.8:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.8 and abs(goal_y - 0.0) <= 0.8:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 13)
                print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawnModel()
        
        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y

        return self.goal_position.position.x, self.goal_position.position.y
        
     
