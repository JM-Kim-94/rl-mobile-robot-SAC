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

from tf.transformations import euler_from_quaternion, quaternion_from_euler



def getscandata(scan_msg):
    #print(odom_msg.pose.pose.orientation)
    #print("Scan:",len(scan_msg.ranges))
    print((scan_msg.ranges).index(min(scan_msg.ranges)))
        

if __name__=='__main__':
    rospy.init_node('subs_odom')
    rospy.Subscriber('scan', LaserScan, getscandata)
    
    rospy.spin()



    






