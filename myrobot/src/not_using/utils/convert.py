#!/usr/bin/env python3

import rospy
import time
import numpy as np
from std_msgs.msg import Float32, Int64
from geometry_msgs.msg import Twist
from math import pi as PI

wheel_diameter = 0.10122  # [m]
wheel_seperation = 0.316  # [m]

linear_vel_robot = 0.0
angular_vel_robot = 0.0
rpm_left = 0.0
rpm_right = 0.0

def cmd_vel_callback(cmd_vel_msg):
    global linear_vel_robot, angular_vel_robot
    
    linear_vel_robot = cmd_vel_msg.linear.x
    angular_vel_robot = cmd_vel_msg.angular.z
    
    
def main():
    global linear_vel_robot, angular_vel_robot
                   
    rpm_left = (linear_vel_robot - angular_vel_robot*wheel_seperation/2)*(60/(PI*wheel_diameter))  #[rpm]
    rpm_right = (linear_vel_robot + angular_vel_robot*wheel_seperation/2)*(60/(PI*wheel_diameter)) #[rpm]   
    
    motor_pid.linear.x = rpm_left 
    motor_pid.linear.y = rpm_right
    rpm_pub.publish(motor_pid)
    print("rpm_left: %0.3f" % rpm_left, "  rpm_right: %0.3f" % rpm_right)
    

if __name__=='__main__':
    rospy.init_node("convert")
    
    rate = rospy.Rate(100)
    
    rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback)    
    rpm_pub = rospy.Publisher('rpm', Twist, queue_size = 10)
    
    motor_pid = Twist()    
    
    while not rospy.is_shutdown():
        main()
        rate.sleep()




