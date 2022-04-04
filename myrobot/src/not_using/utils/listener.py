#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from math import pi as PI
    
gps_data = Twist()
yaw_data = 0.0


def GPS_callback(gps_msg):
    global gps_data
    gps_data = gps_msg
    
def IMU_callback(yaw_msg):
    global yaw_data
    yaw_data = yaw_msg.data
    
def main():
    global gps_data, yaw_data 
    
    x,y = gps_data.linear.x, gps_data.linear.y
    print("X:",x, "  Y:",y, "  YAW:",-yaw_data*PI/180)
    


if __name__ == '__main__':
    rospy.init_node('GPS_IMU_listener')

    rospy.Subscriber('GPS_pub', Twist, GPS_callback)
    rospy.Subscriber('imu_yaw', Float32, IMU_callback)
    rate = rospy.Rate(100)  

    
    
    while not rospy.is_shutdown():	
        main()
        rate.sleep()
    
    
    
    
    
    
