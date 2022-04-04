#!/usr/bin/env python3

import rospy
import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
import time
from std_msgs.msg import Float32MultiArray, Int32
from geometry_msgs.msg import Twist

result = Float32MultiArray()
R = Twist()


if __name__ == '__main__':
    rospy.init_node('publisher')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    
    result = Float32MultiArray()
    scores, episodes = [], []
    i = 1
    while True:
        result.data = [np.random.randint(10), i]
        #R.linear.x = np.random.randint(10)
        #R.linear.y = i
        print(result)
        pub_result.publish(result)
        
        i = i + 1
        time.sleep(0.1)








