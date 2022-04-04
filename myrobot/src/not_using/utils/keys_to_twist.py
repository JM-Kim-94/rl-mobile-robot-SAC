#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

key_mapping = {'w':[0,1], 'x':[0,-1], 'a':[1,0], 'd':[-1,0], 's':[0,0]}

def keys_cb(msg, twist_pub):
    if len(msg.data) == 0 or (msg.data[0] not in key_mapping):
        return # 알려지지 않은 키
    vels = key_mapping[msg.data[0]]
    t = Twist()
    t.angular.z = vels[0]
    t.linear.x = vels[1]
    twist_pub.publish(t)
    
if __name__=='__main__':
    rospy.init_node('keys_to_twist')
    twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 1)
    rospy.Subscriber('keys', String, keys_cb, twist_pub)
    rospy.spin()












