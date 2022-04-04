#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Gazebo Cam",img)
    cv2.waitKey(1)


class Detector():
    def __init__(self):
        self.br = CvBridge()
        rospy.Subscriber('/camera/rgb/image_raw', Image, self.img_callback)
        
    def img_callback(self, msg):
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        detect(im)


if __name__=='__main__':
    rospy.init_node("gazebocam")
    my_detector = Detector()
    rospy.spin()
    
    
    
    
    
    
    
    
    
