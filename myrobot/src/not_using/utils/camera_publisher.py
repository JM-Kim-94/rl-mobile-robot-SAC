#!/usr/bin/env python3

import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time


bridge = CvBridge()


def publish_image(imgdata):
    
    image_message = bridge.cv2_to_imgmsg(imgdata, "bgr8")    
    image_pub.publish(image_message)
    print("pubulish")
    
    
if __name__ == '__main__':

    rospy.init_node('pub_cv2_camera', anonymous=True)
    image_pub = rospy.Publisher('image', Image, queue_size=1)

    cap = cv2.VideoCapture('/dev/video0')
    prevTime = 0
    
    while True:
        _, img = cap.read()
        
        curTime = time.time()
        fps = 1/(curTime - prevTime)
        prevTime = curTime
        
        publish_image(img)        
        
        cv2.putText(img, "FPS : %0.1f" % fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('publisher',img)           
        
        if cv2.waitKey(1) == 27:
            break
            
    cap.release() 
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
