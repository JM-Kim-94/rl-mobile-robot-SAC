#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64
# from cv_bridge import CvBridge, CvBridgeError

import torch
import time
import numpy as np
from LaneModel.model import parsingNet
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
from PIL import Image as IMG
import scipy.special


img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

tusimple_griding_num = 100
culane_griding_num = 200

tusimple_row_anchor = [64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

tusimple_cls_num_per_lane = 56
culane_cls_num_per_lane = 18

offset = 400
black_offset = np.zeros((480, offset, 3), dtype=np.uint8)

img_w = int(0.5*(640 + 2*offset))  # 0.5*(400+640+400) 720
img_h = int(0.5*(480))             # 0.5*(480)         240



def image_callback(msg):
    global image
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = cv2.hconcat([black_offset, image, black_offset])
    new_image = cv2.resize(new_image, dsize=(img_w,img_h),interpolation=cv2.INTER_LINEAR)
    lane_detect(new_image)



def lane_detect(img):
        
        
        middle_point = np.array([])
        middle = np.array([])
        middle_mean_x, middle_mean_y, middle_mean_x_old, middle_mean_y_old = 0,0,0,0

        t1 = time.time()

        img2 = IMG.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda() + 1
        out = net(x)

        col_sample = np.linspace(0, 800 - 1, tusimple_griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(tusimple_griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == tusimple_griding_num] = 0
        out_j = loc[:, 1:3]
        #out_j = loc
        for n in range(56):
            if np.dot(out_j[n,0], out_j[n,1]) != 0:
                middle_point = np.append(middle_point, 0.5*np.sum(out_j[n]))
        middle_point = middle_point.reshape(len(middle_point),1)

        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                               int(img_h * (tusimple_row_anchor[tusimple_cls_num_per_lane - 1 - k] / 288)) - 1)
                        cv2.circle(img, ppp, 2, (255, 0, 0), -1)
                        
         
        for k in range(len(middle_point)):
           p = (int(middle_point[k] * col_sample_w * img_w / 800) - 1,int(img_h * (tusimple_row_anchor[tusimple_cls_num_per_lane - 1 - k] / 288)) - 1)
           middle = np.append(middle, p)            
           cv2.circle(img, p, 2, (255, 0, 0), -1)  ## Uncomment to show total middle points.
        middle = middle.reshape(int(len(middle)/2),2)
        print("middle:", middle)
        
        if len(middle) != 0:
            middle_mean_x = int(np.sum(middle[:,0])/len(middle))
            middle_mean_y = int(np.sum(middle[:,1])/len(middle))           
            middle_mean_x_old = middle_mean_x
            middle_mean_y_old = middle_mean_y
        else:
            middle_mean_x = middle_mean_x_old
            middle_mean_y = middle_mean_y_old
        
        
        
        cv2.circle(img, (middle_mean_x, middle_mean_y), 2, (255, 0, 0), -1)
		
        error = int(img_w/2) - middle_mean_x
        print("error:", error)
        cv2.circle(img, (int(img_w/2), middle_mean_y), 2, (0, 0, 255), -1)
		
        cv2.line(img, (int(img_w/2), middle_mean_y), (middle_mean_x, middle_mean_y), (0, 255, 0), 1)
        
        cv2.putText(img, "error : %d" % error, (int(img_w/2), middle_mean_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
        
        
        
        #middle_ROI = middle[10:15]
        #middle_ROI_mean_x = int(np.sum(middle_ROI[:,0])/5)
        #middle_ROI_mean_y = int(np.sum(middle_ROI[:,1])/5)
        #cv2.circle(img, (middle_ROI_mean_x, middle_ROI_mean_y), 2, (255, 0, 0), -1)
		
        #error = int(img_w/2) - middle_ROI_mean_x
        #print("error:", error)
        #cv2.circle(img, (int(img_w/2), middle_ROI_mean_y), 2, (0, 0, 255), -1)
		
        #cv2.line(img, (int(img_w/2), middle_ROI_mean_y), (middle_ROI_mean_x, middle_ROI_mean_y), (0, 255, 0), 1)
        
        #cv2.putText(img, "error : %d" % error, (int(img_w/2), middle_ROI_mean_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
        
        #P_gain = 1.5
        #if error<200 and error>-200:
        #    twist.linear.x = 8
        #    twist.angular.z = float(error)*P_gain
        #    
        #elif error>200 or error<-200:
        #    twist.linear.x = 2
        #    twist.angular.z = 0
        
        #cmd_vel_pub.publish(twist)
        
        error_pub.publish(error)
		
		
		
        t2 = time.time()
        
        f = 1 / (t2 - t1)

        cv2.putText(img, "FPS : %0.2f" % f, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow("Lane Detection", img)

        cv2.waitKey(1)
        
        

if __name__ == "__main__" :

    
    torch.backends.cudnn.benchmark = True
    net = parsingNet(pretrained=False, backbone='18', cls_dim=(100+1, 56, 4), use_aux=False).cuda()

    state_dict = torch.load('/home/jm-kim/catkin_ws/src/myrobot/tusimple_18.pth', map_location='cuda')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    
    
    
    rospy.init_node("LaneDetection")
    topic_name = '/camera/rgb/image_raw'
    rospy.Subscriber(topic_name, Image, image_callback, queue_size=1, buff_size=52428800)
    
    #cmd_vel_pub = rospy.Publisher('cmd_vel',Twist, queue_size=1)
    error_pub = rospy.Publisher('error',Int64, queue_size=1)
    twist = Twist()
    
    rospy.spin()
    
    
