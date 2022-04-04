#!/usr/bin/env python3

import serial
import time
import rospy
from std_msgs.msg import Float32
from math import pi as PI

global port
port = "/dev/ttyUSB0"
global yaw

yaw = 0.0
imu_yaw = 0
yaw_old = yaw
threshold = 0.08
loop_delay = 0.01
		



def getIMU():
    print('getIMU activated')
    global yaw
    global start_flag
    global imu_flag
    imu_flag = 1
    start_flag = 1
    serIMU = serial.Serial(port, 115200, timeout = 0)
    serIMU.write(b'<sor50>')
   

    while start_flag:
        pass

    while imu_flag:
        IMUdata = serIMU.readline().decode('ascii')
        idata = IMUdata.strip()
        idata = idata.strip('*')
        data = idata.split(',')
        try :
            if IMUdata != '':
                
                yaw_old = imu_yaw
                imu_yaw = float(data[2])
                d_yaw = imu_yaw - yaw_old

                if abs(d_yaw) < threshold:
                    d_yaw = 0
                yaw += d_yaw

                serIMU.reset_input_buffer()
        except:
            continue




    while True:
        IMUdata = serIMU.readline().decode('ascii')
        idata = IMUdata.strip()
        idata = idata.strip('*')
        data = idata.split(',')




        try :
            if IMUdata != '':
                
                yaw = float(data[2])
                serIMU.reset_input_buffer()
        except:
            continue

def getyaw():
    global yaw
    return yaw

def imu_start():
    global start_flag
    start_flag=0

def imu_check():
    global imu_flag
    imu_flag = 0


if __name__ == '__main__' :
    
    rospy.init_node('get_imu', anonymous=True)
    pub = rospy.Publisher('imu_yaw',Float32, queue_size=10)
    serIMU = serial.Serial(port, 115200, timeout = 0)
    serIMU.write(b'<sor50>')
    #rate = rospy.Rate(1)

    while not rospy.is_shutdown():					
        IMUdata = serIMU.readline().decode('ascii')
        idata = IMUdata.strip()
        idata = idata.strip('*')
        data = idata.split(',')
	

        try :
            if IMUdata != '':
                
                yaw_old = imu_yaw
                imu_yaw = float(data[2])
                d_yaw = imu_yaw - yaw_old

                if abs(d_yaw) < threshold:
                    d_yaw = 0
                yaw += d_yaw
                
                serIMU.reset_input_buffer()
                pub.publish(yaw)
                #rate.sleep()
            print(yaw)

        except:
            continue




