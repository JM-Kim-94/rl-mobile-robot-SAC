#!/usr/bin/env python3

import rospy
import time
import psutil
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray, Int32
from matplotlib.animation import FuncAnimation


class Visualiser:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ln, = plt.plot([], [], 'b')
        self.x_data, self.y_data = [] , []

    def plot_init(self):
        self.ax.set_xlim(0, 80)
        self.ax.set_ylim(-5, 20)
        return self.ln
    
    def getYaw(self, pose):
        quaternion = (pose.orientation.x, pose.orientation.y, pose.orientation.z,
                pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2] 
        return yaw   

    def callback(self, msg):
        y = (msg.data[0])
        self.y_data.append(y)
        x_index = msg.data[1]
        self.x_data.append(x_index)
    
    def update_plot(self, frame):
        self.ln.set_data(self.x_data, self.y_data)
        return self.ln


rospy.init_node('loss_graph_node')
vis = Visualiser()
sub = rospy.Subscriber('loss_result', Float32MultiArray, vis.callback)

ani = FuncAnimation(vis.fig, vis.update_plot, init_func=vis.plot_init)
#ani = FuncAnimation(vis.fig, vis.update_plot)
plt.show(block=True) 
    
    
    
    
    
    
    
    
    
    
    
    








