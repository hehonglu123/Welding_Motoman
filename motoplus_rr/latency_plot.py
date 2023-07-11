import numpy as np
import traceback, time, sys
from RobotRaconteur.Client import *
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt
from StreamingSend import *


RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
RR_robot = RR_robot_sub.GetDefaultClientWait(1)
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.reset_errors()
RR_robot.enable()
RR_robot.command_mode = halt_mode
time.sleep(0.1)
RR_robot.command_mode = position_mode
streaming_rate=125.
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=streaming_rate)

SS.jog2q(np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))

total_time=8
num_points=total_time*streaming_rate
timestamp_cmd=np.linspace(0,total_time,num_points)
q6=np.sin(np.linspace(0,4*np.pi,num_points))
curve_js_all=np.hstack((np.zeros((num_points,5)),q6.reshape((num_points,-1)),0.5*np.pi*np.ones((num_points,1)),np.zeros((num_points,5)),np.radians(-15)*np.ones((num_points,1)),np.pi*np.ones((num_points,1))))

timestamp_recording,joint_recording=SS.traj_streaming(curve_js_all[:int(len(curve_js_all)/2)])
plt.title('JOINT LATENCY PLOT')
plt.plot(timestamp_cmd,q6,label='cmd')
plt.plot(timestamp_recording,joint_recording[:,5],label='exe')
plt.legend()
    
plt.show()
