import numpy as np
import traceback, time, sys
from RobotRaconteur.Client import *
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt
from StreamingSend import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

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
SS=StreamingSend(robot,RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=streaming_rate)

SS.jog2q(np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))

total_time=20
num_points=int(total_time*streaming_rate)
timestamp_cmd=np.linspace(0,total_time,num_points)
q6=np.zeros(num_points)
q6[:num_points//4]=np.radians(0.5)*np.ones(num_points//4)
q6[num_points//4:num_points//2]=-np.radians(0.5)*np.ones(num_points//4)
q6[num_points//2:3*num_points//4]=np.radians(0.5)*np.ones(num_points//4)
q6[3*num_points//4:]=-np.radians(0.5)*np.ones(num_points//4)
# q6[::2]=np.radians(0.5)*np.ones(int(num_points/2))
# q6[1::2]=-np.radians(0.5)*np.ones(int(num_points/2))


timestamp_recording,joint_recording=SS.traj_streaming(q6.reshape((-1,1)),ctrl_joints=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0]))
# timestamp_recording,joint_recording=SS.traj_tracking_js(q6.reshape((len(q6),-1)),ctrl_joints=np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0]))

plt.title('JOINT6 LATENCY PLOT')
plt.plot(timestamp_recording,q6,label='cmd')
plt.plot(timestamp_recording,joint_recording,label='exe')
plt.xlabel('time (s)')
plt.ylabel('joint6 (rad)')
plt.legend()
    
plt.show()
