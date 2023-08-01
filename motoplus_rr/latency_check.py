import numpy as np
import traceback, time, sys, copy
from RobotRaconteur.Client import *
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt
from StreamingSend import *

timestamp=0
joint_reading=[0]*14
def robot_state_cb(sub, value, ts):
	global timestamp,joint_reading
	joint_reading=value.joint_position
	timestamp=value.ts['microseconds'][0]/1e6


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
RR_robot_state.WireValueChanged += robot_state_cb		###CALLBACK HANDLER
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

q_start=np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi]))
q_end=np.hstack((np.zeros(5),np.radians([5]),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi]))

latency=[]
while True:
	try:
		SS.jog2q(q_start)
		time.sleep(1)
		now=time.time()
		while np.linalg.norm(joint_reading[5]-q_end[5])>0.001:	###reached end
			if np.linalg.norm(joint_reading[5]-q_start[5])>0.001:	###start moving
				latency.append(time.time()-now)
				print('latency: ',latency[-1])
				break
			SS.position_cmd(q_end)
	except:
		break
print('average,std,min,max all latency: ',np.mean(latency),np.std(latency),np.min(latency),np.max(latency))

# while True:
# 	try:
# 		SS.jog2q(q_start)
# 		time.sleep(1)
# 		now=copy.deepcopy(timestamp)
# 		while np.linalg.norm(joint_reading[5]-q_end[5])>0.001:	###before reached end
# 			if np.linalg.norm(joint_reading[5]-q_start[5])>0.001:	###if start moving, record latency, then break & loop
# 				latency.append(timestamp-now)
# 				print('latency: ',latency[-1])
# 				break
# 			SS.position_cmd(q_end)
# 	except:
# 		break
# print('average,std,min,max controller latency: ',np.mean(latency),np.std(latency),np.min(latency),np.max(latency))
