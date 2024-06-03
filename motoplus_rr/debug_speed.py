import sys, time
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../toolbox/')
from StreamingSend import *


########################################################Robot CONFIG########################################################
robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../../config/MA1440_A0_robot_default_config.yml',tool_file_path='../../config/flir.csv',\
		pulse2deg_file_path='../../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../../config/MA1440_pose_mocap.csv')
positioner=positioner_obj('D500B',def_path='../../config/D500B_robot_default_config.yml',tool_file_path='../../config/positioner_tcp.csv',\
	pulse2deg_file_path='../../config/D500B_pulse2deg_real.csv',base_transformation_file='../../config/D500B_pose.csv')

########################################################RR STREAMING########################################################

# RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.10:59945?service=robot')
RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
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
point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)



p_start=np.array([1700,-780,-260])
p_end=np.array([1600,-780,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])
v=10.
num_points=int(streaming_rate*np.linalg.norm(p_end-p_start)/v)
curve=np.linspace(p_start,p_end,num_points)
curve_js=robot.find_curve_js(curve,[R]*num_points,q_seed)

q_all=[]
ts_all=[]

SS.jog2q(np.hstack((curve_js[0],[0,0,0,0,0,0,np.radians(-15),np.pi])))
for i in range(num_points):
    point_stream_start_time=time.time()
    robot_timestamp,q14=SS.position_cmd(np.hstack((curve_js[i],[0,0,0,0,0,0,np.radians(-15),np.pi])),time.time())
    q_all.append(q14)
    ts_all.append(robot_timestamp)

p_all=robot.fwd(q_all).p_all
#plot out velocity with ts_all timestamp
v_all=np.linalg.norm(np.diff(p_all,axis=0),axis=1)/(np.diff(ts_all))
plt.plot(ts_all[1:],v_all)
plt.show()


