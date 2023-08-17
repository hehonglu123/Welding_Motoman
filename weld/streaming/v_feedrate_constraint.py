import numpy as np
import traceback, time, sys
from RobotRaconteur.Client import *
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt
from StreamingSend import *

####Adjust the connection URL to the driver
fronius_client = RRN.ConnectService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client.job_number = 200
fronius_client.prepare_welder()

###MOTOPLUS RR CONNECTION

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
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=125.)
# rate = RRN.CreateRate(125)

##########KINEMATICS 
robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
x_all=[1615,1630,1645,1660,1675,1690]
v_all=[10,20,30,40,50,60]
feedrate=150

for m in range(len(x_all)):
    p_start=np.array([x_all[m],-860,-260])
    p_end=np.array([x_all[m],-760,-260])
    q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
    layer_height=0.2
    
    q_all=[]
    for i in range(10):
        if i%2==0:
            p1=p_start+np.array([0,0,i*layer_height])
            p2=p_end+np.array([0,0,i*layer_height])
        else:
            p1=p_end+np.array([0,0,i*layer_height])
            p2=p_start+np.array([0,0,i*layer_height])

        if i==0:
            fronius_client.job_number=int(200/10+200)
            v=1
        else:
            fronius_client.job_number=int(feedrate/10+200)
            v=55

        num_points=int(np.ceil(125*np.linalg.norm(p2-p1)/v))
        
        for j in range(num_points):
            q_all.append(robot.inv(p1*(num_points-j)/num_points+p2*j/num_points,R,q_seed)[0])

        q_all=np.concatenate(q_all).reshape((-1,6))


        ###JOG TO starting pose first
        SS.jog2q(np.hstack((q_all[0],[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))


        fronius_client.start_weld()

        curve_js_all=np.hstack((q_all,0.5*np.pi*np.ones((len(q_all),1)),np.zeros((len(q_all),5)),np.radians(-15)*np.ones((len(q_all),1)),np.pi*np.ones((len(q_all),1))))
        timestamp_recording,joint_recording=SS.traj_streaming(q_all,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))


        fronius_client.stop_weld()
        fronius_client.release_welder()


