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
p_start=np.array([1645,-860,-260])
p_end=np.array([1645,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
layer_height=0.2
feedrate=150
fronius_client.job_number=int(feedrate/10+200)
q_all=[]
for i in range(2,10):
    if i%2==0:
        p1=p_start+np.array([0,0,i*layer_height])
        p2=p_end+np.array([0,0,i*layer_height])
    else:
        p1=p_end+np.array([0,0,i*layer_height])
        p2=p_start+np.array([0,0,i*layer_height])

    v=55
    v1=v
    v2=v
    p_mid1=p1+5*(p2-p1)/np.linalg.norm(p2-p1)
    num_points1=int(np.ceil(125*np.linalg.norm(p1-p_mid1)/v1))
    num_points2=int(np.ceil(125*np.linalg.norm(p2-p_mid1)/v2))
    
    for j in range(num_points1):
        q_all.append(robot.inv(p1*(num_points1-j)/num_points1+p_mid1*j/num_points1,R,q_seed)[0])
    for j in range(num_points2):
        q_all.append(robot.inv(p_mid1*(num_points2-j)/num_points2+p2*j/num_points2,R,q_seed)[0])
        
q_all=np.concatenate(q_all).reshape((-1,6))

res, robot_state, _ = RR_robot_state.TryGetInValue()
rob1_pos=robot_state.joint_position[:6]
init_joint_pos=robot_state.joint_position

###JOG TO starting pose first
SS.jog2q(np.hstack((q_all[0],[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))


# ###start welding
joint_recording=[]
timestamp_recording=[]

fronius_client.start_weld()

curve_js_all=np.hstack((q_all,0.5*np.pi*np.ones((len(q_all),1)),np.zeros((len(q_all),5)),np.radians(-15)*np.ones((len(q_all),1)),np.pi*np.ones((len(q_all),1))))
# timestamp_recording,joint_recording=SS.traj_streaming(q_all[:int(len(curve_js_all)/2)],ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))
# # fronius_client.job_number = 200
# timestamp_recording,joint_recording=SS.traj_streaming(q_all[int(len(curve_js_all)/2):],ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))
timestamp_recording,joint_recording=SS.traj_streaming(q_all,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))


fronius_client.stop_weld()
fronius_client.release_welder()




# curve_exe=robot.fwd(np.array(joint_recording[:,:6])).p_all
# timestamp_recording=np.array(timestamp_recording)
# joint_recording=np.array(joint_recording)

# timestamp_recording-=timestamp_recording[0]
# lam=calc_lam_cs(curve_exe)
# speed=np.gradient(lam)/np.gradient(timestamp_recording)

# plt.plot(lam,speed,label='v_exe')
# plt.legend()
# plt.show()

# for i in range(6):
#     plt.figure(i)
#     plt.title('JOINT %i'%i)
#     plt.plot(timestamp_cmd,q_all[:,i],label='cmd')
#     plt.plot(timestamp_recording,joint_recording[:,i],label='exe')
#     plt.legend()
    
# plt.show()