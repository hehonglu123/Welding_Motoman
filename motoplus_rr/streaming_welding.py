import numpy as np
import traceback, time, sys
from RobotRaconteur.Client import *
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt

# Adjust the connection URL to the driver
fronius_client = RRN.ConnectService('rr+tcp://192.168.55.10:60823?service=welder')

# Set the job number to use for this weld
fronius_client.job_number = 215
fronius_client.prepare_welder()

###MOTOPLUS RR CONNECTION

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
command_seqno = 0
# rate = RRN.CreateRate(125)

##########KINEMATICS 
robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
p_start=np.array([1630,-860,-260])
p_end=np.array([1630,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
base_layer_height=2
layer_height=1.0

for i in range(0,1):
    if i%2==0:
        p1=p_start+np.array([0,0,i*base_layer_height])
        p2=p_end+np.array([0,0,i*base_layer_height])
    else:
        p1=p_end+np.array([0,0,i*base_layer_height])
        p2=p_start+np.array([0,0,i*base_layer_height])

    v=20
    num_points=np.ceil(125*np.linalg.norm(p1-p2)/v)
    q_all=np.zeros((int(num_points),6))
    for j in range(int(num_points)):
        q_all[j]=robot.inv(p1*(num_points-j)/num_points+p2*j/num_points,R,q_seed)[0]
        


res, robot_state, _ = RR_robot_state.TryGetInValue()
rob1_pos=robot_state.joint_position[:6]
init_joint_pos=robot_state.joint_position

###JOG TO starting pose first
num_points_jogging=np.linalg.norm(rob1_pos-q_all[0])/0.001


for j in range(int(num_points_jogging)):

    rob1_target = (rob1_pos*(num_points_jogging-j))/num_points_jogging+q_all[0]*j/num_points_jogging


    # Retreive the current robot state
    res, robot_state, _ = RR_robot_state.TryGetInValue()


    # Increment command_seqno
    command_seqno += 1

    # Create Fill the RobotJointCommand structure
    joint_cmd1 = RobotJointCommand()
    joint_cmd1.seqno = command_seqno # Strictly increasing command_seqno
    joint_cmd1.state_seqno = robot_state.seqno # Send current robot_state.seqno as failsafe
    
    # Generate a joint command, in this case a sin wave
    cmd = np.zeros((14,))
    cmd += init_joint_pos
    cmd[:6]=rob1_target
    
    # Set the joint command
    joint_cmd1.command = cmd

    # Send the joint command to the robot
    RR_robot.position_command.PokeOutValue(joint_cmd1)

    # rate.Sleep()
    time.sleep(0.008)

###init point wait
for i in range(125):
    # Increment command_seqno
    command_seqno += 1

    # Create Fill the RobotJointCommand structure
    joint_cmd1 = RobotJointCommand()
    joint_cmd1.seqno = command_seqno # Strictly increasing command_seqno
    joint_cmd1.state_seqno = robot_state.seqno # Send current robot_state.seqno as failsafe
    
    # Generate a joint command, in this case a sin wave
    cmd = np.zeros((14,))
    cmd += init_joint_pos
    cmd[:6]=q_all[0]
    
    # Set the joint command
    joint_cmd1.command = cmd

    # Send the joint command to the robot
    RR_robot.position_command.PokeOutValue(joint_cmd1)

    # rate.Sleep()
    time.sleep(0.008)


# ###start welding
joint_recording=[]
timestamp_recording=[]
timestamp_cmd=[]

fronius_client.start_weld()
for j in range(int(num_points)):
    
    # Retreive the current robot state
    res, robot_state, _ = RR_robot_state.TryGetInValue()

    # Increment command_seqno
    command_seqno += 1

    # Create Fill the RobotJointCommand structure
    joint_cmd1 = RobotJointCommand()
    joint_cmd1.seqno = command_seqno # Strictly increasing command_seqno
    joint_cmd1.state_seqno = robot_state.seqno # Send current robot_state.seqno as failsafe
    
    # Generate a joint command, in this case a sin wave
    cmd = np.zeros((14,))
    cmd += init_joint_pos
    cmd[:6]=q_all[j]
    
    # Set the joint command
    joint_cmd1.command = cmd

    # Send the joint command to the robot
    RR_robot.position_command.PokeOutValue(joint_cmd1)
    
    # rate.Sleep()
    time.sleep(0.008)
    

    joint_recording.append(robot_state.joint_position[:6])
    timestamp_recording.append(float(robot_state.ts['microseconds'])/1e6)
    timestamp_cmd.append(time.time())

fronius_client.stop_weld()
fronius_client.release_welder()

curve_exe=robot.fwd(np.array(joint_recording)).p_all
timestamp_recording=np.array(timestamp_recording)
joint_recording=np.array(joint_recording)
timestamp_cmd=np.array(timestamp_cmd)

timestamp_cmd-=timestamp_cmd[0]
timestamp_recording-=timestamp_recording[0]
lam=calc_lam_cs(curve_exe)
speed=np.gradient(lam)/np.gradient(timestamp_recording)
speed_desired=np.gradient(calc_lam_js(q_all,robot))/np.gradient(timestamp_cmd)

plt.plot(lam,speed,label='v_exe')
plt.plot(lam,speed_desired,label='v_cmd')
plt.legend()
plt.show()

for i in range(6):
    plt.figure(i)
    plt.title('JOINT %i'%i)
    plt.plot(timestamp_cmd,q_all[:,i],label='cmd')
    plt.plot(timestamp_recording,joint_recording[:,i],label='exe')
    plt.legend()
    
plt.show()