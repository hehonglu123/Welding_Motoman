from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import pickle
from MocapPoseListener import *

dataset_date = '0725'

config_dir='../config/'
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA1440_'+dataset_date+'_marker_config.yaml',tool_marker_config_file=config_dir+'mti_'+dataset_date+'_marker_config.yaml')

test_qs = []
sample_q = np.radians([[37,35,-2,19,-53,-2],[-34,40,0,-15,-49,44],\
                       [0,-50,-50,0,-22,0],[0,0,0,0,0,0],\
                       [0,45,20,0,22,0],[37,35,-2,19,-53,-2],[0,0,0,0,0,0]])
sample_N = [369,238,193,203,264,233] # len(sample_q)-1
# sample_N = [2,2,2,2,2,2] # len(sample_q)-1
for i in range(len(sample_N)):
    start_T = robot_scan.fwd(sample_q[i])
    end_T = robot_scan.fwd(sample_q[i+1])
    k,dtheta = R2rot(np.matmul(start_T.R.T,end_T.R))
    dp_vector = end_T.p-start_T.p
    for n in range(sample_N[i]):
        this_R=np.matmul(start_T.R,rot(k,dtheta/sample_N[i]*n))
        this_p=start_T.p+dp_vector/sample_N[i]*n
        if n==0:
            this_q=deepcopy(sample_q[i])
        else:
            this_q=robot_scan.inv(this_p,this_R,last_joints=sample_q[i])[0]
        test_qs.append(np.round(np.degrees(this_q),4))

# test_qs=[np.zeros(6),np.zeros(6)+2,np.zeros(6)+4,np.zeros(6)+6]

# print(np.array(test_qs))
print(dataset_date)
print(len(test_qs))
# exit()

# mocap pose listener
mocap_url = 'rr+tcp://localhost:59823?service=optitrack_mocap'
mocap_url = mocap_url
mocap_cli = RRN.ConnectService(mocap_url)
all_ids=[]
all_ids.extend(robot_scan.tool_markers_id)
all_ids.extend(robot_scan.base_markers_id)
all_ids.append(robot_scan.base_rigid_id)
all_ids.append(robot_scan.tool_rigid_id)
mpl_obj = MocapFrameListener(mocap_cli,all_ids,'world',use_quat=True)

data_dir = 'kinematic_raw_data/'

repeats_N = 1
rob_speed = 1
waitTime = 0.75

robot_client = MotionProgramExecClient()
mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
for N in range(repeats_N):
    for test_q in test_qs:
        # move robot
        mp.MoveJ(test_q,rob_speed,0)
        mp.setWaitTime(waitTime)

robot_client.execute_motion_program_nonblocking(mp)
###streaming
robot_client.StartStreaming()
start_time=time.time()

state_flag=0
robot_q_align=[]
mocap_T_align=[]

robot_q_raw=[]
tool_T_raw=[]
base_T_raw=[]

joint_recording=[]
robot_stamps=[]
r_pulse2deg = robot_scan.pulse2deg
T_base_basemarker = robot_scan.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()
while True:
    if state_flag & 0x08 == 0 and time.time()-start_time>1.:
        break
    res, data = robot_client.receive_from_robot(0.01)
    if res:
        state_flag=data[16]

        # print(data[18])
        # if data[18]!=0 and data[18]%2==0:
        #     print(time.time()-start_time)

        # print(np.divide(np.array(data[20:26]),r_pulse2deg))
        # print("================")
        
        if data[18]!=0 and data[18]%2==0: # when the robot stop
            if len(joint_recording)==0:
                mpl_obj.run_pose_listener()
            joint_angle=np.radians(np.divide(np.array(data[26:32]),r_pulse2deg))
            joint_recording.append(joint_angle)
            timestamp=data[0]+data[1]*1e-9
            robot_stamps.append(timestamp)
        else:
            if len(joint_recording)>0:

                robot_stamps=np.array(robot_stamps)
                joint_recording=np.array(joint_recording)
                mpl_obj.stop_pose_listener()
                mocap_curve_p,mocap_curve_R,mocap_timestamps = mpl_obj.get_frames_traj()

                print("# of Mocap Data:",len(mocap_timestamps[robot_scan.base_rigid_id]))
                print("# of Robot Data:",len(robot_stamps))

                start_i = np.argmin(np.fabs(mocap_timestamps[robot_scan.base_rigid_id]-(mocap_timestamps[robot_scan.base_rigid_id][0]+waitTime/5)))
                end_i = np.argmin(np.fabs(mocap_timestamps[robot_scan.base_rigid_id]-(mocap_timestamps[robot_scan.base_rigid_id][0]+waitTime/5*4)))
                print("# of Mocap Data Used:",end_i-start_i)
                this_mocap_ori = []
                this_mocap_p = []
                base_rigid_R=mocap_curve_R[robot_scan.base_rigid_id]
                mocap_R=mocap_curve_R[robot_scan.tool_rigid_id]
                base_rigid_p=mocap_curve_p[robot_scan.base_rigid_id]
                mocap_p=mocap_curve_p[robot_scan.tool_rigid_id]
                for k in range(start_i,end_i):
                    tool_T_raw.append(np.append(mocap_p[k],mocap_R[k]))
                    base_T_raw.append(np.append(base_rigid_p[k],base_rigid_R[k]))

                    T_mocap_basemarker = Transform(q2R(base_rigid_R[k]),base_rigid_p[k]).inv()
                    T_marker_mocap = Transform(q2R(mocap_R[k]),mocap_p[k])
                    T_marker_basemarker = T_mocap_basemarker*T_marker_mocap
                    T_marker_base = T_basemarker_base*T_marker_basemarker
                    this_mocap_ori.append(R2rpy(T_marker_base.R))
                    this_mocap_p.append(T_marker_base.p)
                this_mocap_p = np.mean(this_mocap_p,axis=0)
                this_mocap_ori = R2q(rpy2R(np.mean(this_mocap_ori,axis=0)))
                mocap_T_align.append(np.append(this_mocap_p,this_mocap_ori))

                start_i = np.argmin(np.fabs(robot_stamps-(robot_stamps[0]+waitTime/5)))
                end_i = np.argmin(np.fabs(robot_stamps-(robot_stamps[0]+waitTime/5*4)))
                print("# of Robot Data Used:",end_i-start_i)
                joint_recording = joint_recording[start_i:end_i]
                robot_stamps = robot_stamps[start_i:end_i]
                robot_q_align.append(np.mean(joint_recording,axis=0))
                joint_recording=[]
                robot_stamps=[]
                mpl_obj.clear_traj()

                print("Q align num:",len(robot_q_align))
                print("mocap align num:",len(mocap_T_align))
                print("mocap tool raw num:",len(tool_T_raw))
                print("mocap base raw num:",len(base_T_raw))
                print("=========================")

robot_client.servoMH(False)

np.savetxt(data_dir+'robot_q_align.csv',robot_q_align,delimiter=',')
np.savetxt(data_dir+'mocap_T_align.csv',mocap_T_align,delimiter=',')
np.savetxt(data_dir+'mocap_tool_T_raw.csv',tool_T_raw,delimiter=',')
np.savetxt(data_dir+'mocap_base_T_raw.csv',base_T_raw,delimiter=',')

print("Q align num:",len(robot_q_align))
print("mocap align num:",len(mocap_T_align))
print("Tool T raw num:",len(tool_T_raw))
print("Base T raw num:",len(base_T_raw))

exit()

robot_scan.robot.P = robot_scan.calib_P
robot_scan.robot.H = robot_scan.calib_H
robot_scan.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
#### using tool
robot_scan.robot.R_tool = robot_scan.T_tool_toolmarker.R
robot_scan.robot.p_tool = robot_scan.T_tool_toolmarker.p

for i in range(0,len(robot_q_align),int(len(robot_q_align)/4)):
    rob_T = robot_scan.fwd(robot_q_align[i])
    print(rob_T)
    mT = mocap_T_align[i]
    print(Transform(q2R(mT[3:]),mT[:3]))
    print("=========================")