from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from PH_interp import *
from dx200_motion_program_exec_client import *
from WeldSend import *

############## Robot definition ##############
config_dir='../config/'
# robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
#     pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
#     base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
# robot_2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
#                         pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')

ph_dataset_date='0801'
test_dataset_date='0801'
robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'torch.csv',d=15,\
                    #  tool_file_path='',d=0,\
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')
ph_dataset_date='0804'
test_dataset_date='0804'
robot_marker_dir=config_dir+'MA1440_marker_config/'
tool_marker_dir=config_dir+'mti_marker_config/'
robot_2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'mti.csv',\
                    pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA1440_'+ph_dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'mti_'+ph_dataset_date+'_marker_config.yaml',\
                    base_transformation_file=config_dir+'MA1440_pose.csv')

### Nominal PH
nom_P_r1=np.array([[0,0,0],[150,0,0],[0,0,760],\
                [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H_r1=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
nom_P_r2=np.array([[0,0,0],[155,0,0],[0,0,614],\
                [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H_r2=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T

robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)

def get_joint_angle():
    for attempt_i in range(10):
        res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
        if res:
            joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position))
            return joint_angle
    return None

####### PH Parameters #######
calib_file_name = 'calib_PH_q_ana.pickle'
PH_r1_data_dir='PH_grad_data/test0801_R1/train_data_'
PH_r2_data_dir='PH_grad_data/test0804_R2/train_data_'
with open(PH_r1_data_dir+calib_file_name,'rb') as file:
    PH_q_r1=pickle.load(file)
ph_param_fbf_r1=PH_Param(nom_P_r1,nom_H_r1)
ph_param_fbf_r1.fit(PH_q_r1,method='FBF')
with open(PH_r2_data_dir+calib_file_name,'rb') as file:
    PH_q_r2=pickle.load(file)
ph_param_fbf_r2=PH_Param(nom_P_r2,nom_H_r2)
ph_param_fbf_r2.fit(PH_q_r2,method='FBF')

# try:
#     tool_calib_joints = np.radians(np.loadtxt('calib_data/tool_calib_joints.csv', delimiter=','))
# except FileNotFoundError:
#     tool_calib_joints = []
# try:
#     reference_joints = np.radians(np.loadtxt('calib_data/reference_joints.csv', delimiter=','))
#     zero_position_joints = reference_joints[0]
#     r1_out_r2_inward_joints = reference_joints[1]
#     r1_inward_r2_out_joints = reference_joints[2]
# except FileNotFoundError:
#     zero_position_joints = None
#     r1_out_r2_inward_joints = None
#     r1_inward_r2_out_joints = None
# while True:
#     option_chosen = input("Choose an option:\n1. Tool Calibration Joints\n\
# 2. Zero Position Joints\n\
# 3. R1 Out R2 Inward\n\
# 4. R1 Inward R2 Out\n\
# 5. Exit\n")
#     this_joint = get_joint_angle()
#     if this_joint is None:
#         print("Failed to get joint angles. Retrying...")
#         continue
#     if option_chosen == '1':
#         tool_calib_joints.append(deepcopy(this_joint))
#         print("Get tool calibration joints: ", np.degrees(tool_calib_joints))
#     elif option_chosen == '2':
#         zero_position_joints = deepcopy(this_joint)
#         print("Get zero position joints: ", np.degrees(zero_position_joints))
#     elif option_chosen == '3':
#         r1_out_r2_inward_joints = deepcopy(this_joint)
#         print("Get R1 Out R2 Inward joints: ", np.degrees(r1_out_r2_inward_joints))
#     elif option_chosen == '4':
#         r1_inward_r2_out_joints = deepcopy(this_joint)
#         print("Get R1 Inward R2 Out joints: ", np.degrees(r1_inward_r2_out_joints))
#     elif option_chosen == '5':
#         print("Exiting...")
#         break

#     if len(tool_calib_joints) > 0 and zero_position_joints is not None and \
#          r1_out_r2_inward_joints is not None and r1_inward_r2_out_joints is not None:
#         np.savetxt('calib_data/tool_calib_joints.csv', np.degrees(tool_calib_joints), delimiter=',')
#         np.savetxt('calib_data/reference_joints.csv', np.vstack((np.degrees(zero_position_joints), np.degrees(r1_out_r2_inward_joints), np.degrees(r1_inward_r2_out_joints))), delimiter=',')

tool_calib_joints = np.radians(np.loadtxt('calib_data/tool_calib_joints.csv', delimiter=','))
reference_joints = np.radians(np.loadtxt('calib_data/reference_joints.csv', delimiter=','))
# print("Tool calibration joints: ", np.degrees(tool_calib_joints))
# print("Zero position joints: ", np.degrees(zero_position_joints))
# print("R1 Out R2 Inward joints: ", np.degrees(r1_out_r2_inward_joints))
# print("R1 Inward R2 Out joints: ", np.degrees(r1_inward_r2_out_joints))

def get_residual_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.sqrt(np.mean(errors**2))
def get_mean_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.mean(errors)
def get_std_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.std(errors)
def get_max_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.max(errors)

origin_p_tool = deepcopy(robot_1.p_tool)
origin_R_tool = deepcopy(robot_1.R_tool)
origin_P_R1 = deepcopy(robot_1.robot.P)
origin_H_R1 = deepcopy(robot_1.robot.H)
origin_P_R2 = deepcopy(robot_2.robot.P)
origin_H_R2 = deepcopy(robot_2.robot.H)

# for use_cdc in [False, True]:
for methods in ['nominal','CPA','CDC']:
    use_cdc = True if methods == 'CDC' else False

    if methods == 'nominal' or methods == 'CDC':
        robot_1.robot.P = deepcopy(origin_P_R1)
        robot_1.robot.H = deepcopy(origin_H_R1)
        robot_2.robot.P = deepcopy(origin_P_R2)
        robot_2.robot.H = deepcopy(origin_H_R2)
    else:
        robot_1.robot.P = deepcopy(robot_1.calib_P)
        robot_1.robot.H = deepcopy(robot_1.calib_H)
        robot_2.robot.P = deepcopy(robot_2.calib_P)
        robot_2.robot.H = deepcopy(robot_2.calib_H)

    # calibrate tool
    # get flange

    robot_1.p_tool = np.zeros(3)
    robot_1.R_tool = np.eye(3)
    robot_1.robot.p_tool = np.zeros(3)
    robot_1.robot.R_tool = np.eye(3)
    ###
    num_js = len(tool_calib_joints)
    robot_Ts=[]
    robot_ps = []
    for i in range(num_js):
        q=tool_calib_joints[i][:6]
        # robot_T=robot.fwd_ph(q,ph_param)
        if use_cdc:
            robot_T=robot_1.fwd_ph(q,ph_param_fbf_r1)
        else:
            robot_T=robot_1.fwd(q)
        robot_Ts.append(H_from_RT(robot_T.R,robot_T.p))
        robot_ps.append(robot_T.p)
    # print("Residual tool position:", get_residual_error(robot_ps))
    # print("==============")

    A=[]
    b=[]

    # num_js=7
    for i in range(num_js-1):
        A.extend(robot_Ts[i][:3,:3]-robot_Ts[i+1][:3,:3])
        b.extend(robot_Ts[i+1][:3,-1]-robot_Ts[i][:3,-1])
        # b.extend(np.zeros(3))  # assume no translation change
    # print("A",A, "b",b)

    p_tool=np.linalg.pinv(A)@b
    # print(p_tool)
    # find null space of A
    # p_tool=np.linalg.lstsq(A, b, rcond=None)[0]

    # use the calibrated tool position
    robot_1.p_tool = deepcopy(p_tool)
    robot_1.R_tool = deepcopy(origin_R_tool)
    robot_1.robot.p_tool = deepcopy(p_tool)
    robot_1.robot.R_tool = deepcopy(origin_R_tool)
    robot_ps = []
    for i in range(num_js):
        q=tool_calib_joints[i][:6]
        # robot_T=robot.fwd_ph(q,ph_param)
        if use_cdc:
            robot_T=robot_1.fwd_ph(q,ph_param_fbf_r1)
        else:
            robot_T=robot_1.fwd(q)
        robot_ps.append(robot_T.p)
        robot_Ts.append(H_from_RT(robot_T.R,robot_T.p))
    # print("Residual tool position after calibration:", get_residual_error(robot_ps))

    reference_ps = []
    for joint_i, r_joint in enumerate(reference_joints):
        if joint_i==1:
            continue
        if use_cdc:
            t2 = robot_2.fwd_ph(r_joint[6:12], ph_param_fbf_r2, world=True)
            t1 = robot_1.fwd_ph(r_joint[0:6], ph_param_fbf_r1, world=True)
        else:
            t2 = robot_2.fwd(r_joint[6:12], world=True)
            t1 = robot_1.fwd(r_joint[0:6], world=True)
        t1_t2 = t2.inv() * t1
        # print("Relative tool position:", t1_t2.p)
        reference_ps.append(t1_t2.p)

    print("Method:", methods)
    print("Residual tool position error:", get_residual_error(reference_ps))
    print("Mean tool error:", get_mean_error(reference_ps))
    print("Standard deviation tool error:", get_std_error(reference_ps))
    print("Max tool error:", get_max_error(reference_ps))
    print("===================================")