from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *

from general_robotics_toolbox import *
import numpy as np
import time
import yaml
from PH_interp import *
from calib_analytic_grad import *

# random seed for reproducibility
np.random.seed(0)

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

def PH_to_frame(P: np.ndarray, H: np.ndarray, tool: Transform, frame: Transform):

    # convert joint axis direction to the inertial frame
    H_new = []
    for h in H.T:
        h_new = frame.R @ h
        H_new.append(h_new)
    H_new = np.array(H_new).T
    # convert link axis vector to the inertial frame
    P_new = []
    for p in P.T:
        p_new = frame.R @ p
        P_new.append(p_new)
    P_new[0] += frame.p
    P_new = np.array(P_new).T
    # convert the tool transformation to the inertial frame
    tool_new = Transform(frame.R @ tool.R, frame.R @ tool.p)
    return P_new, H_new, tool_new

def convert_PH_to_inertial_frame(robot: robot_obj):
    
    base_T = Transform(robot.base_H[:3,:3], robot.base_H[:3,3]) 
    tool_T = Transform(robot.robot.R_tool, robot.robot.p_tool) # tool transformation in the base frame
    P_new, H_new, _ = PH_to_frame(robot.robot.P, robot.robot.H, tool_T, base_T) # convert current robot PH
    calib_P_new, calib_H_new, tool_new = PH_to_frame(robot.calib_P, robot.calib_H, tool_T, base_T) # convert CPA PH

    robot.robot.P = P_new
    robot.robot.H = H_new
    robot.calib_P = calib_P_new
    robot.calib_H = calib_H_new
    robot.p_tool = robot.robot.p_tool = tool_new.p
    robot.R_tool = robot.robot.R_tool = tool_new.R

    robot.base_H = np.eye(4) # reset the base frame to identity
    return robot

def get_robot_prepared(robot: robot_obj, unit='radians'):

    # using detectable markers/camera pose etc, as the tool
    T_base_basemarker = robot.T_base_basemarker
    T_basemarker_base = T_base_basemarker.inv()
    robot.T_basemarker_base = T_basemarker_base
    # remove last P, flange redundancy. All should be described by the tool
    T_tool_composite = robot.robot.T_flange * robot.T_toolmarker_flange # add T flange to the tool transformation
    T_tool_composite.p = T_tool_composite.p + robot.robot.P[:,-1] # add the last P to the tool transformation
    robot.R_tool = robot.robot.R_tool = T_tool_composite.R
    robot.p_tool = robot.robot.p_tool = T_tool_composite.p
    robot.robot.P[:,-1] = np.zeros(3) # remove the last P, as it is already in the tool transformation
    robot.robot.T_flange = Transform(np.eye(3),[0,0,0])
    robot.T_tool_toolmarker = Transform(np.eye(3),[0,0,0])

    # convert the base frame information in PH, tool parameters
    robot = convert_PH_to_inertial_frame(robot)
    # get the nominal P and H
    robot.P_nominal=deepcopy(robot.robot.P)
    robot.H_nominal=deepcopy(robot.robot.H)
    robot.P_nominal=robot.P_nominal.T
    robot.H_nominal=robot.H_nominal.T

    # get the axis to parametrize H
    robot = get_H_param_axis(robot)

    # get param ph and param tool
    param_ph, param_tool = get_param_from_PH_minimal_tool(robot, robot.robot.P, robot.robot.H, unit=unit)

    return robot, param_ph, param_tool

def main():

    config_dir='../config/'
    using_unit = 'radians'

    r1_ph_dataset = '0801'
    r1_marker_dir = config_dir+'MA2010_marker_config/'
    t1_marker_dir = config_dir+'weldgun_marker_config/'
    robot1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                            tool_file_path=config_dir+'torch.csv',d=15,\
                            #  tool_file_path='',d=0,\
                            pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                            base_marker_config_file=r1_marker_dir+'MA2010_'+r1_ph_dataset+'_marker_config.yaml',\
                            tool_marker_config_file=t1_marker_dir+'weldgun_'+r1_ph_dataset+'_marker_config.yaml')
    jN1 = len(robot1.robot.H.T)

    r2_ph_dataset = '0804'
    r2_marker_dir = config_dir+'MA1440_marker_config/'
    t2_marker_dir = config_dir+'mti_marker_config/'
    robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                            tool_file_path=config_dir+'mti.csv',\
                            pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                            base_marker_config_file=r2_marker_dir+'MA1440_'+r2_ph_dataset+'_marker_config.yaml',\
                            tool_marker_config_file=t2_marker_dir+'mti_'+r2_ph_dataset+'_marker_config.yaml',\
                            base_transformation_file=config_dir+'MA1440_pose.csv')
    jN2 = len(robot2.robot.H.T)

    # choose an inertial frame for the two robots
    inertial_frame_p = robot2.base_H[:3,3]/2 # inertial frame position in robot 1 base frame, mm
    inertial_frame_R = np.array(R2rpy(robot2.base_H[:3,:3]))/2 # inertial frame rotation in robot 1 base frame, radians
    print('inertial frame position in robot 1 base frame:', inertial_frame_p)
    print('inertial frame rotation in robot 1 base frame:', np.degrees(inertial_frame_R))
    robot1_base = Transform(rpy2R(inertial_frame_R), inertial_frame_p).inv() # inertial frame in robot 1 base frame
    robot1.base_H = H_from_RT(robot1_base.R, robot1_base.p) # robot 1 base frame in inertial frame
    robot2.base_H = robot1.base_H @ robot2.base_H # robot 2 base frame in inertial frame
    print('robot 1 base frame in inertial frame:', robot1.base_H)
    print('robot 2 base frame in inertial frame:', robot2.base_H)
    # convert the base frame information in PH, tool parameters, and get robot prepared
    robot1, param_ph1, param_t1 = get_robot_prepared(robot1, unit=using_unit)
    robot2, param_ph2, param_t2 = get_robot_prepared(robot2, unit=using_unit)
    assert type(robot1) == robot_obj, 'robot1 is not a robot_obj'
    assert type(robot2) == robot_obj, 'robot2 is not a robot_obj'

    # print zero configuration in the inertial frame
    print("robot 1 zero configuration in inertial frame:", robot1.fwd(np.zeros(jN1), world=True))
    print("robot 2 zero configuration in inertial frame:", robot2.fwd(np.zeros(jN2), world=True))
    print("Tool 1 in tool 2 frame zero configuration:", robot2.fwd(np.zeros(jN2)).inv() * robot1.fwd(np.zeros(jN1)))
    print("robot 1 initial parameters:", param_ph1, param_t1)
    print("robot 2 initial parameters:", param_ph2, param_t2)

    # generate a random ph, t, ground truth
    param_ph1_gt = deepcopy(param_ph1)
    param_ph2_gt = deepcopy(param_ph2)
    param_t1_gt = deepcopy(param_t1)
    param_t2_gt = deepcopy(param_t2)
    param_ph1_gt[:jN1*2] = np.random.uniform(-0.5,0.5,jN1*2) # vi, wi of robot1, mm
    param_ph1_gt[jN1*2:] = np.radians(np.random.uniform(-0.025,0.025,jN1*2)) # th_i, phi_i of robot1, radians
    param_ph2_gt[:jN1*2] = np.random.uniform(-0.5,0.5,jN2*2) # vi, wi of robot2, mm
    param_ph2_gt[jN1*2:] = np.radians(np.random.uniform(-0.025,0.025,jN2*2)) # th_i, phi_i of robot2, radians
    param_t1_gt[:3] = np.random.uniform(-1,1,3) # tool dp of robot1, mm
    param_t1_gt[3:] = np.radians(np.random.uniform(-0.05,0.05,3)) # tool dR of robot1, radians
    param_t2_gt[:3] = np.random.uniform(-1,1,3) # tool dp of robot2, mm
    param_t2_gt[3:] = np.radians(np.random.uniform(-0.05,0.05,3)) # tool dR of robot2, radians

    param_test = np.random.uniform(-0.5,0.5,24) # test parameters, mm and radians
    param_test = np.insert(param_test,2*jN1,np.zeros(3))
    robot1 = get_PH_from_param_minimal(param_test, robot1, unit=using_unit)
    param_test_ver = get_param_from_PH_minimal(robot1, robot1.robot.P, robot1.robot.H, robot1.P_nominal.T, robot1.H_nominal.T, unit=using_unit)
    print("test parameters:", param_test, param_test_ver)
    print("diff:", param_test_ver-param_test)
    # exit()

    print("robot 1 ground truth parameters:",param_ph1_gt, param_t1_gt)
    robot1, param_t1_gt = get_PH_tool_from_param_minimal(param_ph1_gt, param_t1_gt, robot1, unit=using_unit)
    print("robot 1 ground truth PH:", robot1.robot.P.T, robot1.robot.H.T)
    print("robot 1 ground truth tool:", robot1.robot.R_tool, robot1.robot.p_tool)
    param_ph1_gt_ver, param_t1_gt_ver = get_param_from_PH_minimal_tool(robot1, robot1.robot.P, robot1.robot.H, unit=using_unit)
    print("robot 1 ground truth parameters ver:", param_ph1_gt_ver, param_t1_gt_ver)

    print("diff ph:",param_ph1_gt_ver-param_ph1_gt)
    print("diff tool:",param_t1_gt_ver-param_t1_gt)
    exit()

    # generate the simulated dataset
    data_N = 1000
    data_joints = []
    data_T = []
    t1_t2_lower_limit_p = np.array([-500, -500, -1500])
    t1_t2_upper_limit_p = np.array([500, 500, 0])
    r1_lower_limit = np.clip(robot1.robot.joint_lower_limit, -np.pi, np.pi)
    r1_upper_limit = np.clip(robot1.robot.joint_upper_limit, -np.pi, np.pi)
    r2_lower_limit = np.clip(robot2.robot.joint_lower_limit, -np.pi, np.pi)
    r2_upper_limit = np.clip(robot2.robot.joint_upper_limit, -np.pi, np.pi)
    for i in range(data_N):
        # randomize a set of joint angles for robot2
        q1 = np.random.uniform(r1_lower_limit, r1_upper_limit, jN1)
        q2 = np.random.uniform(r2_lower_limit, r2_upper_limit, jN2)
        # check if t1_t2 p is in the limit
        t1 = robot1.fwd(q1)
        t2 = robot2.fwd(q2)
        t1_t2 = t2.inv() * t1
        if np.any(t1_t2.p < t1_t2_lower_limit_p) or np.any(t1_t2.p > t1_t2_upper_limit_p):
            continue
        # get ground truth T and joints
        data_joints.append(np.concatenate((q1, q2)))
        # get the tool transformation in the inertial frame, with using the ground truth parameters

if __name__ == '__main__':
    main()