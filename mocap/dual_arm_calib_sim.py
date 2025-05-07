from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *

from general_robotics_toolbox import *
import numpy as np
import time
import yaml
from qpsolvers import solve_qp
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

    # print("robot 1 ground truth parameters:",param_ph1_gt, param_t1_gt)
    robot1_gt, _ = get_PH_tool_from_param_minimal(param_ph1_gt, param_t1_gt, robot1, unit=using_unit)
    # print("robot 1 ground truth PH:", robot1_gt.robot.P.T, robot1_gt.robot.H.T)
    # print("robot 1 ground truth tool:", robot1_gt.robot.R_tool, robot1_gt.robot.p_tool)

    # print("robot 2 ground truth parameters:",param_ph2_gt, param_t2_gt)
    robot2_gt, _ = get_PH_tool_from_param_minimal(param_ph2_gt, param_t2_gt, robot2, unit=using_unit)
    # print("robot 2 ground truth PH:", robot2_gt.robot.P.T, robot2_gt.robot.H.T)
    # print("robot 2 ground truth tool:", robot2_gt.robot.R_tool, robot2_gt.robot.p_tool)

    # generate the simulated dataset
    data_N = 500
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
        t1_gt = robot1_gt.fwd(q1)
        t2_gt = robot2_gt.fwd(q2)
        t1_t2_gt = t2_gt.inv() * t1_gt
        if np.any(t1_t2_gt.p < t1_t2_lower_limit_p) or np.any(t1_t2_gt.p > t1_t2_upper_limit_p):
            continue
        # get ground truth T and joints
        data_joints.append(np.concatenate((q1, q2)))
        # get the tool transformation in the inertial frame, with the ground truth parameters
        data_T.append(np.append(t1_t2_gt.p, R2q(t1_t2_gt.R)))

    # calibration
    weight_P = 1
    weight_H = 1
    weight_pos = 1
    weight_ori = 1641
    alpha=0.01
    lambda_H = 30
    lambda_P = 2.5
    lambda_tool_p = 2.5
    lambda_tool_R = 15
    total_P1 = 2*jN1 # total number of P parameters to be estimated. robot 1
    total_H1 = 2*jN1 # total number of H parameters to be estimated. robot 1
    total_P2 = 2*jN2 # total number of P parameters to be estimated. robot 2
    total_H2 = 2*jN2 # total number of H parameters to be estimated. robot 2
    total_tool_p = 3 # total number of tool p parameters to be estimated, for 1 robot
    total_tool_R = 3 # total number of tool R parameters to be estimated, for 1 robot
    max_iteration = 200
    
    pos_error_norm_progress = []
    ori_error_norm_progress = []
    param_ph1_error_progress = []
    param_ph2_error_progress = []
    param_t1_error_progress = []
    param_t2_error_progress = []
    for iter_N in range(max_iteration):
        # get the current robots using params
        robot1, param_t1 = get_PH_tool_from_param_minimal(param_ph1, param_t1, robot1, unit=using_unit)
        robot2, param_t2 = get_PH_tool_from_param_minimal(param_ph2, param_t2, robot2, unit=using_unit)

        J_ana = []
        error_pos_ori = []
        for data_q in data_joints:
            # get J_ana
            this_J_dual = jacobian_param_minimal_dual(param_ph1, data_q[:jN1], robot1, \
                                                      param_ph2, data_q[jN1:], robot2, unit=using_unit)
            this_J_tool = jacobian_tool_dual(data_q[:jN1], robot1, \
                                             data_q[jN1:], robot2, unit=using_unit)
            # get error
            pass

        J_ana = np.array(J_ana)

        # update PH using QP
        # parameters: param_ph1, param_t1, param_ph2, param_t2
        G = J_ana
        Kq = np.hstack((np.ones(total_P1)*lambda_P, np.ones(total_H1)*lambda_H, \
                        np.ones(total_tool_p)*lambda_tool_p, np.ones(total_tool_R)*lambda_tool_R,\
                        np.ones(total_P2)*lambda_P, np.ones(total_H2)*lambda_H, \
                        np.ones(total_tool_p)*lambda_tool_p, np.ones(total_tool_R)*lambda_tool_R))
        Kq = np.diag(Kq)
        H=G.T@G + Kq
        H = (H + H.T) / 2
        f = -G.T@error_pos_ori
        dparam = solve_qp(H, f, solver='quadprog')

        param_ph1 = param_ph1 + dparam[:total_P1+total_H1]
        dparam = dparam[total_P1+total_H1:]
        param_t1 = param_t1 + dparam[:total_tool_p+total_tool_R]
        dparam = dparam[total_tool_p+total_tool_R:]
        param_ph2 = param_ph2 + dparam[:total_P2+total_H2]
        dparam = dparam[total_P2+total_H2:]
        param_t2 = param_t2 + dparam[:total_tool_p+total_tool_R]


if __name__ == '__main__':
    main()