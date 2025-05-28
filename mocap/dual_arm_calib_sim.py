from matplotlib import pyplot as plt
from copy import deepcopy
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

    # generate the simulated dataset
    try:
        data_joints = np.loadtxt('data_joints_dual_sim.csv',delimiter=',')
        data_T_gt = np.loadtxt('data_T_gt_dual_sim.csv',delimiter=',')
        param_ph1_gt = np.loadtxt('param_ph1_gt.csv',delimiter=',')
        param_ph2_gt = np.loadtxt('param_ph2_gt.csv',delimiter=',')
        param_t1_gt = np.loadtxt('param_t1_gt.csv',delimiter=',')
        param_t2_gt = np.loadtxt('param_t2_gt.csv',delimiter=',')
        robot1_gt, _ = get_PH_tool_from_param_minimal(param_ph1_gt, param_t1_gt, robot1, unit=using_unit)
        robot2_gt, _ = get_PH_tool_from_param_minimal(param_ph2_gt, param_t2_gt, robot2, unit=using_unit)
        data_N = len(data_joints)
    except:
        # generate a random ph, t, ground truth
        param_ph1_gt = deepcopy(param_ph1)
        param_ph2_gt = deepcopy(param_ph2)
        param_t1_gt = deepcopy(param_t1)
        param_t2_gt = deepcopy(param_t2)
        # param_ph1_gt[:jN1*2] = np.random.uniform(-0.5,0.5,jN1*2) # vi, wi of robot1, mm
        param_ph1_gt[:jN1*2] = np.random.normal(0.4,0.1,jN1*2) # vi, wi of robot1, mm
        # param_ph1_gt[jN1*2:] = np.radians(np.random.uniform(-0.025,0.025,jN1*2)) # th_i, phi_i of robot1, radians
        param_ph1_gt[jN1*2:] = np.radians(np.random.normal(0.03,0.05,jN1*2)) # th_i, phi_i of robot1, radians
        # param_ph2_gt[:jN1*2] = np.random.uniform(-0.5,0.5,jN2*2) # vi, wi of robot2, mm
        param_ph2_gt[:jN2*2] = np.random.normal(0.4,0.1,jN2*2) # vi, wi of robot2, mm
        #param_ph2_gt[jN1*2:] = np.radians(np.random.uniform(-0.025,0.025,jN2*2)) # th_i, phi_i of robot2, radians
        param_ph2_gt[jN2*2:] = np.radians(np.random.normal(0.03,0.05,jN2*2)) # th_i, phi_i of robot2, radians

        # param_t1_gt[:3] = np.random.uniform(-1,1,3) # tool dp of robot1, mm
        param_t1_gt[:3] = np.random.normal(1,0.1,3) # tool dp of robot1, mm
        # param_t1_gt[3:] = np.radians(np.random.uniform(-0.05,0.05,3)) # tool dR of robot1, radians
        param_t1_gt[3:] = np.radians(np.random.normal(0.03,0.05,3))
        #param_t2_gt[:3] = np.random.uniform(-1,1,3) # tool dp of robot2, mm
        #param_t2_gt[3:] = np.radians(np.random.uniform(-0.05,0.05,3)) # tool dR of robot2, radians
        param_t2_gt[:3] = np.random.normal(1,0.1,3) # tool dp of robot2, mm
        param_t2_gt[3:] = np.radians(np.random.normal(0.03,0.05,3)) # tool dR of robot2, radians

        np.savetxt('param_ph1_gt.csv', param_ph1_gt, delimiter=',')
        np.savetxt('param_ph2_gt.csv', param_ph2_gt, delimiter=',')
        np.savetxt('param_t1_gt.csv', param_t1_gt, delimiter=',')
        np.savetxt('param_t2_gt.csv', param_t2_gt, delimiter=',')

        # print("robot 1 ground truth parameters:",param_ph1_gt, param_t1_gt)
        robot1_gt, _ = get_PH_tool_from_param_minimal(param_ph1_gt, param_t1_gt, robot1, unit=using_unit)
        # print("robot 1 ground truth PH:", robot1_gt.robot.P.T, robot1_gt.robot.H.T)
        # print("robot 1 ground truth tool:", robot1_gt.robot.R_tool, robot1_gt.robot.p_tool)

        # print("robot 2 ground truth parameters:",param_ph2_gt, param_t2_gt)
        robot2_gt, _ = get_PH_tool_from_param_minimal(param_ph2_gt, param_t2_gt, robot2, unit=using_unit)
        # print("robot 2 ground truth PH:", robot2_gt.robot.P.T, robot2_gt.robot.H.T)
        # print("robot 2 ground truth tool:", robot2_gt.robot.R_tool, robot2_gt.robot.p_tool)

        data_N = 100
        data_joints = []
        data_T_gt = []
        t1_t2_lower_limit_p = np.array([-500, -500, -1500])
        t1_t2_upper_limit_p = np.array([500, 500, 0])
        r1_lower_limit = np.clip(robot1.robot.joint_lower_limit, -np.pi, np.pi)
        r1_upper_limit = np.clip(robot1.robot.joint_upper_limit, -np.pi, np.pi)
        r2_lower_limit = np.clip(robot2.robot.joint_lower_limit, -np.pi, np.pi)
        r2_upper_limit = np.clip(robot2.robot.joint_upper_limit, -np.pi, np.pi)
        while len(data_joints) < data_N:
            # randomize a set of joint angles for robot2
            q1 = np.random.uniform(r1_lower_limit, r1_upper_limit, jN1)
            q2 = np.random.uniform(r2_lower_limit, r2_upper_limit, jN2)
            # check if t1_t2 p is in the limit
            t1_gt = robot1_gt.fwd(q1)
            t2_gt = robot2_gt.fwd(q2)
            t1_t2_gt = t2_gt.inv() * t1_gt
            if np.any(t1_t2_gt.p < t1_t2_lower_limit_p) or np.any(t1_t2_gt.p > t1_t2_upper_limit_p):
                continue
            print("data #:", len(data_joints))
            # get ground truth T and joints
            data_joints.append(np.concatenate((q1, q2)))
            # get the tool transformation in the inertial frame, with the ground truth parameters
            data_T_gt.append(np.append(t1_t2_gt.p, R2q(t1_t2_gt.R)))
        np.savetxt('data_joints_dual_sim.csv', data_joints, delimiter=',')
        np.savetxt('data_T_gt_dual_sim.csv', data_T_gt, delimiter=',')
    input("Data generation complete. Press Enter to continue...")

    ### test Jacobian accuracy using numerical jacobian
    dP_up_range = 0.005
    dP_low_range = 0.01
    dab_up_range = np.radians(0.001)
    dab_low_range = np.radians(0.03)
    numerical_iteration=1000
    for data_q in data_joints:
        # randomize a set of parameters for robot1
        param_ph1 = deepcopy(param_ph1_gt)
        param_ph2 = deepcopy(param_ph2_gt)
        param_t1 = deepcopy(param_t1_gt)
        param_t2 = deepcopy(param_t2_gt)

        print("robot 1 P",robot1.robot.P.T)
        robot1, param_t1 = get_PH_tool_from_param_minimal(param_ph1, param_t1, robot1, unit=using_unit)
        robot2, param_t2 = get_PH_tool_from_param_minimal(param_ph2, param_t2, robot2, unit=using_unit)
        print("robot 1 P after get_PH_tool_from_param_minimal",robot1.robot.P.T)
        # get J_ana
        this_J_dual_ana = jacobian_param_minimal_dual(param_ph1, data_q[:jN1], robot1, \
                                                  param_ph2, data_q[jN1:], robot2, unit=using_unit)
        this_J_tool_ana = jacobian_tool_dual(data_q[:jN1], robot1, \
                                         data_q[jN1:], robot2, unit=using_unit)
        this_J1_ana = jacobian_param_minimal(np.append(param_ph1,np.zeros(3)), robot1, data_q[:jN1], unit=using_unit)
        this_J1_ana = np.delete(this_J1_ana,[2*jN1,2*jN1+1,2*jN1+2],axis=1)

        t1 = robot1.fwd(data_q[:jN1]) # robot 1 forward kinematics
        t2 = robot2.fwd(data_q[jN1:]) # robot 2 forward kinematics
        t2_t1_init = t2.inv() * t1 # t2_t1 transformation

        # single robot jacobian verification
        
        d_T_all = [] # difference in robot T
        d_param_all = [] # difference in param
        d_T1_all = [] # difference in robot 1 T
        d_param1_all = [] # difference in robot 1 param
        for iter_i in range(numerical_iteration):
            # perturb the parameters
            d_param_ph1 = np.random.uniform(-dP_up_range, dP_up_range, jN1*2)
            d_param_ph1 = np.append(d_param_ph1, np.radians(np.random.uniform(-dab_up_range, dab_up_range, jN1*2)))
            d_param_ph2 = np.random.uniform(-dP_up_range, dP_up_range, jN2*2)
            d_param_ph2 = np.append(d_param_ph2, np.radians(np.random.uniform(-dab_up_range, dab_up_range, jN2*2)))
            d_param_t1 = np.random.uniform(-dP_up_range, dP_up_range, 3)
            d_param_t1 = np.append(d_param_t1, np.radians(np.random.uniform(-dab_up_range, dab_up_range, 3)))
            d_param_t2 = np.random.uniform(-dP_up_range, dP_up_range, 3)
            d_param_t2 = np.append(d_param_t2, np.radians(np.random.uniform(-dab_up_range, dab_up_range, 3)))
            # d_param_t1 = np.zeros_like(param_t1) # ignore the tool perturbation for now
            # d_param_t2 = np.zeros_like(param_t2)
            this_d_param_all = np.concatenate((d_param_ph1, d_param_ph2, d_param_t1, d_param_t2))
            d_param_all.append(this_d_param_all) # append the difference in param
            d_param1_all.append(d_param_ph1) # append the difference in param for robot 1


            this_param_ph1 = param_ph1 + d_param_ph1
            this_param_ph2 = param_ph2 + d_param_ph2
            # this_param_ph1 = deepcopy(param_ph1) # deep copy to avoid modifying the original param
            # this_param_ph2 = deepcopy(param_ph2) # deep copy to avoid modifying the original param
            this_param_t1 = param_t1 + d_param_t1
            this_param_t2 = param_t2 + d_param_t2

            # get the new robots using perturbed params
            this_robot1, _ = get_PH_tool_from_param_minimal(this_param_ph1, this_param_t1, robot1, unit=using_unit)
            this_robot2, _ = get_PH_tool_from_param_minimal(this_param_ph2, this_param_t2, robot2, unit=using_unit)

            t1_pert = this_robot1.fwd(data_q[:jN1]) # robot 1 forward kinematics, after parameter perturbation
            t2_pert = this_robot2.fwd(data_q[jN1:]) # robot 2 forward kinematics, after parameter perturbation
            t2_t1_pert = t2_pert.inv() * t1_pert # t2_t1 transformation after parameter perturbation
            dp = t2_t1_pert.p - t2_t1_init.p # position difference
            dR = t2_t1_pert.R - t2_t1_init.R # rotation difference
            dRRT = dR@t2_t1_init.R.T # rotation difference in the initial frame
            ktheta = invhat(dRRT)
            d_T = np.append(ktheta, dp) # difference in T
            d_T_all.append(d_T) # append the difference in T
            # robot 1 jacobian verification
            dp1 = t1_pert.p - t1.p # position difference for robot 1
            dR1 = t1_pert.R - t1.R # rotation difference for robot 1
            dRRT1 = dR1@t1.R.T # rotation difference in the initial frame for robot 1
            ktheta1 = invhat(dRRT1)
            d_T1 = np.append(ktheta1, dp1) # difference in T for robot 1
            d_T1_all.append(d_T1) # append the difference in T for robot 1
            
        d_param_all = np.array(d_param_all).T # shape: (total_params, numerical_iteration)
        d_T_all = np.array(d_T_all).T # shape: (6, numerical_iteration)
        # compute the numerical jacobian
        J_numerical = d_T_all @ np.linalg.pinv(d_param_all) # shape: (6, total_params)
        J_ana = np.hstack((this_J_dual_ana, this_J_tool_ana)) # analytical jacobian

        d_param1_all = np.array(d_param1_all).T # shape: (total_params_robot1, numerical_iteration)
        d_T1_all = np.array(d_T1_all).T # shape: (6, numerical_iteration)
        J1_numerical = d_T1_all @ np.linalg.pinv(d_param1_all) # shape: (6, total_params)
        J1_ana = this_J1_ana # analytical jacobian for robot 1


        # show J1 numerical and J1_ana and J1 error in a 3x1 grid
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.5))
        axs[0].matshow(np.clip(J1_numerical, -1, 1), cmap='jet', interpolation='nearest')
        axs[0].set_title("Numerical Jacobian for Robot 1")
        axs[0].set_xlabel("Parameters")
        axs[0].set_ylabel("Errors")
        axs[1].matshow(np.clip(J1_ana, -1, 1), cmap='jet', interpolation='nearest')
        axs[1].set_title("Analytical Jacobian for Robot 1")
        axs[1].set_xlabel("Parameters")
        axs[1].set_ylabel("Errors")
        axs[2].matshow(np.clip(np.abs(J1_numerical-J1_ana),0,1), cmap='jet', interpolation='nearest')
        axs[2].set_title("Numerical Jacobian vs Analytical Jacobian for Robot 1")
        axs[2].set_xlabel("Parameters")
        axs[2].set_ylabel("Errors")
        # plt.colorbar(ax=axs[2])
        plt.tight_layout()
        plt.show()

        # show J numerical and J_ana and J error in a 3x1 grid
        fig, axs = plt.subplots(3, 1, figsize=(10, 7.5))
        axs[0].matshow(np.clip(J_numerical, -1, 1), cmap='jet', interpolation='nearest')
        axs[0].set_title("Numerical Jacobian")
        axs[0].set_xlabel("Parameters")
        axs[0].set_ylabel("Errors")
        axs[1].matshow(np.clip(J_ana, -1, 1), cmap='jet', interpolation='nearest')
        axs[1].set_title("Analytical Jacobian")
        axs[1].set_xlabel("Parameters")
        axs[1].set_ylabel("Errors")
        axs[2].matshow(np.clip(np.abs(J_numerical-J_ana),0,1), cmap='jet', interpolation='nearest')
        axs[2].set_title("Numerical Jacobian vs Analytical Jacobian")
        axs[2].set_xlabel("Parameters")
        axs[2].set_ylabel("Errors")
        # plt.colorbar(ax=axs[2])
        plt.tight_layout()
        plt.show()

    exit()

    # calibration
    weight_P = 1
    weight_H = 1
    weight_pos = 1
    weight_ori = 1
    # weight_ori = 1641
    alpha=0.1
    lambda_H = 57
    lambda_P = 1
    lambda_tool_p = 1
    lambda_tool_R = 57
    total_P1 = 2*jN1 # total number of P parameters to be estimated. robot 1
    total_H1 = 2*jN1 # total number of H parameters to be estimated. robot 1
    total_P2 = 2*jN2 # total number of P parameters to be estimated. robot 2
    total_H2 = 2*jN2 # total number of H parameters to be estimated. robot 2
    total_tool_p = 3 # total number of tool p parameters to be estimated, for 1 robot
    total_tool_R = 3 # total number of tool R parameters to be estimated, for 1 robot
    max_iteration = 50
    
    pos_error_norm_progress = []
    ori_error_norm_progress = []
    param_p1_error_progress = []
    param_h1_error_progress = []
    param_p2_error_progress = []
    param_h2_error_progress = []
    param_t1p_error_progress = []
    param_t1R_error_progress = []
    param_t2p_error_progress = []
    param_t2R_error_progress = []
    for iter_N in range(max_iteration):
        print("Iteration #:", iter_N)
        # get the current robots using params
        param_t1[3:] *= -1
        robot1, param_t1 = get_PH_tool_from_param_minimal(param_ph1, param_t1, robot1, unit=using_unit)
        robot2, param_t2 = get_PH_tool_from_param_minimal(param_ph2, param_t2, robot2, unit=using_unit)

        J_ana = []
        error_pos_ori = []
        error_pos = []
        error_ori = []
        for (data_q,data_T) in zip(data_joints, data_T_gt):
            # get J_ana
            this_J_dual = jacobian_param_minimal_dual(param_ph1, data_q[:jN1], robot1, \
                                                      param_ph2, data_q[jN1:], robot2, unit=using_unit)
            this_J_tool = jacobian_tool_dual(data_q[:jN1], robot1, \
                                             data_q[jN1:], robot2, unit=using_unit)
            this_J = np.hstack((this_J_dual, this_J_tool))
            J_ana.extend(this_J)
            # get error
            T_gt = Transform(q2R(data_T[3:]), data_T[:3]) # ground truth T
            t2_t1_pred = robot2.fwd(data_q[jN1:]).inv() * robot1.fwd(data_q[:jN1]) # t2_t1_pred
            vd = t2_t1_pred.p - T_gt.p # position error
            omega_d=s_err_func(t2_t1_pred.R@T_gt.R.T)
            error_pos_ori = np.append(error_pos_ori,np.append(omega_d*weight_ori,vd*weight_pos))
            error_pos.append(np.linalg.norm(vd))
            # error_ori.append(np.degrees(omega_d))  # for plotting purpose only (unit: degrees)
            k, ori_diff = R2rot(t2_t1_pred.R@T_gt.R.T)
            error_ori.append(np.abs(ori_diff))  # for plotting purpose only (unit: degrees)   

            # print("vd:", vd, "norm:", np.linalg.norm(vd))
            # print("omega_d:", omega_d, "norm:", np.linalg.norm(omega_d))
            # print("ori_diff:", np.abs(np.degrees(ori_diff)))
            # input("Press Enter to continue...")
        J_ana = np.array(J_ana)
        pos_error_norm_progress.append(np.mean(error_pos))
        ori_error_norm_progress.append(np.mean(error_ori))
        param_p1_error_progress.append(param_ph1[:jN1*2]-param_ph1_gt[:jN1*2])
        param_h1_error_progress.append(param_ph1[jN1*2:]-param_ph1_gt[jN1*2:])
        param_p2_error_progress.append(param_ph2[:jN2*2]-param_ph2_gt[:jN2*2])
        param_h2_error_progress.append(param_ph2[jN2*2:]-param_ph2_gt[jN2*2:])
        param_t1p_error_progress.append(robot1.robot.p_tool-robot1_gt.robot.p_tool)
        param_t1R_error_progress.append(R2rpy(robot1.robot.R_tool@robot1_gt.robot.R_tool.T))
        param_t2p_error_progress.append(robot2.robot.p_tool-robot2_gt.robot.p_tool)
        param_t2R_error_progress.append(R2rpy(robot2.robot.R_tool@robot2_gt.robot.R_tool.T))

        print("Pose error, orientation error:", np.mean(error_pos), np.degrees(np.mean(error_ori)))
        # update PH using QP
        # parameters: param_ph1, param_t1, param_ph2, param_t2
        G = J_ana
        Kq = np.hstack((np.ones(total_P1)*lambda_P, np.ones(total_H1)*lambda_H, \
                        np.ones(total_P2)*lambda_P, np.ones(total_H2)*lambda_H, \
                        np.ones(total_tool_p)*lambda_tool_p, np.ones(total_tool_R)*lambda_tool_R,\
                        np.ones(total_tool_p)*lambda_tool_p, np.ones(total_tool_R)*lambda_tool_R))
        Kq = np.diag(Kq)
        H=G.T@G + Kq
        H = (H + H.T) / 2
        f = -G.T@error_pos_ori
        dparam = solve_qp(H, f, solver='quadprog')

        # param_ph1 = param_ph1 - alpha*dparam[:total_P1+total_H1]
        dparam = dparam[total_P1+total_H1:]
        # param_ph2 = param_ph2 - alpha*dparam[:total_P2+total_H2]
        dparam = dparam[total_P2+total_H2:]
        param_t1 = param_t1 - alpha*dparam[:total_tool_p+total_tool_R]
        dparam = dparam[total_tool_p+total_tool_R:]
        param_t2 = param_t2 - alpha*dparam[:total_tool_p+total_tool_R]
        dparam = dparam[total_tool_p+total_tool_R:]
        print("dparam:", dparam)

    # plot error progress in a 2x3 grid
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].plot(pos_error_norm_progress, label='Position Error', color='blue')
    axs[0, 0].set_title('Position Error Progress')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Error Norm (mm)')
    axs[0, 1].plot(ori_error_norm_progress, label='Orientation Error', color='orange')
    axs[0, 1].set_title('Orientation Error Progress')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Error Norm (degrees)')
    axs[0, 2].plot(np.linalg.norm(param_p1_error_progress,axis=1), label='P1 Error', color='green')
    axs[0, 2].plot(np.linalg.norm(param_p2_error_progress,axis=1), label='P2 Error', color='red')
    axs[0, 2].set_title('P Error Progress')
    axs[0, 2].set_xlabel('Iteration')
    axs[0, 2].set_ylabel('Error Norm (mm)')
    axs[0, 2].legend()
    axs[1, 0].plot(np.linalg.norm(param_h1_error_progress,axis=1), label='H1 Error', color='purple')
    axs[1, 0].plot(np.linalg.norm(param_h2_error_progress,axis=1), label='H2 Error', color='brown')
    axs[1, 0].set_title('H Error Progress')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Error Norm (degrees)')
    axs[1, 0].legend()
    axs[1, 1].plot(np.linalg.norm(param_t1p_error_progress,axis=1), label='Tool P1 Error', color='pink')
    axs[1, 1].plot(np.linalg.norm(param_t2p_error_progress,axis=1), label='Tool P2 Error', color='cyan')
    axs[1, 1].set_title('Tool P Error Progress')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Error Norm (mm)')
    axs[1, 1].legend()
    axs[1, 2].plot(np.linalg.norm(param_t1R_error_progress,axis=1), label='Tool R1 Error', color='gray')
    axs[1, 2].plot(np.linalg.norm(param_t2R_error_progress,axis=1), label='Tool R2 Error', color='olive')
    axs[1, 2].set_title('Tool R Error Progress')
    axs[1, 2].set_xlabel('Iteration')
    axs[1, 2].set_ylabel('Error Norm (degrees)')
    axs[1, 2].legend()
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs[0,0].plot(np.fabs(param_p1_error_progress)[:,:jN1],label=['v1','w1','v2','w2','v3','w3'])
    axs[0,0].legend()
    axs[0,1].plot(np.fabs(param_p1_error_progress)[:,jN1:],label=['v4','w4','v5','w5','v6','w6'])
    axs[0,1].legend()
    axs[1,0].plot(np.fabs(param_p2_error_progress)[:,:jN2],label=['v1','w1','v2','w2','v3','w3'])
    axs[1,0].legend()
    axs[1,1].plot(np.fabs(param_p2_error_progress)[:,jN2:],label=['v4','w4','v5','w5','v6','w6'])
    axs[1,1].legend()
    # axs[0].set_title('P1 Error Progress')
    # axs[1].set_title('P2 Error Progress')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()