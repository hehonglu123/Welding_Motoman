import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt
from scipy.optimize import fminbound
from qpsolvers import solve_qp
from calib_analytic_grad import *

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

r1dataset_date='0801'
r2dataset_date='0804'
config_dir='../config/'

###### define robots ######
robots = []

robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'torch.csv',d=15,\
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+r1dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+r1dataset_date+'_marker_config.yaml')
robots.append(robot)
robot_marker_dir=config_dir+'MA1440_marker_config/'
tool_marker_dir=config_dir+'mti_marker_config/'
robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'mti.csv',\
                    pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
                    base_transformation_file=config_dir+'MA1440_pose.csv',\
                    base_marker_config_file=robot_marker_dir+'MA1440_'+r2dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'mti_'+r2dataset_date+'_marker_config.yaml')
robots.append(robot)
TR2R1 = Transform(robots[1].base_H[:3,:3],robots[1].base_H[:3,3])

print("robot 1 zero config: ", robots[0].fwd(np.zeros(6)))
print("robot 2 zero config: ", robots[1].fwd(np.zeros(6)))
print("robot 2 zero config world: ", robots[1].fwd(np.zeros(6),world=True))
r2T0=robots[1].fwd(np.zeros(6),world=True)
print(r2T0.R@rot(Rx,np.pi)@rot(Rz,np.pi/2))
print("======Start Calibrating======")
###########################

## test iter-ik
# print("upper limits: ", np.degrees(robots[1].upper_limit))
# print("lower limits: ", np.degrees(robots[1].lower_limit))
# for i in range(1000):
#     q2 = rng.uniform(low=robots[1].lower_limit, high=robots[1].upper_limit)
#     print(np.degrees(q2))
#     print("min J singularity: ", np.linalg.svd(robots[1].jacobian(q2))[1].min())
#     T = robots[1].fwd(q2)
#     st=time.time()
#     qsol = robots[1].inv_iter(T.p,T.R,q_seed=q2+rng.uniform(low=-0.1, high=0.1, size=6))
#     print(time.time()-st)
#     print(np.linalg.norm(q2-qsol))
#     input(np.degrees(qsol))
#     print("=====================================")

##### nominal parameters #####
param_noms = []
for robot in robots:
    jN = len(robot.robot.H[0])
    robot.P_nominal=deepcopy(robot.robot.P)
    robot.H_nominal=deepcopy(robot.robot.H)
    robot.P_nominal=robot.P_nominal.T
    robot.H_nominal=robot.H_nominal.T
    robot = get_H_param_axis(robot)
    
    ### nomnial parameters
    param_nom = deepcopy(robot.P_nominal)
    param_nom = np.reshape(param_nom, (param_nom.size, ))
    param_nom = np.append(param_nom,np.zeros(jN*2))
    param_noms.append(param_nom)
##############################

##### simulated actual parameters #####
dP_up_range = 0.3
dP_low_range = 0.1
dab_up_range = np.radians(0.1)
dab_low_range = np.radians(0.03)

param_gts = []
for robot, param_nom in zip(robots, param_noms):
    jN = len(robot.robot.H[0])
    dP=rng.uniform(low=-(dP_up_range-dP_low_range),high=(dP_up_range-dP_low_range),size=((jN+1)*3,))
    dP=dP+dP/np.fabs(dP)*dP_low_range
    dab=rng.uniform(low=-(dab_up_range-dab_low_range),high=(dab_up_range-dab_low_range),size=(jN*2,))
    dab=dab+dab/np.fabs(dab)*dab_low_range
    param_gt = param_nom + np.concatenate((dP, dab))
    param_gts.append(param_gt)
#######################################

##### collect data #####
collected_data_N = 10
joint_data = [[] for i in range(len(robots))]
pose_data = []

limit_factor = np.radians(2)
toolR2R1 = rot(Rx,np.pi)@rot(Rz,np.pi/2)
data_cnt=0
while data_cnt < collected_data_N:
    ## random joint angles with upper and lower limits
    ## start from robot2
    joint_angles = []
    q2 = rng.uniform(low=robots[1].lower_limit+limit_factor, high=robots[1].upper_limit-limit_factor) \
            if data_cnt!=0 else np.array([0,0,0,0,np.radians(10),0])
    ## robot2 forward kinematics using simulated actual parameters
    robots[1] = get_PH_from_param(param_gts[1], robots[1])
    r2T_r1 = robots[1].fwd(q2,world=True)
    ## get robot 1 tool pose
    r1T = deepcopy(r2T_r1)
    r1T.R = r2T_r1.R@toolR2R1
    ## get robot 1 tool pose in robot 2 tool frame
    r1T_r2 = r2T_r1.inv()*r1T
    ## solve for robot1 joint angles
    robots[0] = get_PH_from_param(param_noms[0], robots[0])
    try:
        q1_nom = robots[0].inv(r1T.p, r1T.R, last_joints=np.zeros(6))[0]
    except ValueError:
        continue
    if min(np.linalg.svd(robots[0].jacobian(q1_nom))[1]) < 1e-3:
        print("near singular")
        continue
    robots[0] = get_PH_from_param(param_gts[0], robots[0])
    q1 = robots[0].inv_iter(r1T.p, r1T.R, q_seed=q1_nom, lim_factor=limit_factor/2)
    
    ## record data
    joint_data[0].append(q1)
    joint_data[1].append(q2)
    pose_data.append(np.append(r1T_r2.p,R2q(r1T_r2.R)))
    data_cnt+=1
    
    # print(robots[0].fwd(q1_nom))
    # print("q2 joint angles: ", np.degrees(q2))
    # print("r2T_r1: ", r2T_r1)
    # print("r1T: ", r1T)
    # print("q1_nom: ", np.degrees(q1_nom))
    # print("q1: ", np.degrees(q1))
    # input("=====================")
########################
print("simulated data collected")


def get_dPt1t2dparam(joints, params, robots, TR2R1):
    q1 = joints[0]
    q2 = joints[1]
    jN = len(q1)
    r2T_r1 = robots[1].fwd(q2,world=True)
    r2T = robots[1].fwd(q2)
    r1T = robots[0].fwd(q1)
    t2_t1 = r2T_r1.inv()*r1T
    J1_ana = jacobian_param(params[0],robots[0],q1,unit='degrees')
    J2_ana = jacobian_param(params[1],robots[1],q2,unit='degrees')
    # [p_i / PR1 p_i / HR1]
    dpdparamR1 = r2T_r1.R@J1_ana[3:,:]
    # [p_i / PR2 p_i / HR2]
    dpdparamR2 = -r2T_r1.R@J2_ana[3:,:]
    dpt_r2 = TR2R1.R.T@(r1T.p-r2T.p)
    dRdab = np.zeros_like(dpdparamR1)
    for ab in range((jN+1)*3,(jN+1)*3+jN*2):
        dRdab[:,ab]=hat(J2_ana[3:,ab])@r2T.R@dpt_r2
    dpdparamR2=dpdparamR2+dRdab
    return dpdparamR1, dpdparamR2, t2_t1

##### calibration, using relative distance #####
iter_N = 100
alpha = 0.001
param_calib = deepcopy(param_noms)
for it in range(iter_N):
    print("iter: ", it)
    error_vec = []
    G1 = []
    G2 = []
    for data_i in range(collected_data_N):
        for data_j in range(data_i+1, collected_data_N):
            robots[0] = get_PH_from_param(param_calib[0], robots[0])
            robots[1] = get_PH_from_param(param_calib[1], robots[1])
            
            ## get p_i, and all gradients
            dpidparamR1,dpidparamR2, t2_t1_i = get_dPt1t2dparam([joint_data[0][data_i],joint_data[1][data_i]], param_calib, robots, TR2R1)
            ## get p_j, and all gradients
            dpjdparamR1,dpjdparamR2, t2_t1_j = get_dPt1t2dparam([joint_data[0][data_j],joint_data[1][data_j]], param_calib, robots, TR2R1)
            ## add to gradient
            G1.extend(dpidparamR1-dpjdparamR1)
            G2.extend(dpidparamR2-dpjdparamR2)
            ## add error vec
            error_vec.extend(t2_t1_i.p-t2_t1_j.p)
    G1 = np.array(G1)
    G2 = np.array(G2)
    error_vec = np.array(error_vec)
    G1_m = deepcopy(G1)
    G2_m = deepcopy(G2)
    G1_m[:,-12:] = 0
    G2_m[:,-12:] = 0
    param_calib[0] = param_calib[0] - alpha*np.dot(error_vec,G1_m)
    param_calib[1] = param_calib[1] - alpha*np.dot(error_vec,G2_m)
    print("error norm:", np.linalg.norm(np.array(error_vec)))