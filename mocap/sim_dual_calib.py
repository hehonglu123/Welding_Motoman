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
rng = default_rng(seed=0)

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
dP_up_range = 1.1
dP_low_range = 0.05
dab_up_range = np.radians(0.2)
dab_low_range = np.radians(0.02)

param_gts = []
param_upper_bounds = []
param_lower_bounds = []
for robot, param_nom in zip(robots, param_noms):
    jN = len(robot.robot.H[0])
    dP=rng.uniform(low=-(dP_up_range-dP_low_range),high=(dP_up_range-dP_low_range),size=((jN+1)*3,))
    dP=dP+dP/np.fabs(dP)*dP_low_range
    dab=rng.uniform(low=-(dab_up_range-dab_low_range),high=(dab_up_range-dab_low_range),size=(jN*2,))
    dab=dab+dab/np.fabs(dab)*dab_low_range
    param_gt = param_nom + np.concatenate((dP, dab))
    param_gts.append(param_gt)
    param_upper_bounds.extend(np.append(np.ones((jN+1)*3)*dP_up_range,np.ones(jN*2)*dab_up_range)+param_nom)
    param_lower_bounds.extend(np.append(np.ones((jN+1)*3)*-dP_up_range,np.ones(jN*2)*-dab_up_range)+param_nom)
#######################################

##### collect data #####
collected_data_N = 200
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
    t1_t2 = r2T_r1.inv()*r1T
    J1_ana = jacobian_param(params[0],robots[0],q1,unit='degrees')
    J2_ana = jacobian_param(params[1],robots[1],q2,unit='degrees')
    # [p_i / PR1 p_i / HR1]
    dpdparamR1 = r2T_r1.R.T@J1_ana[3:,:]
    # [p_i / PR2 p_i / HR2]
    dpdparamR2 = -r2T.R.T@J2_ana[3:,:]
    dpt_r2 = (TR2R1.R.T)@(r1T.p-r2T_r1.p)
    dRdab = np.zeros_like(dpdparamR2)
    for ab in range((jN+1)*3,(jN+1)*3+jN*2):
        dRdab[:,ab]=(-r2T.R.T@hat(J2_ana[3:,ab]))@dpt_r2
    dpdparamR2=dpdparamR2+dRdab
    return dpdparamR1, dpdparamR2, t1_t2

def get_dPRt1t2dparam(joints, params, robots, TR2R1):
    
    q1 = joints[0]
    q2 = joints[1]
    jN = len(q1)
    r2T_r1 = robots[1].fwd(q2,world=True)
    r1_t2 = r2T_r1.inv()
    r2T = robots[1].fwd(q2)
    r2_t2 = r2T.inv()
    r1T = robots[0].fwd(q1)
    t1_t2 = r1_t2*r1T
    J1_ana = jacobian_param(params[0],robots[0],q1,unit='radians')
    J2_ana = jacobian_param(params[1],robots[1],q2,unit='radians')
    
    dpt1t2_t2=r1_t2.R@(r1T.p-r2T_r1.p) # note: dpt1t2_t2 and t1_t2.p is the same
    
    J1p=np.matmul(r1_t2.R,J1_ana[3:,:])
    J1R=np.matmul(r1_t2.R,J1_ana[:3,:])
    
    J2p=np.matmul(r2_t2.R,J2_ana[3:,:])
    J2R=np.matmul(r2_t2.R,J2_ana[:3,:])
    
    dpRdparamR1 = np.vstack((J1R,J1p))
    dpRdparamR2 = np.vstack((-J2R,-J2p+hat(dpt1t2_t2)@J2R))
    
    return dpRdparamR1, dpRdparamR2, t1_t2

##### calibration, using relative pose #####
iter_N = 200
alpha = 0.03
lambda_P=1
lambda_H=1000
P_size=7
H_size=6
weight_H = 0.1
weight_P = 1
r1_param_weight = np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2)*lambda_H)
r2_param_weight = np.append(np.ones(P_size*3)*lambda_P,np.ones(H_size*2)*lambda_H)

param_calib = deepcopy(param_noms)
ave_error_iter=[]
param1_norm_iter=[np.linalg.norm(param_gts[0]-param_calib[0])]
param2_norm_iter=[np.linalg.norm(param_gts[1]-param_calib[1])]
for it in range(iter_N):
    try:
        print("iter: ", it)
        error_nu = []
        J_all = []
        ave_error = []
        this_param = np.resize(param_calib,(len(param_calib)*len(param_calib[0]),))
        for data_i in range(collected_data_N):
            robots[0] = get_PH_from_param(param_calib[0], robots[0])
            robots[1] = get_PH_from_param(param_calib[1], robots[1])
            dpRdparamR1, dpRdparamR2, t1_t2 = get_dPRt1t2dparam([joint_data[0][data_i],joint_data[1][data_i]], param_calib, robots, TR2R1)
            
            # weighting
            dpRdparamR1[:,:P_size*3] *= weight_P
            dpRdparamR1[:,P_size*3:] *= weight_H
            dpRdparamR2[:,:P_size*3] *= weight_P
            dpRdparamR2[:,P_size*3:] *= weight_H
            
            # ground truth relative pose
            gt_p = pose_data[data_i][:3]
            gt_R = q2R(pose_data[data_i][3:])
            
            vd = gt_p-t1_t2.p
            omega_d=s_err_func(t1_t2.R@gt_R.T)
            ave_error.append(np.linalg.norm(np.append(omega_d,vd)))
            error_nu.extend(np.append(omega_d,vd))
            if data_i==0:
                J_all = np.hstack((dpRdparamR1,dpRdparamR2))
            else:
                J_all = np.vstack((J_all,np.hstack((dpRdparamR1,dpRdparamR2))))
        
        
        Kq = np.diag(np.append(r1_param_weight,r2_param_weight))
        H=np.matmul(J_all.T,J_all)+Kq
        H=(H+np.transpose(H))/2
        f=-np.matmul(J_all.T,error_nu)
        # dph=solve_qp(H,f,solver='quadprog',lb=np.array(param_lower_bounds)-this_param,ub=np.array(param_upper_bounds)-this_param)
        dph=solve_qp(H,f,solver='quadprog')
        
        # eps=0.1
        # dph = np.linalg.pinv(J_all)@error_nu
        # dph = -J_all.T@(J_all@J_all.T+eps*np.eye(J_all.shape[0]))@error_nu
        
        d_pH_update = alpha*dph
        param_calib[0] = param_calib[0]+d_pH_update[:len(param_calib[0])]
        param_calib[1] = param_calib[1]+d_pH_update[len(param_calib[0]):]
        print("Ave error: ", np.mean(ave_error))
        print("Actual param vs calib param: ", np.linalg.norm(param_gts[0]-param_calib[0]), np.linalg.norm(param_gts[1]-param_calib[1]))
        u,s,v=np.linalg.svd(J_all)
        print("J rank (numpy) / Total singular values: %d/%d"%(np.linalg.matrix_rank(J_all),len(s)))
        print("J condition number: ", s[0]/s[-1])
        print("============")
        ave_error_iter.append(np.mean(ave_error))
        param1_norm_iter.append(np.linalg.norm(param_gts[0]-param_calib[0]))
        param2_norm_iter.append(np.linalg.norm(param_gts[1]-param_calib[1]))
        
        # visualize jacobian matrix
        # plt.scatter(np.arange(len(s)),s)
        # plt.title("Singular values")
        # plt.show()
        
        # plt.matshow(v.T)
        # plt.title("Right singular vectors")
        # plt.colorbar()
        # plt.show()
        
    except KeyboardInterrupt:
        break

plt.plot(ave_error_iter)
plt.title("Ave error")
plt.xlabel("iter")
plt.ylabel("nu error norm")
plt.show()

plt.plot(param1_norm_iter, label='robot 1 params')
plt.plot(param2_norm_iter, label='robot 2 params')
plt.xlabel("iter")
plt.legend()
plt.title("Param Error norm")
plt.show()
exit()
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
            dpidparamR1,dpidparamR2, t1_t2_i = get_dPt1t2dparam([joint_data[0][data_i],joint_data[1][data_i]], param_calib, robots, TR2R1)
            ## get p_j, and all gradients
            dpjdparamR1,dpjdparamR2, t1_t2_j = get_dPt1t2dparam([joint_data[0][data_j],joint_data[1][data_j]], param_calib, robots, TR2R1)
            ## add to gradient
            G1.extend(dpidparamR1-dpjdparamR1)
            G2.extend(dpidparamR2-dpjdparamR2)
            ## add error vec
            error_vec.extend(t1_t2_i.p-t1_t2_j.p)
            
            # print(robots[0].fwd(joint_data[0][data_i]))
            # print(robots[1].fwd(joint_data[1][data_i], world=True))
            # print(t1_t2_i)
            # print("==")
            # print(robots[0].fwd(joint_data[0][data_j]))
            # print(t1_t2_j)
            # print(np.round(error_vec,3))
            # print(np.round(dpidparamR1,3).T)
            # print(np.round(dpjdparamR1,3).T)
            # # print(np.round(G1,3).T)
            # input('====================')
    
    G1 = np.array(G1)
    G2 = np.array(G2)
    error_vec = np.array(error_vec)
    G1_m = deepcopy(G1)
    G2_m = deepcopy(G2)
    G1_m[:,-12:] *= 0.001
    G2_m[:,-12:] *= 0.001
    
    # plt.matshow(G1_m)
    # plt.colorbar()
    # plt.show()
    
    param_calib[0] = param_calib[0] - alpha*np.dot(error_vec,G1_m)
    # param_calib[1] = param_calib[1] - alpha*np.dot(error_vec,G2_m)
    print("error norm:", np.linalg.norm(np.array(error_vec)))