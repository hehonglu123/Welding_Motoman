import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

from qpsolvers import solve_qp

from PH_interp import *

ph_dataset_date='0516'
test_dataset_date='0516'
PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_R1/train_data_'
test_data_dir='kinematic_raw_data/test'+test_dataset_date+'/'

config_dir='../config/'
### z pointing x-axis (with 22 deg angle), y pointing y-axis
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',d=15,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')

# print(robot_weld.fwd(np.zeros(6)))
# exit()

T_base_basemarker = robot_weld.T_base_basemarker
T_basemarker_base = T_base_basemarker.inv()

origin_P = deepcopy(robot_weld.robot.P)
origin_H = deepcopy(robot_weld.robot.H)
origin_flange=deepcopy(robot_weld.robot.T_flange)
origin_R_tool = deepcopy(robot_weld.robot.R_tool)
origin_p_tool = deepcopy(robot_weld.robot.p_tool)

#### using rigid body
# robot_weld.T_tool_toolmarker=Transform(np.eye(3),[0,0,0])
#### using tool

with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
ph_param_lin=PH_Param()
ph_param_lin.fit(PH_q,method='linear')

test_robot_q = np.loadtxt(test_data_dir+'robot_q_align.csv',delimiter=',')

all_error_norm=[]
all_qdiff_norm=[]
all_qdiff_ave=[]
for test_q in test_robot_q:
    # testq_i = np.random.randint(0,len(test_robot_q))
    # test_q=test_robot_q[testq_i]

    # print("Test Joint Angle:",np.degrees(test_q))

    J=robot_weld.jacobian(test_q)
    u,s,v=np.linalg.svd(J)
    print("Min S",np.min(s))
    if np.min(s)<0.1:
        print("Skip Test Joint Angle:",np.degrees(test_q))
        continue
    
    # robot_T = robot_weld.fwd(test_q)
    # print("Nominal Transform:",robot_T)
    opt_P,opt_H = ph_param_lin.predict(test_q[1:3])

    robot_weld.robot.P=deepcopy(opt_P)
    robot_weld.robot.H=deepcopy(opt_H)
    robot_weld.robot.T_flange = deepcopy(robot_weld.T_tool_flange)
    robot_weld.robot.R_tool = deepcopy(robot_weld.T_tool_toolmarker.R)
    robot_weld.robot.p_tool = deepcopy(robot_weld.T_tool_toolmarker.p)
    robot_T = robot_weld.fwd(test_q)
    target_T = deepcopy(robot_T)
    # print("Target Transform:",robot_T)

    robot_weld.robot.P=deepcopy(origin_P)
    robot_weld.robot.H=deepcopy(origin_H)
    robot_weld.robot.T_flange = deepcopy(origin_flange)
    robot_weld.robot.R_tool=deepcopy(origin_R_tool)
    robot_weld.robot.p_tool=deepcopy(origin_p_tool)
    robot_nom_q = robot_weld.inv(robot_T.p,robot_T.R,test_q)[0]
    # print("IK using nominal PH:",np.degrees(robot_nom_q))

    ### solve IK with new PH
    q_sol = deepcopy(robot_nom_q)
    robot_weld.robot.T_flange = deepcopy(robot_weld.T_tool_flange)
    robot_weld.robot.R_tool = deepcopy(robot_weld.T_tool_toolmarker.R)
    robot_weld.robot.p_tool = deepcopy(robot_weld.T_tool_toolmarker.p)

    ## qp IK
    Kw=0.1
    Kq=0.1*np.eye(6)
    lim_factor=np.radians(1)
    alpha=1
    error_fb=999
    # while error_fb>0.0001:
    for i in range(51):
        
        ## find the PH
        opt_P,opt_H = ph_param_lin.predict(q_sol[1:3])
        robot_weld.robot.P=deepcopy(opt_P)
        robot_weld.robot.H=deepcopy(opt_H)
        robot_T = robot_weld.fwd(q_sol)
        ## error=euclideans norm (p)+forbinius norm (R)
        p_norm= np.linalg.norm(robot_T.p-target_T.p)
        R_norm=np.linalg.norm(np.matmul(robot_T.R.T,target_T.R)-np.eye(3))
        error_fb=p_norm+R_norm

        # print(error_fb)
        if error_fb>1000:
            print("Error too large:",error_fb)
            raise AssertionError
        
        ## prepare Jacobian matrix w.r.t positioner
        J=robot_weld.jacobian(q_sol)
        J_all_p=J[3:,:]
        J_all_R=J[:3,:]

        H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
        H=(H+np.transpose(H))/2

        vd = target_T.p-robot_T.p
        omega_d=s_err_func(robot_T.R@target_T.R.T)
        # omega_d=s_err_func(self.scan_R[i].T@Rt1_t2)

        f=-np.dot(np.transpose(J_all_p),vd)+Kw*np.dot(np.transpose(J_all_R),omega_d)
        qdot=solve_qp(H,f,lb=robot_weld.lower_limit-q_sol+lim_factor*np.ones(6),\
                    ub=robot_weld.upper_limit-q_sol-lim_factor*np.ones(6),solver='quadprog')
        
        q_sol=q_sol+alpha*qdot

    # print("IK using Config-dependent PH and qp:",np.degrees(q_sol))
    # print("Joint Differences:",np.degrees(np.fabs(q_sol-test_q)))
    # print("Cartesian P differences:",p_norm)
    # print("Cartesian R differences:",R_norm)

    all_qdiff_norm.append(np.linalg.norm(np.degrees(q_sol-test_q)))
    all_qdiff_ave.append(np.mean(np.fabs(np.degrees(q_sol-test_q))))
    all_error_norm.append(error_fb)

plt.plot(all_error_norm,'.')
plt.title("Euclidean Norm + R Matrix Frobenius norm")
plt.show()

plt.plot(all_qdiff_norm,'.')
plt.title("Euclidean Norm Error of QP Joint Angle Solution (Deg.)")
plt.show()

plt.plot(all_qdiff_ave,'.')
plt.title("Ave Error of QP Joint Angle Solution (Deg.)")
plt.show()