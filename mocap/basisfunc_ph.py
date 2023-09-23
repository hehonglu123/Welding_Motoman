import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
sys.path.append('../toolbox/')
from robot_def import *
from matplotlib import pyplot as plt

from PH_interp import *

ph_dataset_date='0801'
test_dataset_date='0801'
PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_R1/train_data_'
test_data_dir='kinematic_raw_data/test'+test_dataset_date+'/'

config_dir='../config/'
### z pointing x-axis (with 22 deg angle), y pointing y-axis
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',d=15,\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_'+ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_'+ph_dataset_date+'_marker_config.yaml')

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
# ph_param_lin=PH_Param()
# ph_param_lin.fit(PH_q,method='linear')

## nominal
ph_q_nom = []
for j in range(6):
    ph_q_nom = np.append(ph_q_nom,origin_P[:,j])
for j in range(6):
    ph_q_nom = np.append(ph_q_nom,origin_H[:,j])
    
###### define basis function ######
basis_func=[]
basis_func.append(lambda q2,q3,a: np.sin(a*q2))
basis_func.append(lambda q2,q3,a: np.cos(a*q2))
basis_func.append(lambda q2,q3,a: np.sin(a*q3))
basis_func.append(lambda q2,q3,a: np.cos(a*q3))
basis_func.append(lambda q2,q3,a: np.sin(a*(q2+q3)))
basis_func.append(lambda q2,q3,a: np.cos(a*(q2+q3)))
basis_function_num=2
###################################

diff_ph_q = []
basis_func_q2q3=[]
for q in PH_q.keys():
    diff_ph_q.append(np.append(PH_q[q]['P'][:,:-1].T.flatten(),PH_q[q]['H'].T.flatten())-ph_q_nom)
    this_basis = []
    for a in range(1,basis_function_num+1):
        for func in basis_func:
            # print(np.degrees([q[0],q[1]]))
            # print(func(q[0],q[1],a))
            # input("========================")
            this_basis.append(func(q[0],q[1],a))
    this_basis.append(1) # constant function
    basis_func_q2q3.append(this_basis)

diff_ph_q=np.array(diff_ph_q).T
basis_func_q2q3=np.array(basis_func_q2q3).T

# print(diff_ph_q.shape)
# print("SVD")
# U,S,V = np.linalg.svd(diff_ph_q)
# plt.plot(np.log10(S),'-o')
# plt.xlabel('Singular Value Index')
# plt.ylabel('Singular Value (log 10 scale)')
# plt.show()

coeff_A = diff_ph_q@np.linalg.pinv(basis_func_q2q3)
residual_poses = np.squeeze(np.linalg.norm(diff_ph_q-coeff_A@basis_func_q2q3,axis=0))
# residual_PH = np.squeeze(np.mean(np.fabs(diff_ph_q-coeff_A@basis_func_q2q3),axis=1))
residual_PH = np.squeeze(np.sqrt(np.mean(np.fabs(diff_ph_q-coeff_A@basis_func_q2q3)**2,axis=1)))
residual_poses_mean = np.mean(residual_poses)
residual_PH_mean = np.mean(residual_PH)

print("Resdual mean",residual_poses_mean)
# plt.plot(residual_poses,'-o')
# plt.xlabel("Poses")
# plt.ylabel("Norm of PH Error")
# plt.title("Residual by Poses")
# plt.show()

print("Resdual mean",residual_PH_mean)
# plt.plot(residual_PH,'-o')
# plt.xlabel("P1~H6")
# plt.ylabel("RMSE")
# plt.title("Residual")
# plt.show()

### SVD on coefficients of basis function
U,S,VT = np.linalg.svd(coeff_A)

print(S)
print(VT.shape)
print(VT[:2])
print(VT[-2:])

plt.imshow(np.fabs(VT))
plt.colorbar()
plt.title("V Transpose (Abs)",fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

plt.plot(np.log10(S),'-o')
plt.xticks(np.arange(0,len(S),2),fontsize=15)
plt.xlabel('Singular Value Index',fontsize=15)
plt.ylabel('Singular Value (log 10 scale)',fontsize=15)
plt.yticks(fontsize=15)
plt.title("Singular Values of Coefficient Matrix (A)",fontsize=20)
plt.show()