from copy import deepcopy
from pathlib import Path
import glob
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from utils import *
from lambda_calc import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weld_dh2v import *

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
from qpsolvers import solve_qp
import open3d as o3d

R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'
# 0. robots"
config_dir='../../config/'
R1_marker_dir=config_dir+'MA2010_marker_config/'
weldgun_marker_dir=config_dir+'weldgun_marker_config/'
R2_marker_dir=config_dir+'MA1440_marker_config/'
mti_marker_dir=config_dir+'mti_marker_config/'
S1_marker_dir=config_dir+'D500B_marker_config/'
S1_tcp_marker_dir=config_dir+'positioner_tcp_marker_config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=R1_marker_dir+'MA2010_'+R1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=weldgun_marker_dir+'weldgun_'+R1_ph_dataset_date+'_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=R2_marker_dir+'MA1440_'+R2_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=mti_marker_dir+'mti_'+R2_ph_dataset_date+'_marker_config.yaml')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=S1_marker_dir+'D500B_'+S1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=S1_tcp_marker_dir+'positioner_tcp_marker_config.yaml')

learn_layer=1
seg_dist=1.6
dh=2.5
ipm_weld=250
ipm_for_calculation=210

datadir_all='data/'
# datasets=['weld_scan_ilc_dh_2023_11_01_17_44_58','weld_scan_error_grad_2023_11_07_15_39_30',
#           'weld_scan_error_smooth_2023_11_07_16_48_39','weld_scan_error_smooth_dh_2023_11_08_13_25_03']
datasets=['weld_scan_error_grad_2023_11_07_15_39_30',
          'weld_scan_error_smooth_2023_11_07_16_48_39','weld_scan_error_smooth_dh_2023_11_08_13_25_03']

use_dh = True

# create data matrix
Y=[]
U=[]
for dataset in datasets:
    datadir=datadir_all+dataset+'/'
    # input data location
    if 'error' in dataset:
        input_data='input_uk.csv'
    elif 'ilc' in dataset:
        input_data='half_0/input_uk.csv'
    
    iteration_N = len(glob.glob(datadir+'iteration_*'))
    for iter_i in range(iteration_N):
        # if iter_i==0:
        #     continue
        datadir_iter_i=datadir+'iteration_'+str(iter_i)+'/'
        input_u = np.loadtxt(datadir_iter_i+input_data,delimiter=',')
        if use_dh:
            if 'dh' not in dataset:
                input_u=v2dh_loglog(input_u,mode=ipm_for_calculation)
        else:
            if 'dh' in dataset:
                input_u=dh2v_loglog(input_u,mode=ipm_for_calculation)
        # input_u=input_u[:23]
        # input_u=input_u[23:]
        # input_u = np.append(input_u,1)
        
        output_y = np.loadtxt(datadir_iter_i+'yk.csv',delimiter=',')
        # output_y=output_y[:23]
        # output_y=output_y[23:]
        U.append(input_u)
        Y.append(output_y)
Y=np.array(Y)
U=np.array(U)
# A=np.linalg.pinv(U[:,23:])@Y[:,23:]
# A=np.linalg.pinv(U[:,:23])@Y[:,:23]
# A=np.linalg.pinv(U)@Y

total_stamp = len(Y[0])
A_part=[]
start_k=0
for stamp_k in range(start_k,total_stamp):
    U_cal=U[:,start_k:]
    Y_cal=Y[:,stamp_k]
    A_length=len(U_cal[0])
    lam_A=np.ones(A_length)*10
    lam_A[stamp_k-start_k]=1
    Kq = np.diag(lam_A)
    H=np.matmul(U_cal.T,U_cal)+Kq
    H=(H+np.transpose(H))/2
    f=-np.matmul(U_cal.T,Y_cal)
    this_A=solve_qp(H,f,lb=np.zeros(A_length),solver='quadprog')
    A_part.append(this_A)

fig, ax = plt.subplots()
im = ax.matshow(A_part,cmap='RdBu')
fig.colorbar(im)
plt.show()

N=5
total_stamp = len(Y[0])
A_all=np.zeros((total_stamp,total_stamp))
A_part=[]
for stamp_k in range(total_stamp):
    print(stamp_k)
    start_stamp = max(0,stamp_k-N)
    end_stamp=min(total_stamp,stamp_k+N+1)
    A_length=end_stamp-start_stamp
    
    this_U = U[:,start_stamp:end_stamp]
    this_Y = Y[:,stamp_k]
    
    # get A
    # this_A = np.linalg.pinv(this_U)@this_Y
    lam_A = np.ones(A_length)
    lam_A[stamp_k-start_stamp]=1
    Kq = np.diag(lam_A)
    H=np.matmul(this_U.T,this_U)+Kq
    H=(H+np.transpose(H))/2
    f=-np.matmul(this_U.T,this_Y)
    this_A=solve_qp(H,f,lb=np.zeros(A_length),solver='quadprog')
    
    print(this_A.shape)
    A_all[stamp_k,start_stamp:end_stamp]=this_A
    
    this_A = np.append(np.zeros(max(0,N-stamp_k)),this_A)
    this_A = np.append(this_A,np.zeros(max(0,N-(total_stamp-1-stamp_k))))
    A_part.append(this_A)

print(A_part[-5:])
fig, ax = plt.subplots()
im = ax.matshow(A_all,cmap='RdBu')
fig.colorbar(im)
plt.show()

fig, ax = plt.subplots()
im = ax.matshow(A_part,cmap='RdBu')
fig.colorbar(im)
plt.show()


    
    
