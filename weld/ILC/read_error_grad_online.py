from copy import deepcopy
from pathlib import Path
import pickle
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

data_dir='data/weld_scan_error_grad_2023_11_07_15_39_30/'
# data_dir='data/weld_scan_error_smooth_2023_11_07_16_48_39/'
seg_dist=1.6
dh=2.5
yk_d=[dh]
smooth=False
if 'smooth' in data_dir or smooth:
    yk0=np.loadtxt(data_dir+'iteration_0/yk.csv',delimiter=',')
    yk_d=[np.mean(yk0)]
    print(np.mean(yk0))
ipm_weld=250
ipm_for_calculation=210
uk_nom=dh2v_loglog(dh,mode=ipm_for_calculation)

total_iteration=6

show_pcd=False
show_yk=True
show_uk=True
show_actual_uk=False
show_grad=False
show_norm=True
show_norm_parts=False

## show pcd of iterations
show_pcd=[]
if show_pcd:
    for iter_read in range(total_iteration):
        if iter_read in show_pcd:
            pcd_all=o3d.geometry.PointCloud()
            for layer in range(2):
                pcd=o3d.io.read_point_cloud(data_dir+'iteration_'+str(iter_read)+'/layer_'+str(layer)+'/processed_pcd.pcd')
                if layer==0:
                    pcd=pcd.paint_uniform_color([0.1,0.1,0.1])
                elif layer==1:
                    pcd=pcd.paint_uniform_color([1,0,0])
                pcd_all+=pcd
            visualize_pcd([pcd_all])

## show output of iterations (yk)
if show_yk:
    plt.axhline(y = yk_d[0], color = 'r', linestyle = '-',label='desired h') 
    for iter_read in range(total_iteration):

        yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
        # yk=yk[4:-9]
        # try:
        #     yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
        # except:
        #     yk_prime=deepcopy(yk)
        plt.plot(np.arange(len(yk))*seg_dist,yk,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
        # plt.scatter(np.arange(len(yk_prime)),yk_prime)
    plt.xlabel("L (mm)",fontsize=14)
    plt.ylabel("Height (mm)",fontsize=14)
    plt.legend()
    plt.title("Output height of iterations",fontsize=14)
    plt.show()

if show_actual_uk:
    for iter_read in range(total_iteration):
        uk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/input_uk.csv',delimiter=',')
        uk=np.append(uk[0],uk)
        profile_dh=np.load(data_dir+'iteration_'+str(iter_read)+'/layer_1/height_profile.npy')
        weld_js_exe=np.loadtxt(data_dir + 'iteration_'+str(iter_read)+'/layer_1/weld_js_exe.csv',delimiter=',')
        weld_stamps=np.loadtxt(data_dir + 'iteration_'+str(iter_read)+'/layer_1/weld_robot_stamps.csv',delimiter=',')
        lam_exe = calc_lam_js(weld_js_exe[:,:6],robot_weld)
        ldot=np.diff(lam_exe)/np.diff(weld_stamps)
        ldot=np.append(ldot[0],ldot)
        ldot=moving_average(ldot,padding=True)
        plt.plot(profile_dh[:,0],uk,marker='o')
        plt.plot(lam_exe,ldot,marker='o')
        plt.title("Iteration "+str(iter_read))
        plt.show()

## show input of iterations (uk)
if show_uk:
    plt.axhline(y = uk_nom, color = 'r', linestyle = '-',label='desired h') 
    for iter_read in range(total_iteration):

        uk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/input_uk.csv',delimiter=',')
        # try:
        #     yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
        # except:
        #     yk_prime=deepcopy(yk)
        plt.plot(np.arange(len(uk))*seg_dist,uk,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
        # plt.scatter(np.arange(len(yk_prime)),yk_prime)
    plt.xlabel("L (mm)",fontsize=14)
    plt.ylabel("Height (mm)",fontsize=14)
    plt.legend()
    plt.title("Input u of iterations",fontsize=14)
    plt.show()

## show gradient
if show_grad:
    plt.axhline(y = 0, color = 'r', linestyle = '-',label='zero')
    for iter_read in range(total_iteration):

        yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
        ek=yk-yk_d[0]
        gradient_direction=deepcopy(ek)*-1 # negative direction
        plt.plot(np.arange(len(gradient_direction))*seg_dist,gradient_direction,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
        # plt.scatter(np.arange(len(yk_prime)),yk_prime)
    plt.xlabel("L (mm)",fontsize=14)
    plt.ylabel("Height (mm)",fontsize=14)
    plt.legend()
    plt.title("gradient direction of iterations",fontsize=14)
    plt.show()

## show L2 norm of iterations
if show_norm:
    norm_iter=[]
    for iter_read in range(total_iteration):
        yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
        # yk=yk[4:-9]
        norm_iter.append(np.linalg.norm(yk-yk_d[0]))
    plt.plot(norm_iter,'-o',markersize=6,linewidth=2)
    plt.xlabel("Iteration",fontsize=14)
    plt.ylabel("Error Norm (mm)",fontsize=14)
    plt.legend()
    plt.title("Error norm of iterations",fontsize=14)
    plt.show()

# norm of each part
if show_norm_parts:
    norm_iter=[]
    for iter_read in range(total_iteration):
        yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
        yk=np.reshape(yk,(9,5))
        norm_iter.append(np.squeeze(np.linalg.norm(yk-yk_d[0],axis=1)))
    norm_iter=np.array(norm_iter)
    draw_part=[0,1,4,7,8]
    for part_n in range(9):
        if part_n in draw_part:  
            plt.plot(norm_iter[:,part_n],'-o',markersize=6,linewidth=2,label='part '+str(part_n))
    plt.xlabel("Iteration",fontsize=14)
    plt.ylabel("Error Norm of parts (mm)",fontsize=14)
    plt.legend()
    plt.title("Error norm of parts of iterations",fontsize=14)
    plt.show()