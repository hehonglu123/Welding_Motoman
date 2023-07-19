from copy import deepcopy
import sys
sys.path.append('../../toolbox/')
sys.path.append('../')
from robot_def import * 
from general_robotics_toolbox import *

from pandas import read_csv
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from fitting_3dcircle import *

def detect_axis(points,rough_axis_direction,calib_marker_ids):

    all_normals=[]
    all_centers=[]
    for i in range(len(calib_marker_ids)):
        center, normal = fitting_3dcircle(points[calib_marker_ids[i]])

        if calib_marker_ids[i]=='marker8_rigid4':
            print("Radius:",np.mean(np.linalg.norm(points[calib_marker_ids[i]]-center,axis=1)))
            print("Radius std:",np.std(np.linalg.norm(points[calib_marker_ids[i]]-center,axis=1)))

        if np.sum(np.multiply(normal,rough_axis_direction)) < 0:
            normal = -1*normal
        all_normals.append(normal)
        all_centers.append(center)
    normal_mean = np.mean(all_normals,axis=0)
    normal_mean = normal_mean/np.linalg.norm(normal_mean)
    center_mean = np.mean(all_centers,axis=0)

    return center_mean,normal_mean

def to_frame(curve_p,curve_R,mocap_stamps,target_frame,markers_id):
    curve_p_frame = {}
    curve_R_frame = {}
    mocap_stamps_frame = []
    for sample_i in range(len(mocap_stamps[target_frame])):
        basemarker_stamp = mocap_stamps[target_frame][sample_i]
        basemarker_T = Transform(curve_R[target_frame][sample_i],
                                curve_p[target_frame][sample_i]).inv()
        for i in range(len(markers_id)):
            this_k = np.argwhere(mocap_stamps[markers_id[i]]==basemarker_stamp)
            if len(this_k)!=1 or len(this_k[0])!=1:
                continue
            this_k=this_k[0][0]
            mocap_stamps_frame.append(basemarker_stamp)
            if markers_id[i] not in curve_p_frame.keys():
                curve_p_frame[markers_id[i]] = []
                curve_R_frame[markers_id[i]] = []
            curve_p_frame[markers_id[i]].append(np.matmul(basemarker_T.R,curve_p[markers_id[i]][this_k])\
                                                            + basemarker_T.p)
            curve_R_frame[markers_id[i]].append(np.matmul(basemarker_T.R,curve_R[markers_id[i]][this_k]))

    return curve_p_frame,curve_R_frame,mocap_stamps_frame

# test_a = np.loadtxt('data/071423_GP1_phasespace/PhaseSpace/J6cal_20230714.csv',\
#             skiprows=8)
test_a = read_csv('data/071423_GP1_phasespace/PhaseSpace/J6cal_20230714.csv')

exit()


config_dir='config/'
robot_type='GP1'

dataset_date='0714'

if robot_type=='GP1':
    base_marker_config_file=config_dir+'m10ia_marker_config.yaml'
    tool_marker_config_file=config_dir+'ge_pointer1_marker_config.yaml'
    robot=robot_obj('M10iA',def_path=config_dir+'FANUC_m10ia_robot_default_config.yml',tool_file_path=config_dir+'laser_ge.csv',\
    base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

    # only R matter
    nominal_robot_base = Transform(np.array([[0,-1,0],
                                            [0,0,1],
                                            [-1,0,0]]),[0,0,0]) 
    H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)

    jN=6

    output_base_marker_config_file = config_dir+'m10ia_'+dataset_date+'_marker_config.yaml'
    output_tool_marker_config_file = config_dir+'ge_pointer1_'+dataset_date+'_marker_config.yaml'
    
elif robot_type=='GP2':
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
    tool_marker_config_file=config_dir+'weldgun_marker_config.yaml'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
    base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

    # only R matter
    nominal_robot_base = Transform(np.array([[0,1,0],
                                            [0,0,1],
                                            [1,0,0]]),[0,0,0]) 
    H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)

    jN=6
    
    output_base_marker_config_file = config_dir+'MA2010_'+dataset_date+'_marker_config.yaml'
    output_tool_marker_config_file = config_dir+'weldgun_'+dataset_date+'_marker_config.yaml'
    

H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)