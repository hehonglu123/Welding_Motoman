from copy import deepcopy
import sys
sys.path.append('../../toolbox/')
sys.path.append('../')
from robot_def import * 
from general_robotics_toolbox import *

import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from fitting_3dcircle import *

marker_group={'rigid1':[1,2,3,4,5],'rigid2':[16,17,18,19,20,21,22,23],'rigid3':[8,9,10,11,12,13,14,15]}

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

def read_points(filename):
    
    data_curve=np.loadtxt(filename,delimiter=',')
    curve_p = {}
    curve_R = {}
    mocap_stamps = {}
    
    for data in data_curve:
        marker_id=int(data[1])
        for marker_rigid in marker_group.keys():
            if marker_id in marker_group[marker_rigid]:
                marker_id = marker_rigid+'_marker'+str(marker_id)
                break
        if marker_id not in curve_p.keys():
            curve_p[marker_id]=[]
            curve_R[marker_id]=[]
            mocap_stamps[marker_id]=[]
        curve_p[marker_id].append(np.array(data[2:5]))
        curve_R[marker_id].append(q2R(data[5:9]))
        mocap_stamps[marker_id].append(data[0])
    
    return curve_p,curve_R,mocap_stamps

config_dir='config/'
robot_type='GP1'

dataset_date='0714'

datadir='data/071423_GP1_phasespace/'

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

## detect axis
for j in range(6):
    curve_p,curve_R,mocap_stamps=read_points(datadir+'J'+str(j+1)+'cal.csv')
    
    # detect axis
    if j!=1:
        this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,j],robot.tool_markers_id)
    else: # FANUC J2 has weird movement
        this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,j],robot.calib_markers_id)
    H_act[:,j] = this_axis_normal
    axis_p[:,j] = this_axis_p
    print("Axis",j+1,"done.")

H = H_act
H_point = axis_p
for i in range(jN):
    H[:,i]=H[:,i]/np.linalg.norm(H[:,i])

# rotate R
z_axis = H[:,0]
y_axis = H[:,1]
y_axis = y_axis-np.dot(z_axis,y_axis)*z_axis
y_axis = y_axis/np.linalg.norm(y_axis)
x_axis = np.cross(y_axis,z_axis)
x_axis = x_axis/np.linalg.norm(x_axis)
R = np.array([x_axis,y_axis,z_axis])

# get P
# joint 1 is the closest point on H1 to H2
# joint 2 is the closest point on H2 to H1
ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,0],-H[:,1]]).T),
                                    -(H_point[:,0]-H_point[:,1]))
j1_center = H_point[:,0]+ab_coefficient[0]*H[:,0]
j2_center = H_point[:,1]+ab_coefficient[1]*H[:,1]

# joint 3 is the closest point on H3 to H4
ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,2],-H[:,3]]).T),
                                    -(H_point[:,2]-H_point[:,3]))
j3_center = H_point[:,2]+ab_coefficient[0]*H[:,2]

# joint 4 is the closest point on H4 to H5
# joint 5 is the closest point on H5 to H4
ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,3],-H[:,4]]).T),
                                    -(H_point[:,3]-H_point[:,4]))
j4_center = H_point[:,3]+ab_coefficient[0]*H[:,3]
j5_center = H_point[:,4]+ab_coefficient[1]*H[:,4]

# joint 6 is the closest point on H6 to H5
ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,4],-H[:,5]]).T),
                                    -(H_point[:,4]-H_point[:,5]))
j6_center = H_point[:,5]+ab_coefficient[1]*H[:,5]

###### get robot base frame and convert to base frame from basemarker frame
T_base_basemarker = Transform(R.T,j1_center)
T_basemarker_base = T_base_basemarker.inv()
H = np.matmul(T_basemarker_base.R,H)
for i in range(jN):
    H_point[:,i] = np.matmul(T_basemarker_base.R,H_point[:,i])+T_basemarker_base.p
j1_center = np.matmul(T_basemarker_base.R,j1_center)+T_basemarker_base.p
j2_center = np.matmul(T_basemarker_base.R,j2_center)+T_basemarker_base.p
j3_center = np.matmul(T_basemarker_base.R,j3_center)+T_basemarker_base.p
j4_center = np.matmul(T_basemarker_base.R,j4_center)+T_basemarker_base.p
j5_center = np.matmul(T_basemarker_base.R,j5_center)+T_basemarker_base.p
j6_center = np.matmul(T_basemarker_base.R,j6_center)+T_basemarker_base.p
#######################################

P=np.zeros((3,7))
P[:,0]=np.array([0,0,0])
P[:,1]=j2_center-j1_center
P[:,2]=j3_center-j2_center
P[:,3]=j4_center-j3_center
P[:,4]=j5_center-j4_center
P[:,5]=j6_center-j5_center
# P[:,6]=tcp_base-j6_center
P[:,6] = np.linalg.norm(robot.robot.P[:,5]+robot.robot.P[:,6])*(-1*H[:,5])

robot_zero_config=np.radians([0,-30,-50,0,-75,-20])

zero_H=deepcopy(H)
zero_P=deepcopy(P)

R = np.eye(3)
for j in range(6,0,-1):
    R = rot(zero_H[:,j-1],-robot_zero_config[j-1])
    for i in range(j,7):
        if i!=6:
            zero_H[:,i] = R@zero_H[:,i]
        zero_P[:,i] = R@zero_P[:,i]
print('P',np.round(zero_P[:,1:7],3).T)
print('H',np.round(zero_H,3).T)