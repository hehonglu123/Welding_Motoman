from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../redundancy_resolution/')
from utils import *
from robot_def import * 
from general_robotics_toolbox import *

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
        if np.sum(np.multiply(normal,rough_axis_direction)) < 0:
            normal = -1*normal
        all_normals.append(normal)
        all_centers.append(center)
    normal_mean = np.mean(all_normals,axis=0)
    normal_mean = normal_mean/np.linalg.norm(normal_mean)
    center_mean = np.mean(all_centers,axis=0)

    return center_mean,normal_mean

config_dir='../config/'
base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=base_marker_config_file,tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

# only R matter
nominal_robot_base = Transform(np.array([[1,0,0],
                                        [0,1,0],
                                        [0,0,1]]),[0,0,0]) 
H_nom = np.matmul(nominal_robot_base.R,robot_weld.robot.H)
H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)

# all_datasets=['train_data','valid_data_1','valid_data_2']
all_datasets=['train_data']
for dataset in all_datasets:
    print(dataset)
    for j in range(6):
        # read raw data
        raw_data_dir = 'PH_raw_data/'+dataset
        with open(raw_data_dir+'_'+str(j+1)+'.pickle', 'rb') as handle:
            curve_p = pickle.load(handle)
        # convert to a usual frame
        R_zaxis_up = np.array([[0,0,1],
                                [1,0,0],
                                [0,1,0]])
        for marker_id in curve_p.keys():
            curve_p[marker_id] = np.array(curve_p[marker_id])
            curve_p[marker_id] = np.matmul(R_zaxis_up,curve_p[marker_id].T).T

        # detect axis
        this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,j],robot_weld.tool_markers_id)
        H_act[:,j] = this_axis_normal
        axis_p[:,j] = this_axis_p

    H = H_act
    H_point = axis_p
    for i in range(6):
        H[:,i]=H[:,i]/np.linalg.norm(H[:,i])

    # print(np.round(H,5).T)

    # rotate R
    z_axis = H[:,0]
    y_axis = H[:,1]
    y_axis = y_axis-np.dot(z_axis,y_axis)*z_axis
    y_axis = y_axis/np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)
    R = np.array([x_axis,y_axis,z_axis])

    T_frame_mocap = Transform(R,np.matmul(R,-H_point[:,0]))
    H = np.matmul(R,H)
    for i in range(6):
        H_point[:,i] = np.matmul(T_frame_mocap.R,H_point[:,i])+T_frame_mocap.p
    
    P_marker_id = 'rigid3_marker1'
    raw_data_dir = 'PH_raw_data/'+dataset
    with open(raw_data_dir+'_zero_config.pickle', 'rb') as handle:
        curve_p = pickle.load(handle)
    tcp_frame = np.matmul(T_frame_mocap.R,np.mean(curve_p[P_marker_id],axis=0))+T_frame_mocap.p

    # print(np.round(H,5).T)
    # H_point = (H_point.T-H_point[:,0]).T
    # H_point = np.matmul(R,H_point)
    # print(H_point.T)

    diff_H = np.linalg.norm(robot_weld.robot.H-H,axis=0)
    # print(diff_H)

    # for i in range(6):
    #     print(np.degrees(np.arccos(np.dot(H[:,i],robot_weld.robot.H[:,i])/(np.linalg.norm(H[:,i])*np.linalg.norm(robot_weld.robot.H[:,i])))))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(6):
    #     start_p = H_point[:,i]-H[:,i]*1000
    #     end_p = H_point[:,i]+H[:,i]*1000
    #     ax.plot([start_p[0],end_p[0]], [start_p[1],end_p[1]], [start_p[2],end_p[2]], label='axis '+str(i+1))
    # plt.legend()
    # plt.show()

 
    # get P

    # joint 1 is the closest point on H1 to H2
    ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,0],-H[:,1]]).T),
                                        -(H_point[:,0]-H_point[:,1]))
    j1_center = H_point[:,0]+ab_coefficient[0]*H[:,0]
    j2_center = H_point[:,1]+ab_coefficient[1]*H[:,1]
    print("P2:",j2_center-j1_center)

    k=(j2_center[1]-H_point[1,2])/H[1,2]
    j3_center = H_point[:,2]+k*H[:,2]
    print("P3:",j3_center-j2_center)

    # p456
    # ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,0],-H[:,1]]).T),
    #                                     -(H_point[:,0]-H_point[:,1]))
    # j1_center = H_point[:,0]+ab_coefficient[0]*H[:,0]
    # j2_center = H_point[:,1]+ab_coefficient[1]*H[:,1]

    # p5
    k=(j3_center[1]-H_point[1,4])/H[1,4]
    j5_center = H_point[:,4]+k*H[:,4]
    # p4
    k=(j5_center[0]-H_point[0,3])/H[0,3]
    j4_center = H_point[:,3]+k*H[:,3]
    # p6
    k=(j5_center[0]-H_point[0,5])/H[0,5]
    j6_center = H_point[:,5]+k*H[:,5]
    print("P4:",j4_center-j3_center)
    print("P5:",j5_center-j4_center)
    print("P6:",j6_center-j5_center)
    print(j1_center)
    print(j6_center)
    print(j6_center-j1_center)
    print("====================")

    P=deepcopy(H)
    P[:,0]=np.array([0,0,0])
    P[:,1]=j2_center-j1_center
    P[:,2]=j3_center-j2_center
    P[:,3]=j4_center-j3_center
    P[:,4]=j5_center-j4_center
    P[:,5]=j6_center-j5_center
    P[:,6]=tcp_frame-j6_center

with open(base_marker_config_file,'r') as file:
    base_marker_data = yaml.safe_load(file)
base_marker_data['H']=[]
base_marker_data['P']=[]
for j in range(len(H[0])):
    this_H = {}
    this_H['x']=float(H[0,j])
    this_H['y']=float(H[1,j])
    this_H['z']=float(H[2,j])
    base_marker_data['H'].append(this_H)
for j in range(len(P[0])):
    this_P = {}
    this_P['x']=float(P[0,j])
    this_P['y']=float(P[1,j])
    this_P['z']=float(P[2,j])
    base_marker_data['P'].append(this_P)
with open(base_marker_config_file,'w') as file:
    yaml.safe_dump(base_marker_data,file)