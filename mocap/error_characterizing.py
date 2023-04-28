from copy import deepcopy
import sys
sys.path.append('../toolbox/')
from utils import *
from robot_def import * 
from general_robotics_toolbox import *

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from fitting_3dcircle import *

def detect_axis(points,rough_axis_direction,calib_marker_ids):

    point_N=-1

    all_normals=[]
    all_centers=[]
    for i in range(len(calib_marker_ids)):
        center, normal = fitting_3dcircle(points[calib_marker_ids[i]][:point_N])
        if np.sum(np.multiply(normal,rough_axis_direction)) < 0:
            normal = -1*normal
        all_normals.append(normal)
        all_centers.append(center)
    normal_mean = np.mean(all_normals,axis=0)
    normal_mean = normal_mean/np.linalg.norm(normal_mean)
    center_mean = np.mean(all_centers,axis=0)

    return center_mean,normal_mean

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

# tool_markers_id=['marker4_rigid3','marker3_rigid3','marker2_rigid3','marker1_rigid3']
tool_markers_id=['marker4_rigid3']

# choose an axis
axis_j = 1

# only R matter
nominal_robot_base_R = np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1]])
H_nom = np.matmul(nominal_robot_base_R,robot_weld.robot.H)
H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)

sigma=0.8
sample_T = 20+1

all_normal_6=[]
all_axis_p_6=[]
for axis_j in range(6):
    # read raw data
    raw_data_dir = 'PH_raw_data/train_data'
    with open(raw_data_dir+'_'+str(axis_j+1)+'.pickle', 'rb') as handle:
        curve_p = pickle.load(handle)
    # convert to a usual frame
    R_zaxis_up = np.array([[0,0,1],
                            [1,0,0],
                            [0,1,0]])
    curve_p_sample={}
    for marker_id in tool_markers_id:
        curve_p[marker_id] = np.array(curve_p[marker_id])
        curve_p[marker_id] = np.matmul(R_zaxis_up,curve_p[marker_id].T).T
        curve_p_sample[marker_id] = deepcopy(curve_p[marker_id])

    all_normal=[]
    all_axis_p = []
    for T in range(sample_T):
        print("Sample T:",T)
        for marker_id in tool_markers_id:
            for xyz in range(3):
                if T<sample_T-1:
                    curve_p_sample[marker_id][xyz] = np.random.normal(curve_p[marker_id][xyz],sigma)
                else:
                    print("No noise")
                    curve_p_sample[marker_id][xyz] = deepcopy(curve_p[marker_id][xyz])

        # detect axis
        this_axis_p,this_axis_normal = detect_axis(curve_p_sample,H_nom[:,0],tool_markers_id)

        all_normal.append(this_axis_normal)
        all_axis_p.append(this_axis_p)

    # print(np.std(all_axis_p,axis=0))
    # print(np.std(all_normal,axis=0))
    all_axis_p_mean = np.mean(all_axis_p,axis=0)
    all_normal_mean = np.mean(all_normal,axis=0)

    # print(all_axis_p-all_axis_p_mean)
    # print(all_normal-all_normal_mean)

    all_normal_6.append(np.array(all_normal))
    all_axis_p_6.append(np.array(all_axis_p))

    ### validation data
    # for valid_i in range(2):
    #     print("Validation Set :",valid_i)
    #     raw_data_dir = 'PH_raw_data/valid_data_'+str(valid_i+1)
    #     with open(raw_data_dir+'_'+str(axis_j)+'.pickle', 'rb') as handle:
    #         curve_p = pickle.load(handle)
    #     # convert to a usual frame
    #     R_zaxis_up = np.array([[0,0,1],
    #                             [1,0,0],
    #                             [0,1,0]])
    #     for marker_id in tool_markers_id:
    #         curve_p[marker_id] = np.array(curve_p[marker_id])
    #         curve_p[marker_id] = np.matmul(R_zaxis_up,curve_p[marker_id].T).T

    #     # detect axis
    #     this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,0],tool_markers_id)

    #     print("Difference between validation and train")
    #     print(this_axis_p-all_axis_p_mean)
    #     print(this_axis_normal-all_normal_mean)
    #     print(np.degrees(np.arccos(np.dot(this_axis_normal,all_normal_mean))))

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # for i in range(len(all_axis_p)):
    #     start_p = all_axis_p[i]-all_normal[i]*1000
    #     end_p = all_axis_p[i]+all_normal[i]*1000
    #     ax.plot([start_p[0],end_p[0]], [start_p[1],end_p[1]], [start_p[2],end_p[2]])
    # plt.show()

all_normal_6=np.array(all_normal_6)
all_axis_p_6=np.array(all_axis_p_6)
all_P2=[]
all_P3=[]
all_P4=[]
all_P5=[]
all_P6=[]
all_H1=[]
all_H2=[]
all_H3=[]
all_H4=[]
all_H5=[]
all_H6=[]
for T in range(sample_T):
    H = deepcopy(all_normal_6[:,T,:])
    H=H.T
    H_point = deepcopy(all_axis_p_6[:,T,:])
    H_point=H_point.T

    # rotate R
    z_axis = H[:,0]
    y_axis = H[:,1]
    y_axis = y_axis-np.dot(z_axis,y_axis)*z_axis
    y_axis = y_axis/np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis,z_axis)
    x_axis = x_axis/np.linalg.norm(x_axis)
    R = np.array([x_axis,y_axis,z_axis])

    H = np.matmul(R,H)
    # print(H.T)
    H_point = (H_point.T-H_point[:,0]).T
    H_point = np.matmul(R,H_point)

    # joint 1 is the closest point on H1 to H2
    ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,0],-H[:,1]]).T),
                                        -(H_point[:,0]-H_point[:,1]))
    j1_center = H_point[:,0]+ab_coefficient[0]*H[:,0]
    j2_center = H_point[:,1]+ab_coefficient[1]*H[:,1]
    # print("P2:",j2_center-j1_center)

    k=(j2_center[1]-H_point[1,2])/H[1,2]
    j3_center = H_point[:,2]+k*H[:,2]
    # print("P3:",j3_center-j2_center)

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
    # print("P4:",j4_center-j3_center)
    # print("P5:",j5_center-j4_center)
    # print("P6:",j6_center-j5_center)

    all_P2.append(j2_center-j1_center)
    all_P3.append(j3_center-j2_center)
    all_P4.append(j4_center-j3_center)
    all_P5.append(j5_center-j4_center)
    all_P6.append(j6_center-j5_center)
    all_H1.append(H[:,0])
    all_H2.append(H[:,1])
    all_H3.append(H[:,2])
    all_H4.append(H[:,3])
    all_H5.append(H[:,4])
    all_H6.append(H[:,5])

print("P Deviation")
print(np.std(all_P2[:-1],axis=0))
print(np.std(all_P3[:-1],axis=0))
print(np.std(all_P4[:-1],axis=0))
print(np.std(all_P5[:-1],axis=0))
print(np.std(all_P6[:-1],axis=0))
print("P Bias")
print(np.mean(all_P2[:-1],axis=0)-all_P2[-1])
print(np.mean(all_P3[:-1],axis=0)-all_P3[-1])
print(np.mean(all_P4[:-1],axis=0)-all_P4[-1])
print(np.mean(all_P5[:-1],axis=0)-all_P5[-1])
print(np.mean(all_P6[:-1],axis=0)-all_P6[-1])
print("H Deviation")
print(np.std(all_H1[:-1],axis=0))
print(np.std(all_H2[:-1],axis=0))
print(np.std(all_H3[:-1],axis=0))
print(np.std(all_H4[:-1],axis=0))
print(np.std(all_H5[:-1],axis=0))
print(np.std(all_H6[:-1],axis=0))
print("H Bias")
print(np.mean(all_H1[:-1],axis=0)-all_H1[-1])
print(np.mean(all_H2[:-1],axis=0)-all_H2[-1])
print(np.mean(all_H3[:-1],axis=0)-all_H3[-1])
print(np.mean(all_H4[:-1],axis=0)-all_H4[-1])
print(np.mean(all_H5[:-1],axis=0)-all_H5[-1])
print(np.mean(all_H6[:-1],axis=0)-all_H6[-1])