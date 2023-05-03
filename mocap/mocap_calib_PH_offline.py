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

def read_and_convert_frame(filename,target_frame,markers_id):

    base_filename = filename+'_'+target_frame
    mocap_filename = filename+'_mocap'

    try:
        with open(base_filename+'_p.pickle', 'rb') as handle:
            curve_p = pickle.load(handle)
        with open(base_filename+'_R.pickle', 'rb') as handle:
            curve_R = pickle.load(handle)
        with open(base_filename+'_timestamps.pickle', 'rb') as handle:
            mocap_stamps = pickle.load(handle)
        for marker_id in markers_id:
            curve_p[marker_id]
    except:
        with open(mocap_filename+'_p.pickle', 'rb') as handle:
            curve_p = pickle.load(handle)
        with open(mocap_filename+'_R.pickle', 'rb') as handle:
            curve_R = pickle.load(handle)
        with open(mocap_filename+'_timestamps.pickle', 'rb') as handle:
            mocap_stamps = pickle.load(handle)

        # convert everything in basemarker frame
        curve_p,curve_R,mocap_stamps = to_frame(curve_p,curve_R,mocap_stamps,target_frame,markers_id)

        with open(base_filename+'_p.pickle', 'wb') as handle:
            pickle.dump(curve_p, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(base_filename+'_R.pickle', 'wb') as handle:
            pickle.dump(curve_R, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(base_filename+'_timestamps.pickle', 'wb') as handle:
            pickle.dump(mocap_stamps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return curve_p,curve_R,mocap_stamps

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

config_dir='../config/'
base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
tool_marker_config_file=config_dir+'weldgun_marker_config.yaml'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

# only R matter
nominal_robot_base = Transform(np.array([[0,1,0],
                                        [0,0,1],
                                        [1,0,0]]),[0,0,0]) 
H_nom = np.matmul(nominal_robot_base.R,robot_weld.robot.H)
H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)

# all_datasets=['train_data','valid_data_1','valid_data_2']
all_datasets=['test0502_noanchor/train_data']
# all_datasets=['test0502_anchor/train_data']
# P_marker_id = 'marker4_rigid3'
P_marker_id = robot_weld.tool_rigid_id
for dataset in all_datasets:
    print(dataset)
    raw_data_dir = 'PH_raw_data/'+dataset
    for j in range(6):
        # read raw data
        curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_'+str(j+1),robot_weld.base_rigid_id,robot_weld.tool_markers_id)

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
    
    # read raw data
    curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot_weld.base_rigid_id,[P_marker_id])

    tcp = np.mean(curve_p[P_marker_id],axis=0)
    R_tool_basemarker = curve_R[P_marker_id] # need this later for tool calibration

    # get P
    # joint 1 is the closest point on H1 to H2
    ab_coefficient = np.matmul(np.linalg.pinv(np.array([H[:,0],-H[:,1]]).T),
                                        -(H_point[:,0]-H_point[:,1]))
    j1_center = H_point[:,0]+ab_coefficient[0]*H[:,0]
    j2_center = H_point[:,1]+ab_coefficient[1]*H[:,1]

    ###### get robot base frame and convert to base frame from basemarker frame
    T_base_basemarker = Transform(R.T,j1_center)
    T_basemarker_base = T_base_basemarker.inv()
    H = np.matmul(T_basemarker_base.R,H)
    for i in range(6):
        H_point[:,i] = np.matmul(T_basemarker_base.R,H_point[:,i])+T_basemarker_base.p
    tcp_base = np.matmul(T_basemarker_base.R,tcp)+T_basemarker_base.p
    j1_center = np.matmul(T_basemarker_base.R,j1_center)+T_basemarker_base.p
    j2_center = np.matmul(T_basemarker_base.R,j2_center)+T_basemarker_base.p
    #######################################

    k=(j2_center[1]-H_point[1,2])/H[1,2]
    j3_center = H_point[:,2]+k*H[:,2]

    # p5
    k=(j3_center[1]-H_point[1,4])/H[1,4]
    j5_center = H_point[:,4]+k*H[:,4]
    # p4
    k=(j5_center[0]-H_point[0,3])/H[0,3]
    j4_center = H_point[:,3]+k*H[:,3]
    # p6
    k=(j5_center[0]-H_point[0,5])/H[0,5]
    j6_center = H_point[:,5]+k*H[:,5]

    P=np.zeros((3,7))
    P[:,0]=np.array([0,0,0])
    P[:,1]=j2_center-j1_center
    P[:,2]=j3_center-j2_center
    P[:,3]=j4_center-j3_center
    P[:,4]=j5_center-j4_center
    P[:,5]=j6_center-j5_center
    P[:,6]=tcp_base-j6_center

    print("P2:",P[:,1])
    print("P3:",P[:,2])
    print("P4:",P[:,3])
    print("P5:",P[:,4])
    print("P6:",P[:,5])
    print("P7:",P[:,6])
    print("H:",H.T)
    # print("J1 Center:",j1_center)
    # print("J6 Center:",j6_center)
    # print("J6 in J1:",j6_center-j1_center)
    print("====================")

# Find R^toolmarker_base
# similar to the idea of "flange frame"
rpy_tool_basemarker = []
for Rtool in R_tool_basemarker:
    rpy_tool_basemarker.append(np.array(R2rpy(Rtool)))
rpy_tool_basemarker = np.mean(rpy_tool_basemarker,axis=0)
R_tool_basemarker = rpy2R(rpy_tool_basemarker)
R_tool_base = np.matmul(T_basemarker_base.R,R_tool_basemarker)

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
base_marker_data['calib_base_basemarker_pose'] = {}
base_marker_data['calib_base_basemarker_pose']['position'] = {}
base_marker_data['calib_base_basemarker_pose']['position']['x'] = float(T_base_basemarker.p[0])
base_marker_data['calib_base_basemarker_pose']['position']['y'] = float(T_base_basemarker.p[1])
base_marker_data['calib_base_basemarker_pose']['position']['z'] = float(T_base_basemarker.p[2])
quat = R2q(T_base_basemarker.R)
base_marker_data['calib_base_basemarker_pose']['orientation'] = {}
base_marker_data['calib_base_basemarker_pose']['orientation']['w'] = float(quat[0])
base_marker_data['calib_base_basemarker_pose']['orientation']['x'] = float(quat[1])
base_marker_data['calib_base_basemarker_pose']['orientation']['y'] = float(quat[2])
base_marker_data['calib_base_basemarker_pose']['orientation']['z'] = float(quat[3])

base_marker_data['calib_tool_flange_pose'] = {}
base_marker_data['calib_tool_flange_pose']['position'] = {}
base_marker_data['calib_tool_flange_pose']['position']['x'] = 0
base_marker_data['calib_tool_flange_pose']['position']['y'] = 0
base_marker_data['calib_tool_flange_pose']['position']['z'] = 0
quat = R2q(R_tool_base)
base_marker_data['calib_tool_flange_pose']['orientation'] = {}
base_marker_data['calib_tool_flange_pose']['orientation']['w'] = float(quat[0])
base_marker_data['calib_tool_flange_pose']['orientation']['x'] = float(quat[1])
base_marker_data['calib_tool_flange_pose']['orientation']['y'] = float(quat[2])
base_marker_data['calib_tool_flange_pose']['orientation']['z'] = float(quat[3])

with open(base_marker_config_file,'w') as file:
    yaml.safe_dump(base_marker_data,file)

# calibrate tool
# Find T^tool_toolmarker
curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot_weld.tool_rigid_id,robot_weld.tool_markers_id)

# A: known tool maker position in tool frame, p^marker_tool
# B: capture tool marker position in rigid body frame (from motiv), p^marker_toolmarker
marker_B = {}
for marker_id in robot_weld.tool_markers_id: # average across the captured
    marker_B[marker_id] = np.mean(curve_p[marker_id],axis=0)
# find center of A B
marker_A = deepcopy(robot_weld.tool_markers)
center_A = []
center_B = []
A = []
B = []
for marker_id in robot_weld.tool_markers_id:
    center_A.append(marker_A[marker_id])
    A.append(marker_A[marker_id])
    center_B.append(marker_B[marker_id])
    B.append(marker_B[marker_id])
center_A = np.mean(center_A,axis=0)
A = np.array(A)
center_B = np.mean(center_B,axis=0)
B = np.array(B)

A_centered = A-center_A
B_centered = B-center_B
H = np.matmul(A_centered.T,B_centered)
u,s,vT = np.linalg.svd(H)
R = np.matmul(vT.T,u.T)
if np.linalg.det(R)<0:
    u,s,v = np.linalg.svd(R)
    v=v.T
    v[:,2] = v[:,2]*-1
    R = np.matmul(v,u.T)

t = center_B-np.dot(R,center_A)
T_tool_toolmarker = Transform(R,t)

with open(tool_marker_config_file,'r') as file:
    tool_marker_data = yaml.safe_load(file)
tool_marker_data['calib_tool_toolmarker_pose'] = {}
tool_marker_data['calib_tool_toolmarker_pose']['position'] = {}
tool_marker_data['calib_tool_toolmarker_pose']['position']['x'] = float(T_tool_toolmarker.p[0])
tool_marker_data['calib_tool_toolmarker_pose']['position']['y'] = float(T_tool_toolmarker.p[1])
tool_marker_data['calib_tool_toolmarker_pose']['position']['z'] = float(T_tool_toolmarker.p[2])
quat = R2q(T_tool_toolmarker.R)
tool_marker_data['calib_tool_toolmarker_pose']['orientation'] = {}
tool_marker_data['calib_tool_toolmarker_pose']['orientation']['w'] = float(quat[0])
tool_marker_data['calib_tool_toolmarker_pose']['orientation']['x'] = float(quat[1])
tool_marker_data['calib_tool_toolmarker_pose']['orientation']['y'] = float(quat[2])
tool_marker_data['calib_tool_toolmarker_pose']['orientation']['z'] = float(quat[3])

with open(tool_marker_config_file,'w') as file:
    yaml.safe_dump(tool_marker_data,file)

print("Done")