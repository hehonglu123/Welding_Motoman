from copy import deepcopy
import sys
sys.path.append('../toolbox/')
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

config_dir='../config/'

# robot_type='R1'
robot_type='R2'
# robot_type='S1'

# all_datasets=['train_data','valid_data_1','valid_data_2']
dataset_date='0524'
# all_datasets=['test'+dataset_date+'_R1_aftercalib/train_data']
all_datasets=['test'+dataset_date+'_'+robot_type+'/train_data']

if robot_type=='R1':
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml'
    tool_marker_config_file=config_dir+'weldgun_marker_config.yaml'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

    # only R matter
    nominal_robot_base = Transform(np.array([[0,1,0],
                                            [0,0,1],
                                            [1,0,0]]),[0,0,0]) 
    H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)

    jN=6

    # output_base_marker_config_file = config_dir+'MA2010_marker_config.yaml'
    output_base_marker_config_file = config_dir+'MA2010_'+dataset_date+'_marker_config.yaml'
    # output_base_marker_config_file = config_dir+'MA2010_0613_marker_config.yaml'
    # output_base_marker_config_file = config_dir+'MA2010_0504stretch_marker_config.yaml'
    # output_base_marker_config_file = config_dir+'MA2010_0504inward_marker_config.yaml'

    output_tool_marker_config_file = config_dir+'weldgun_'+dataset_date+'_marker_config.yaml'

elif robot_type=='R2':
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml'
    tool_marker_config_file=config_dir+'mti_marker_config.yaml'
    robot=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
    pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)

    # only R matter
    nominal_robot_base = Transform(np.array([[0,-1,0],
                                            [0,0,1],
                                            [-1,0,0]]),[0,0,0]) 
    H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)

    jN=6

    output_base_marker_config_file = config_dir+'MA1440_'+dataset_date+'_marker_config.yaml'
    output_tool_marker_config_file = config_dir+'mti_'+dataset_date+'_marker_config.yaml'

elif robot_type=='S1':
    base_marker_config_file=config_dir+'D500B_marker_config.yaml'
    tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml'
    robot=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=base_marker_config_file,tool_marker_config_file=tool_marker_config_file)
    # only R matter
    nominal_robot_base = Transform(np.array([[-1,0,0],
                                            [0,0,1],
                                            [0,1,0]]),[0,0,0]) 
    H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)

    jN=2

    output_base_marker_config_file = config_dir+'D500B_'+dataset_date+'_marker_config.yaml'
    output_tool_marker_config_file = tool_marker_config_file

H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)

P_marker_id = robot.tool_rigid_id
zero_config_q = [[],[],[],[],[],[]]
for dataset in all_datasets:
    print(dataset)
    raw_data_dir = 'PH_rotate_data/'+dataset
    for j in range(jN):
        # read raw data
        curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_'+str(j+1),robot.base_rigid_id,robot.tool_markers_id)

        # read q
        with open(raw_data_dir+'_'+str(j+1)+'_robot_q.pickle', 'rb') as handle:
            robot_q = pickle.load(handle)
            robot_q = robot_q[:,:jN]
        for i in range(jN):
            if i!=j:
                zero_config_q[i].extend(robot_q[:,i])

        # detect axis
        this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,j],robot.tool_markers_id)
        H_act[:,j] = this_axis_normal
        axis_p[:,j] = this_axis_p
        print("Axis",j+1,"done.")

    H = H_act
    H_point = axis_p
    for i in range(jN):
        H[:,i]=H[:,i]/np.linalg.norm(H[:,i])

    if robot_type!='S1':
        # rotate R
        z_axis = H[:,0]
        y_axis = H[:,1]
        y_axis = y_axis-np.dot(z_axis,y_axis)*z_axis
        y_axis = y_axis/np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis,z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        R = np.array([x_axis,y_axis,z_axis])

        # read raw data
        curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot.base_rigid_id,[P_marker_id])

        tcp = np.mean(curve_p[P_marker_id],axis=0)
        R_tool_basemarker = curve_R[P_marker_id] # need this later for tool calibration

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
        tcp_base = np.matmul(T_basemarker_base.R,tcp)+T_basemarker_base.p
        j1_center = np.matmul(T_basemarker_base.R,j1_center)+T_basemarker_base.p
        j2_center = np.matmul(T_basemarker_base.R,j2_center)+T_basemarker_base.p
        j3_center = np.matmul(T_basemarker_base.R,j3_center)+T_basemarker_base.p
        j4_center = np.matmul(T_basemarker_base.R,j4_center)+T_basemarker_base.p
        j5_center = np.matmul(T_basemarker_base.R,j5_center)+T_basemarker_base.p
        j6_center = np.matmul(T_basemarker_base.R,j6_center)+T_basemarker_base.p
        #######################################
    
        # k=(j2_center[1]-H_point[1,2])/H[1,2]
        # j3_center = H_point[:,2]+k*H[:,2]
        # # p5
        # k=(j3_center[1]-H_point[1,4])/H[1,4]
        # j5_center = H_point[:,4]+k*H[:,4]
        # # p4
        # k=(j5_center[0]-H_point[0,3])/H[0,3]
        # j4_center = H_point[:,3]+k*H[:,3]
        # # p6
        # k=(j5_center[0]-H_point[0,5])/H[0,5]
        # j6_center = H_point[:,5]+k*H[:,5]

        P=np.zeros((3,7))
        P[:,0]=np.array([0,0,0])
        P[:,1]=j2_center-j1_center
        P[:,2]=j3_center-j2_center
        P[:,3]=j4_center-j3_center
        P[:,4]=j5_center-j4_center
        P[:,5]=j6_center-j5_center
        # P[:,6]=tcp_base-j6_center
        P[:,6] = np.linalg.norm(robot.robot.P[:,5]+robot.robot.P[:,6])*(-1*H[:,5])

    else:
        # rotate R
        y_axis = -H[:,0]
        z_axis = -H[:,1]
        z_axis = z_axis-np.dot(z_axis,y_axis)*y_axis
        z_axis = z_axis/np.linalg.norm(z_axis)

        x_axis = np.cross(y_axis,z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        R = np.array([x_axis,y_axis,z_axis])

        # read raw data
        curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot.base_rigid_id,[P_marker_id])

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
        for i in range(jN):
            H_point[:,i] = np.matmul(T_basemarker_base.R,H_point[:,i])+T_basemarker_base.p
        tcp_base = np.matmul(T_basemarker_base.R,tcp)+T_basemarker_base.p
        j1_center = np.matmul(T_basemarker_base.R,j1_center)+T_basemarker_base.p
        j2_center = np.matmul(T_basemarker_base.R,j2_center)+T_basemarker_base.p
        #######################################

        P=np.zeros((3,3))
        P[:,0]=np.array([0,0,0])
        P[:,1]=j2_center-j1_center
        P[:,2]=tcp_base-j2_center
        
    print("P:",P.T)
    print("H:",H.T)
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

if robot_type!='S1':
    # save zero config q
    for i in range(jN):
        zero_config_q[i] = float(np.mean(zero_config_q[i]))
    base_marker_data['zero_config'] = zero_config_q

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

with open(output_base_marker_config_file,'w') as file:
    yaml.safe_dump(base_marker_data,file)

# calibrate tool
if robot_type!='S1':
    # Find T^tool_toolmarker
    curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot.tool_rigid_id,robot.tool_markers_id)

    # A: known tool maker position in tool frame, p^marker_tool
    # B: capture tool marker position in rigid body frame (from motiv), p^marker_toolmarker
    marker_B = {}
    for marker_id in robot.tool_markers_id: # average across the captured
        marker_B[marker_id] = np.mean(curve_p[marker_id],axis=0)
    
    # find center of A B
    marker_A = deepcopy(robot.tool_markers)
    center_A = []
    center_B = []
    A = []
    B = []
    for marker_id in robot.tool_markers_id:
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
        vT[2,:] *= -1
        R = vT.T @ u.T

    t = center_B-np.dot(R,center_A)
    T_tool_toolmarker = Transform(R,t)
    
else:
    # T^tool_base
    tool_p = j2_center+180*-1*H[:,1] # 180 mm above j2_center
    print(j2_center)
    T_tool_base = Transform(np.eye(3),tool_p)

    #T^tool_toolmarker
    T_toolmarker_base = Transform(R_tool_base,tcp_base)
    T_tool_toolmarker = T_toolmarker_base.inv()*T_tool_base
    # beacuse our positoiner always at 180 the second axis
    T_tool_toolmarker = T_tool_toolmarker*Transform(rot([0,0,1],np.pi),[0,0,0])

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

with open(output_tool_marker_config_file,'w') as file:
    yaml.safe_dump(tool_marker_data,file)
print("Done")