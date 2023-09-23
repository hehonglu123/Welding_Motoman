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

def read_and_convert_frame(filename,target_frame,markers_id):

    mocap_filename = filename+'_mocap'

    with open(mocap_filename+'_p.pickle', 'rb') as handle:
        curve_p = pickle.load(handle)
    with open(mocap_filename+'_R.pickle', 'rb') as handle:
        curve_R = pickle.load(handle)
    with open(mocap_filename+'_timestamps.pickle', 'rb') as handle:
        mocap_stamps = pickle.load(handle)
    with open(mocap_filename+'_cond.pickle', 'rb') as handle:
        mocap_cond = pickle.load(handle)

    # convert everything in basemarker frame
    if target_frame != '':
        curve_p,curve_R,mocap_stamps = to_frame(curve_p,curve_R,mocap_stamps,target_frame,markers_id)
    
    return curve_p,curve_R,mocap_stamps,mocap_cond

def cut_adge(points,center,normal,cut_edge_angle):
    
    points=np.array(points)
    z_axis=normal/np.linalg.norm(normal)
    x_axis=(points[0]-center)/np.linalg.norm((points[0]-center))
    x_axis=x_axis-np.dot(x_axis,z_axis)*z_axis
    x_axis=x_axis/np.linalg.norm(x_axis)
    y_axis=np.cross(z_axis,x_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)
    R = np.array([x_axis,y_axis,z_axis])
    
    p_transformed = (R@((points-center).T)).T
    
    p_atan2 = np.arctan2(p_transformed[:,1],p_transformed[:,0])
    
    p_lower_id = np.where(p_atan2<(np.max(p_atan2)-cut_edge_angle))
    points=points[p_lower_id]
    p_atan2=p_atan2[p_lower_id]
    
    p_upper_id = np.where(p_atan2>(np.min(p_atan2)+cut_edge_angle))
    points=points[p_upper_id]
    p_atan2=p_atan2[p_upper_id]
    
    return points

def detect_axis(points,rough_axis_direction,calib_marker_ids,cut_edge=False,cut_edge_angle=np.radians(3)):

    all_normals=[]
    all_centers=[]
    
    for i in range(len(calib_marker_ids)):
        if len(points[calib_marker_ids[i]])<3000:
            continue
        
        # if i==1:  
        
        
        center, normal = fitting_3dcircle(points[calib_marker_ids[i]])
        if cut_edge:
            points[calib_marker_ids[i]]=cut_adge(points[calib_marker_ids[i]],center,normal,cut_edge_angle)
            center, normal = fitting_3dcircle(points[calib_marker_ids[i]])

        # if calib_marker_ids[i]=='marker8_rigid4':
        #     print("Radius:",np.mean(np.linalg.norm(points[calib_marker_ids[i]]-center,axis=1)))
        #     print("Radius std:",np.std(np.linalg.norm(points[calib_marker_ids[i]]-center,axis=1)))

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

def point_filtering(curve_p,curve_R,mocap_stamps,mocap_cond,tool_markers_id,cond_thres_marker,cond_thres_rigid,skip_ids):
    for marker_id in tool_markers_id:
        curve_p[marker_id]=np.array(curve_p[marker_id])
        curve_R[marker_id]=np.array(curve_R[marker_id])
        mocap_stamps[marker_id]=np.array(mocap_stamps[marker_id])
        mocap_cond[marker_id]=np.array(mocap_cond[marker_id])
        
        # plt.hist(mocap_cond[marker_id],bins=10)
        # plt.show()
        if 'marker' in marker_id:
            # for markers, the smaller the better
            thres_ids = mocap_cond[marker_id]<=cond_thres_marker
        else:
            # for rigids, the larger the better
            thres_ids = mocap_cond[marker_id]>=cond_thres_rigid
        
        curve_p[marker_id]=curve_p[marker_id][thres_ids]
        curve_R[marker_id]=curve_R[marker_id][thres_ids]
        mocap_stamps[marker_id]=mocap_stamps[marker_id][thres_ids]
        mocap_cond[marker_id]=mocap_cond[marker_id][thres_ids]
        
        curve_p[marker_id]=curve_p[marker_id][::skip_ids]
        curve_R[marker_id]=curve_R[marker_id][::skip_ids]
        mocap_stamps[marker_id]=mocap_stamps[marker_id][::skip_ids]
        mocap_cond[marker_id]=mocap_cond[marker_id][::skip_ids]
    
    
    startid=0
    endid=-1
    xupper=1e9
    yupper=1e9
    zupper=1e9
    xlower=-1e9
    ylower=-1e9
    zlower=-1e9
    while True:
        ax = plt.figure().add_subplot(projection='3d')
        for marker_id in tool_markers_id:
            curve_p[marker_id]=curve_p[marker_id][startid:endid]
            xupper_id = curve_p[marker_id][:,2]>=-1*xupper
            xlower_id = curve_p[marker_id][:,2]<=-1*xlower
            yupper_id = curve_p[marker_id][:,0]>=-1*yupper
            ylower_id = curve_p[marker_id][:,0]<=-1*ylower
            zupper_id = curve_p[marker_id][:,1]<=zupper
            zlower_id = curve_p[marker_id][:,1]>=zlower
            
            bbox_id=xupper_id&xlower_id
            bbox_id&=yupper_id
            bbox_id&=ylower_id
            bbox_id&=zupper_id
            bbox_id&=zlower_id
            curve_p[marker_id]=curve_p[marker_id][bbox_id]
            
            print(len(curve_p[marker_id]))
            ax.scatter(-1*curve_p[marker_id][::100][:,2], -1*curve_p[marker_id][::100][:,0], curve_p[marker_id][::100][:,1])
        plt.ion()
        plt.show(block=False)
        upperstr = input("xyz upper:")
        if upperstr!='':
            upperstr = upperstr.split(',')
            xupper = float(upperstr[0])
            yupper = float(upperstr[1])
            zupper = float(upperstr[2])
        lowerstr = input("xyz lower:")
        if lowerstr!='':
            lowerstr = lowerstr.split(',')
            xlower = float(lowerstr[0])
            ylower = float(lowerstr[1])
            zlower = float(lowerstr[2])
        startidstr = input("Start index:")
        if startidstr!='':
            startid = int(startidstr)
        endidstr = input("End index:")
        if endidstr!='':
            endid = int(endidstr)
        q=input("Exit")
        if q=='q':
            break
    
    return curve_p,curve_R,mocap_stamps,mocap_cond

config_dir='../config/'

robot_type='R1'
# robot_type='R2'

# all_datasets=['train_data','valid_data_1','valid_data_2']
dataset_date='0913'
# all_datasets=['test'+dataset_date+'_R1_aftercalib/train_data']
all_datasets=['test'+dataset_date+'_'+robot_type+'/train_data']

cut_edge=False

if robot_type=='R1':
    config_dir='../../config/'
    robot_name='M10ia'
    tool_name='ge_R1_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=1

    # only R matter
    nominal_robot_base = Transform(np.array([[0,-1,0],
                                        [0,0,1],
                                        [-1,0,0]]),[0,0,0]) 

elif robot_type=='R2':
    config_dir='../../config/'
    robot_name='LRMATE200id'
    tool_name='ge_R2_tool'
    robot_marker_dir=config_dir+robot_name+'_marker_config/'
    tool_marker_dir=config_dir+tool_name+'_marker_config/'
    uframe_num=4
    utool_num=1
    robot_group=2
    
    # only R matter
    nominal_robot_base = Transform(np.array([[0,1,0],
                                        [0,0,1],
                                        [1,0,0]]),[0,0,0]) 

print("Dataset Date:",dataset_date)

robot=robot_obj(robot_name,def_path=config_dir+robot_name+'_robot_default_config.yml',tool_file_path=config_dir+tool_name+'.csv',\
base_marker_config_file=robot_marker_dir+robot_name+'_marker_config.yaml',tool_marker_config_file=tool_marker_dir+tool_name+'_marker_config.yaml')

cond_thres_marker=4
cond_thres_rigid=16
skip_ids=4

H_nom = np.matmul(nominal_robot_base.R,robot.robot.H)
jN=6

output_base_marker_config_file = robot_marker_dir+robot_name+'_'+dataset_date+'_marker_config.yaml'
output_tool_marker_config_file = tool_marker_dir+tool_name+'_'+dataset_date+'_marker_config.yaml'

H_act = deepcopy(H_nom)
axis_p = deepcopy(H_nom)

for dataset in all_datasets:
    print(dataset)
    raw_data_dir = 'PH_rotate_data/'+dataset
    for j in range(jN):
        tool_markers_id=deepcopy(robot.tool_markers_id) if True else deepcopy(robot.calib_markers_id)
        # print(tool_markers_id)
        # read raw data
        curve_p,curve_R,mocap_stamps,mocap_cond = read_and_convert_frame(raw_data_dir+'_'+str(j+1),'',tool_markers_id)
        # print(mocap_stamps[tool_markers_id[0]][0])
        # print(mocap_stamps[tool_markers_id[0]][-1])
        print("========")
        # continue
        
        # for mid in tool_markers_id:
        #     print(len(curve_p[mid]))
        curve_p,curve_R,mocap_stamps,mocap_cond = \
            point_filtering(curve_p,curve_R,mocap_stamps,mocap_cond,tool_markers_id,cond_thres_marker,cond_thres_rigid,skip_ids)
        # for mid in tool_markers_id:
        #     print(len(curve_p[mid]))
        # detect axis
        this_axis_p,this_axis_normal = detect_axis(curve_p,H_nom[:,j],tool_markers_id,cut_edge=cut_edge)
        H_act[:,j] = this_axis_normal
        axis_p[:,j] = this_axis_p
        print(H_act.T)
        print("Axis",j+1,"done.")
    # exit()

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
    
    with open(raw_data_dir+'_zero_robot_q.pickle', 'rb') as handle:
        calib_config_q = pickle.load(handle)
    calib_config_q=np.mean(calib_config_q,axis=0)
    if robot_type=='R1':
        calib_config_q=calib_config_q[:jN]
    else:
        calib_config_q=calib_config_q[jN:]
    calib_config_q[2]=calib_config_q[2]+calib_config_q[1]
    print("Calibration config:",calib_config_q)
    calib_config_q=np.radians(calib_config_q)
    
    # convert to zero position
    print("Before zeroing PH")
    print('P',np.round(P,3).T)
    print('H',np.round(H,3).T)
    R = np.eye(3)
    for j in range(jN,1,-1):
        R = rot(H[:,j-1],-calib_config_q[j-1])
        for i in range(j,7):
            if i!=6:
                H[:,i] = R@H[:,i]
            P[:,i] = R@P[:,i]
    print("After zeroing PH")
    print('P',np.round(P,3).T)
    print('H',np.round(H,3).T)
    
    print("Final PH")
    print("P:",P.T)
    print("H:",H.T)
    print("====================")

with open(robot_marker_dir+robot_name+'_marker_config.yaml','r') as file:
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

with open(output_base_marker_config_file,'w') as file:
    yaml.safe_dump(base_marker_data,file)

# calibrate tool (using zero config pose)
tool_markers_id=robot.tool_markers_id.extend(robot.tool_rigid_id)
curve_p,curve_R,mocap_stamps = read_and_convert_frame(raw_data_dir+'_zero',robot.base_rigid_id,tool_markers_id)
curve_p,curve_R,mocap_stamps,mocap_cond = \
            point_filtering(curve_p,curve_R,mocap_stamps,mocap_cond,tool_markers_id,cond_thres_marker,cond_thres_rigid,skip_ids)
# read q
with open(raw_data_dir+'_zero_robot_q.pickle', 'rb') as handle:
    calib_config_q = pickle.load(handle)
calib_config_q=np.mean(calib_config_q,axis=0)

T_tool_flange = Transform(robot.robot.R_tool,robot.robot.p_tool)

robot.robot.R_tool=np.eye(3)
robot.robot.p_tool=np.zeros(3)
robot.robot.P = deepcopy(P)
robot.robot.H = deepcopy(H)
T_flange_base = robot.fwd(calib_config_q)
T_base_flange = T_flange_base.inv()

toolid = robot.tool_rigid_id
all_toolmarker_flange_p=[]
all_toolmarker_flange_rpy=[]
all_tool_toolmarker_p=[]
all_tool_toolmarker_rpy=[]
for i in range(len(curve_p[toolid])):
    T_toolmarker_basemarker=Transform(curve_R[toolid][i],curve_p[toolid][i])
    T_toolmarker_base=T_basemarker_base*T_toolmarker_basemarker
    T_toolmarker_flange=T_base_flange*T_toolmarker_base
    T_flange_toolmaker=T_toolmarker_flange.inv()
    T_tool_toolmarker=T_flange_toolmaker*T_tool_flange
    all_tool_toolmarker_p.append(T_tool_toolmarker.p)
    all_tool_toolmarker_rpy.append(R2rpy(T_tool_toolmarker.R))
    all_toolmarker_flange_p.append(T_toolmarker_flange.p)
    all_toolmarker_flange_rpy.append(R2rpy(T_toolmarker_flange.R))

print("Check if rpy has singularity")
print("tool toolmarker rpy max min:",\
    np.degrees(np.max(all_tool_toolmarker_rpy,axis=0)),np.degrees(np.min(all_tool_toolmarker_rpy,axis=0)))
print("toolmarker flange rpy max min:",\
    np.degrees(np.max(all_toolmarker_flange_rpy,axis=0)),np.degrees(np.min(all_toolmarker_flange_rpy,axis=0)))
print("==============================")
tool_toolmarker_p = np.mean(all_tool_toolmarker_p,axis=0)
tool_toolmarker_rpy=np.mean(all_tool_toolmarker_rpy,axis=0)
T_tool_toolmarker=Transform(rpy2R(tool_toolmarker_rpy),tool_toolmarker_p)
toolmarker_flange_p = np.mean(all_toolmarker_flange_p,axis=0)
toolmarker_flange_rpy = np.mean(all_toolmarker_flange_rpy,axis=0)
T_toolmarker_flange=Transform(rpy2R(toolmarker_flange_rpy),toolmarker_flange_p)

with open(tool_marker_dir,'r') as file:
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

tool_marker_data['calib_toolmarker_flange_pose'] = {}
tool_marker_data['calib_toolmarker_flange_pose']['position'] = {}
tool_marker_data['calib_toolmarker_flange_pose']['position']['x'] = float(T_toolmarker_flange.p[0])
tool_marker_data['calib_toolmarker_flange_pose']['position']['y'] = float(T_toolmarker_flange.p[1])
tool_marker_data['calib_toolmarker_flange_pose']['position']['z'] = float(T_toolmarker_flange.p[2])
quat = R2q(T_toolmarker_flange.R)
tool_marker_data['calib_toolmarker_flange_pose']['orientation'] = {}
tool_marker_data['calib_toolmarker_flange_pose']['orientation']['w'] = float(quat[0])
tool_marker_data['calib_toolmarker_flange_pose']['orientation']['x'] = float(quat[1])
tool_marker_data['calib_toolmarker_flange_pose']['orientation']['y'] = float(quat[2])
tool_marker_data['calib_toolmarker_flange_pose']['orientation']['z'] = float(quat[3])

with open(output_tool_marker_config_file,'w') as file:
    yaml.safe_dump(tool_marker_data,file)
print("Done")