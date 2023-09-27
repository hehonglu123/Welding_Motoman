import open3d as o3d
from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
sys.path.append('../../mocap/')
sys.path.append('../')
from robot_def import *
from traj_manipulation import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from PH_interp import *
from weldCorrectionStrategy import *
from matplotlib.animation import FuncAnimation
from functools import partial
from scipy.interpolate import interp1d
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

from sklearn.cluster import DBSCAN

R1_ph_dataset_date='0801'
R2_ph_dataset_date='0801'
S1_ph_dataset_date='0801'

def get_slope(p1,p2):
    
    return (p2[1]-p1[1])/(p2[0]-p1[0])

zero_config=np.zeros(6)
# 0. robots.
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

# print(robot_scan.fwd(zero_config))
# exit()

#### change base H to calibrated ones ####
robot_scan_base = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
robot_scan.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot_weld.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
# T_to_base = Transform(np.eye(3),[0,0,-380])
# positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# input(positioner.base_H)

#### load R1 kinematic model
PH_data_dir='../../mocap/PH_grad_data/test'+R1_ph_dataset_date+'_R1/train_data_'
with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
ph_param_r1=PH_Param(nom_P,nom_H)
ph_param_r1.fit(PH_q,method='FBF')
ph_param_r1=None
#### load R2 kinematic model
PH_data_dir='../../mocap/PH_grad_data/test'+R2_ph_dataset_date+'_R2/train_data_'
with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
ph_param_r2=PH_Param(nom_P,nom_H)
ph_param_r2.fit(PH_q,method='FBF')
ph_param_r2=None
#### load S1 kinematic model
robot_weld.robot.P=deepcopy(robot_weld.calib_P)
robot_weld.robot.H=deepcopy(robot_weld.calib_H)
robot_scan.robot.P=deepcopy(robot_scan.calib_P)
robot_scan.robot.H=deepcopy(robot_scan.calib_H)
positioner.robot.P=deepcopy(positioner.calib_P)
positioner.robot.H=deepcopy(positioner.calib_H)

dataset='circle_large/'
sliced_alg='static_spiral/'
curve_data_dir = '../../data/'+dataset+sliced_alg
data_dir=curve_data_dir+'weld_scan_2023_09_20_21_11_14'+'/'
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

with open(data_dir+'mti_scans.pickle', 'rb') as file:
    mti_recording_all=pickle.load(file)
with open(data_dir+'robot_js.pickle', 'rb') as file:
    robot_js_all=pickle.load(file)
    
## streaming parameters
point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
segment_distance=0.8 ## segment d, such that ||xi_{j}-xi_{j-1}||=0.8
base_nominal_slice_increment=26
base_layer_N=2
nominal_slice_increment=18
scanner_lag_bp=int(slicing_meta['scanner_lag_breakpoints'])
scanner_lag=slicing_meta['scanner_lag']
end_layer_count = 7
offset_z = 2.2

# all_layers=[0,26,44,62,80,98,116,116]
all_layers=[0,26,52,70,88,106,124,124]

regen_pcd=False
regen_dh=False
show_animation=True
Transz0_H=None

#### Planned Print layers
all_layers=[0]
for i in range(end_layer_count):
    if i<base_layer_N:
        all_layers.append(all_layers[-1]+base_nominal_slice_increment)
    else:
        all_layers.append(all_layers[-1]+nominal_slice_increment)
print("Planned Print Layers:",all_layers)

####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]
curve_relative_all_slices=[]
curve_relative_dense_all_slices=[]
lam_relative_all_slices=[]
lam_relative_dense_all_slices=[]
for layer_n in all_layers:
    rob1_js_all_slices.append(np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(layer_n)+'_0.csv',delimiter=','))
    rob2_js_all_slices.append(np.loadtxt(curve_data_dir+'curve_sliced_js/MA1440_js'+str(layer_n)+'_0.csv',delimiter=','))
    positioner_js_all_slices.append(np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(layer_n)+'_0.csv',delimiter=','))
    curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer_n)+'_0.csv',delimiter=',')
    curve_relative_all_slices.append(curve_sliced_relative)
    lam_relative=calc_lam_cs(curve_sliced_relative)
    lam_relative_all_slices.append(lam_relative)
    lam_relative_dense_all_slices.append(np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance)))
    curve_sliced_dense_relative=interp1d(lam_relative,curve_sliced_relative,kind='cubic',axis=0)(lam_relative_dense_all_slices[-1])
    curve_relative_dense_all_slices.append(curve_sliced_dense_relative)
rob1_js_warp_all_slices=[]
rob2_js_warp_all_slices=[]
positioner_warp_js_all_slices=[]
for i in range(end_layer_count):
    rob1_js=deepcopy(rob1_js_all_slices[i])
    rob2_js=deepcopy(rob2_js_all_slices[i])
    positioner_js=deepcopy(positioner_js_all_slices[i])
    ###TRJAECTORY WARPING
    if i>0:
        rob1_js_prev=deepcopy(rob1_js_all_slices[i-1])
        rob2_js_prev=deepcopy(rob2_js_all_slices[i-1])
        positioner_js_prev=deepcopy(positioner_js_all_slices[i-1])
        rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
    if i<end_layer_count-1:
        rob1_js_next=deepcopy(rob1_js_all_slices[i+1])
        rob2_js_next=deepcopy(rob2_js_all_slices[i+1])
        positioner_js_next=deepcopy(positioner_js_all_slices[i+1])
        rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)

    rob1_js_warp_all_slices.append(rob1_js)
    rob2_js_warp_all_slices.append(rob2_js)
    positioner_warp_js_all_slices.append(positioner_js)

total_layer=len(robot_js_all)
all_pcd=o3d.geometry.PointCloud()
layer=0
x_state_location=[]
x_state_dh = []
for layer_count in range(0,total_layer):
    print("Layer Count",layer_count)
    ## the current curve, the next path curve and the next target curve
    if layer_count!=0:
        curve_sliced_relative=curve_relative_all_slices[layer_count-1]
    path_curve_dense_relative=curve_relative_dense_all_slices[layer_count]
    if layer_count<total_layer:
        target_curve_sliced_relative=curve_relative_all_slices[layer_count+1]
    
    num_segbp_layer=max(2,int(lam_relative_dense_all_slices[layer_count][-1]/segment_distance))
    segment_bp = np.linspace(0,len(lam_relative_dense_all_slices[layer_count])-1,num=num_segbp_layer).astype(int)
    segment_bp_sample = (np.diff(segment_bp)/2+segment_bp[:-1])
    segment_bp_sample=segment_bp_sample.astype(int) # sample the middle point
    segment_bp=segment_bp[1:] # from 1 ~ the last point (ignore 0)
    
    # Here is where the welding happens
    ###################################
    
    x_state_location = path_curve_dense_relative[segment_bp_sample][:,:3]
    x_state_dh = [[] for x in range(len(x_state_location))]
    
    robot_js=robot_js_all[layer_count]
    # wrap to -pi pi
    count=0
    ang=robot_js[0][-1]
    while ang<-np.pi:
        ang+=2*np.pi
        count+=1
    robot_js[:,-1]=robot_js[:,-1]+count*2*np.pi
    
    mti_recording=np.array(mti_recording_all[layer_count])
    # update to scan the same layer
    this_mti_recording = deepcopy(mti_recording[robot_js[:,-1]>-np.pi])
    this_robot_js = deepcopy(robot_js[robot_js[:,-1]>-np.pi])
    if layer_count>0:
        layer_robot_js = np.vstack((last_robot_js,this_robot_js))
        layer_mti_recording=[]
        layer_mti_recording.extend(deepcopy(last_mti_recording))
        layer_mti_recording.extend(deepcopy(this_mti_recording))
    last_robot_js = deepcopy(robot_js[robot_js[:,-1]<=-np.pi])
    last_mti_recording = deepcopy(mti_recording[robot_js[:,-1]<=-np.pi])
    if layer_count>0:
        robot_js=layer_robot_js
        mti_recording=layer_mti_recording
    
    # if layer_count!=0:
    #     mti_recording = mti_recording_all[layer_count].T[]
    
    robot_stamps=robot_js[:,0]
    q_out_exe=robot_js[:,7:]
    # robot_stamps=
    
    if layer_count>2:
        # dbscan = DBSCAN(eps=0.5,min_samples=20)
        dbscan = DBSCAN(eps=0.5,min_samples=20)
        fig = plt.figure()
        playback_speed=2
        
        def updatefig(i):
            print("frame:",i)
            fig.clear()
            
            ## remove not in interested region
            st=time.time()
            mti_pcd=np.delete(mti_recording[i*playback_speed],mti_recording[i*playback_speed][1]==0,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[1]<85,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[1]>100,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[0]<-10,axis=1)
            mti_pcd=np.delete(mti_pcd,mti_pcd[0]>10,axis=1)
            mti_pcd = mti_pcd.T
            
            # cluster based noise remove
            dbscan.fit(mti_pcd)
            cluster_id = dbscan.labels_>=0
            mti_pcd_noise_remove=mti_pcd[cluster_id]
            
            n_clusters_ = len(set(dbscan.labels_))
            # transform to R2TCP
            T_R2TCP_S1TCP=positioner.fwd(q_out_exe[i*playback_speed][6:],world=True).inv()*robot_scan.fwd(q_out_exe[i*playback_speed][:6],world=True)
            # T_S1TCP_R2TCP = T_R2TCP_S1TCP.inv()
            target_z = np.array([0,0,target_curve_sliced_relative[0][2]])
            largest_id = np.argsort(mti_pcd_noise_remove[:,1])[:10]
            point_location = np.mean(mti_pcd_noise_remove[largest_id],axis=0)
            point_location_z_R2TCP = deepcopy(point_location[1])
            point_location=np.insert(point_location,1,0)
            point_location = np.matmul(T_R2TCP_S1TCP.R,point_location)+T_R2TCP_S1TCP.p
            point_location[2]=point_location[2]-offset_z
            
            delta_h = (target_z[2]-point_location[2])
            
            x_state_i = np.argsort(np.linalg.norm(x_state_location-point_location,axis=1))[0]
            x_state_dh[x_state_i].append(delta_h)
            
            # target_z = np.matmul(T_S1TCP_R2TCP.R,target_z)+T_S1TCP_R2TCP.p
            print("X state i:",x_state_i)
            print("Positioner Location:",np.degrees(q_out_exe[i*playback_speed][-1]))
            print("Delta h:",delta_h)
            print("Total T:",time.time()-st)
            print("============")
            for cluster_i in range(n_clusters_-1):
                cluster_id = dbscan.labels_==cluster_i
                plt.scatter(-1*mti_pcd[cluster_id][:,0],mti_pcd[cluster_id][:,1])
            # plt.scatter(-1*mti_pcd[:,0],mti_pcd[:,1])
            # plt.axhline(y = target_z[2], color = 'r', linestyle = '-')
            plt.axhline(y = point_location_z_R2TCP-delta_h, color = 'r', linestyle = '-')
            plt.xlim((-30,30))
            # plt.ylim((50,120))
            plt.ylim((0,120))
            plt.draw()
        if show_animation:
            anim = FuncAnimation(fig, updatefig, np.floor(len(mti_recording)/playback_speed).astype(int),interval=16,repeat=False)
            # anim = FuncAnimation(fig, updatefig, np.floor(500).astype(int),interval=16,repeat=False)
            plt.show()
    
        ## calculate speed for each segments
        for x_state_i in range(len(x_state_location)):
            find_closest_i=0
            while len(x_state_dh[x_state_i+find_closest_i])==0:
                find_closest_i+=1
            if len(x_state_dh[x_state_i+find_closest_i])>0:
                print("Closest i:",find_closest_i)
                print("Mean dh:",np.mean(x_state_dh[x_state_i+find_closest_i]))
        exit()
    
    #### scanning process: processing point cloud and get h
    curve_sliced_relative=np.array(curve_sliced_relative)
    continue
    
    scan_process = ScanProcess(robot_scan,positioner)
    if regen_pcd:
        crop_extend=15
        crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
        crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
        pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
        # pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=False)
        # visualize_pcd([pcd])
        pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                            min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=300)
        o3d.io.write_point_cloud(data_dir+'layer_'+str(layer_count)+'_pcd.pcd',pcd)
    else:
        pcd=o3d.io.read_point_cloud(data_dir+'layer_'+str(layer_count)+'_pcd.pcd')
    # visualize_pcd([pcd])
    
    pcd,Transz0_H=scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)
    if layer_count!=0:
        
        if regen_dh:
            profile_dh,profile_height = scan_process.pcd2dh_compare(pcd,last_pcd,curve_sliced_relative[::4],drawing=False)
            np.save(data_dir+'layer_'+str(layer_count)+'_dh_profile.npy',profile_dh)
            np.save(data_dir+'layer_'+str(layer_count)+'_height_profile.npy',profile_height)
        else:
            profile_dh=np.load(data_dir+'layer_'+str(layer_count)+'_dh_profile.npy')
            profile_height=np.load(data_dir+'layer_'+str(layer_count)+'_height_profile.npy')
            
        
        # profile_dh_first_half = 
        
        # curve_i=0
        # total_curve_i = len(profile_dh)
        # ax = plt.figure().add_subplot()
        # for curve_i in range(total_curve_i):
        #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
        #     ax.scatter(profile_dh[curve_i,0],profile_dh[curve_i,1],c=color_dist)
        # ax.set_xlabel('Lambda')
        # ax.set_ylabel('dh to previous (mm)')
        # ax.set_title("dH Profile")
        # plt.show()
        
        # curve_i=0
        # total_curve_i = len(profile_height)
        # ax = plt.figure().add_subplot()
        # for curve_i in range(total_curve_i):
        #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
        #     ax.scatter(profile_height[curve_i,0],profile_height[curve_i,1],c=color_dist)
        # ax.set_xlabel('Lambda')
        # ax.set_ylabel('dh to Layer N (mm)')
        # ax.set_title("Height Profile")
        # plt.show()
        
    
    all_pcd += pcd
    # visualize_pcd([all_pcd])
    last_pcd=pcd
    
    ## update welding param
    
    
if regen_pcd:
    o3d.io.write_point_cloud(data_dir+'full_pcd.pcd',all_pcd)

crop_min=tuple([-1e5,-1e5,-0.1])
crop_max=tuple([1e5,1e5,1e5])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,max_bound=crop_max)
all_pcd=all_pcd.crop(bbox)
visualize_pcd([all_pcd])
