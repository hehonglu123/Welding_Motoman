from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *

from weld_dh2v import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import open3d as o3d

def moving_average(a,w=3):
    
    if w%2==0:
        w+=1

    ## add padding
    padd_n = int((w-1)/2)
    a = np.append(np.ones(padd_n)*a[0],a)
    a = np.append(a,np.ones(padd_n)*a[-1])
    
    ret = np.cumsum(a, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:] / w

def robot_weld_path_gen(all_layer_z,forward_flag,base_layer):
    R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
    x0 =  1684	# Origin x coordinate
    y0 = -1179 + 428	# Origin y coordinate
    z0 = -260   # 10 mm distance to base

    weld_p=[]
    if base_layer: # base layer
        weld_p.append([x0 - 33, y0 - 20, z0+10])
        weld_p.append([x0 - 33, y0 - 20, z0])
        weld_p.append([x0 - 33, y0 - 105 , z0])
        weld_p.append([x0 - 33, y0 - 105 , z0+10])
    else: # top layer
        weld_p.append([x0 - 33, y0 - 30, z0+10])
        weld_p.append([x0 - 33, y0 - 30, z0])
        weld_p.append([x0 - 33, y0 - 95 , z0])
        weld_p.append([x0 - 33, y0 - 95 , z0+10])

    if not forward_flag:
        weld_p = weld_p[::-1]

    all_path_T=[]
    for layer_z in all_layer_z:
        path_T=[]
        for p in weld_p:
            path_T.append(Transform(R,p+np.array([0,0,layer_z])))

        all_path_T.append(path_T)
    
    return all_path_T

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

Table_home_T = positioner.fwd(np.radians([-15,180]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

# data_dir = '../data/wall_weld_test/full_test_weld_scan_2023_06_06_12_43_57/'
# data_dir = '../data/wall_weld_test/baseline_weld_scan_2023_06_06_15_28_31/'
# data_dir='../data/wall_weld_test/top_layer_test_mti/scans/'

build_height_profile=False
plot_correction=True
show_layer = []

x_lower = -99999
x_upper = 999999

# datasets=['baseline','full_test']
datasets=['full_test']
datasets_h_mean={}
datasets_h_std={}
for dataset in datasets:

    if dataset=='baseline':
        data_dir = '../data/wall_weld_test/baseline_weld_scan_2023_06_06_15_28_31/'
    elif dataset=='full_test':
        data_dir = '../data/wall_weld_test/weld_scan_2023_06_13_15_08_08/'

    forward_flag=False
    all_profile_height=[]
    all_correction_seg=[[],[],[]]
    h_mean=[]
    h_std=[]
    for i in range(0,9999999):
        try:
            pcd = o3d.io.read_point_cloud(data_dir+'layer_'+str(i)+'/scans/processed_pcd.pcd')
            profile_height = np.load(data_dir+'layer_'+str(i)+'/scans/height_profile.npy')
            q_out_exe=np.loadtxt(data_dir +'layer_'+str(i)+ '/scans/scan_js_exe.csv',delimiter=',')
            robot_stamps=np.loadtxt(data_dir +'layer_'+str(i)+ '/scans/scan_robot_stamps.csv',delimiter=',')
            with open(data_dir +'layer_'+str(i)+ '/scans/mti_scans.pickle', 'rb') as file:
                mti_recording=pickle.load(file)
            
            # q_out_exe=np.loadtxt(data_dir +'scan_js_exe.csv',delimiter=',')
            # robot_stamps=np.loadtxt(data_dir +'robot_stamps.csv',delimiter=',')
            # with open(data_dir +'mti_scans.pickle', 'rb') as file:
            #     mti_recording=pickle.load(file)
        except:
            break
        print("Layer",i)
        print("Forward:",not forward_flag)

        if build_height_profile:
            curve_x_start=43
            curve_x_end=-41
            crop_extend=10
            z_height_start=20
            scan_process = ScanProcess(robot_scan,positioner)
            crop_min=(curve_x_end-crop_extend,-30,-10)
            crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
            crop_h_min=(curve_x_end-crop_extend,-20,-10)
            crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
            q_init_table=np.radians([-15,200])
            pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
            # visualize_pcd([pcd])
            pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
            # visualize_pcd([pcd])
            profile_height = scan_process.pcd2height(deepcopy(pcd),-1)

        ### ignore x smaller and larger
        profile_height=np.delete(profile_height,np.where(profile_height[:,0]>x_upper),axis=0)
        profile_height=np.delete(profile_height,np.where(profile_height[:,0]<x_lower),axis=0)

        all_profile_height.append(profile_height)

        print(np.std(profile_height[:,1]))

        # if (plot_correction and i>2):
        #     ## parameters
        #     noise_h_thres = 3
        #     peak_threshold=0.25
        #     flat_threshold=2.5
        #     correct_thres = 2
        #     patch_nb = 2 # 2*0.1
        #     start_ramp_ratio = 0.67
        #     end_ramp_ratio = 0.33
        #     #############
        #     h_largest = np.max(profile_height[:,1])
        #     profile_slope = np.gradient(profile_height[:,1])/np.gradient(profile_height[:,0])
        #     # find slope peak
        #     weld_terrain=[]
        #     last_peak_i=None
        #     lastlast_peak_i=None
        #     for sample_i in range(len(profile_slope)):
        #         if np.fabs(profile_slope[sample_i])<peak_threshold:
        #             weld_terrain.append(0)
        #         else:
        #             if profile_slope[sample_i]>=peak_threshold:
        #                 weld_terrain.append(1)
        #             elif profile_slope[sample_i]<=-peak_threshold:
        #                 weld_terrain.append(-1)
        #             if lastlast_peak_i:
        #                 if (weld_terrain[-1]==weld_terrain[lastlast_peak_i]) and (weld_terrain[-1]!=weld_terrain[last_peak_i]):
        #                     weld_terrain[last_peak_i]=0
        #             lastlast_peak_i=last_peak_i
        #             last_peak_i=sample_i

        #     weld_terrain=np.array(weld_terrain)
        #     weld_peak=[]
        #     weld_peak_id=[]
        #     last_peak=None
        #     last_peak_i=None
        #     for sample_i in range(len(profile_slope)):
        #         if weld_terrain[sample_i]!=0:
        #             if last_peak is None:
        #                 weld_peak.append(profile_height[sample_i])
        #                 weld_peak_id.append(sample_i)
        #             else:
        #                 # if the terrain change
        #                 if (last_peak>0 and weld_terrain[sample_i]<0) or (last_peak<0 and weld_terrain[sample_i]>0):
        #                     weld_peak.append(profile_height[last_peak_i])
        #                     weld_peak.append(profile_height[sample_i])
        #                     weld_peak_id.append(last_peak_i)
        #                     weld_peak_id.append(sample_i)
        #                 else:
        #                     # the terrain not change but flat too long
        #                     if profile_height[sample_i,0]-profile_height[last_peak_i,0]>flat_threshold:
        #                         weld_peak.append(profile_height[last_peak_i])
        #                         weld_peak.append(profile_height[sample_i])
        #                         weld_peak_id.append(last_peak_i)
        #                         weld_peak_id.append(sample_i)
        #             last_peak=deepcopy(weld_terrain[sample_i])
        #             last_peak_i=sample_i
        #     weld_peak.append(profile_height[-1])
        #     weld_peak_id.append(len(profile_height)-1)
        #     weld_peak=np.array(weld_peak)
        #     weld_peak_id=np.array(weld_peak_id)

        #     weld_peak_id[0]=0 # ensure start at 0

        #     correction_index = np.where(profile_height[:,1]-h_largest<-1*correct_thres)[0]

        #     # identified patch
        #     correction_patches = []
        #     patch=[]
        #     for cor_id_i in range(len(correction_index)):
        #         if len(patch)==0:
        #             patch = [correction_index[cor_id_i]]
        #         else:
        #             if correction_index[cor_id_i]-patch[-1]>patch_nb:
        #                 correction_patches.append(deepcopy(patch))
        #                 patch=[correction_index[cor_id_i]]
        #             else:
        #                 patch.append(correction_index[cor_id_i])
        #     correction_patches.append(deepcopy(patch))
        #     # find motion start/end using ramp before and after patch
        #     motion_patches=[]
        #     for patch in correction_patches:
        #         motion_patch=[]
        #         # find start
        #         start_i = patch[0]
        #         if np.all(weld_peak_id>=start_i):
        #             motion_patch.append(start_i)
        #         else:
        #             start_ramp_start_i = np.where(weld_peak_id<=start_i)[0][-1]
        #             start_ramp_end_i = np.where(weld_peak_id>start_i)[0][0]

        #             start_ramp_start_i = max(0,start_ramp_start_i)
        #             start_ramp_end_i = min(start_ramp_end_i,len(weld_peak_id)-1)
        #             if profile_slope[weld_peak_id[start_ramp_start_i]]>0:
        #                 start_ramp_start_i=start_ramp_start_i+1
        #                 start_ramp_end_i=start_ramp_end_i+1
        #             if profile_slope[weld_peak_id[start_ramp_end_i]]>0:
        #                 start_ramp_start_i=start_ramp_start_i-1
        #                 start_ramp_end_i=start_ramp_end_i-1
        #             start_ramp_start_i = max(0,start_ramp_start_i)
        #             start_ramp_end_i = min(start_ramp_end_i,len(weld_peak_id)-1)
        #             start_ramp_start=weld_peak_id[start_ramp_start_i]
        #             start_ramp_end=weld_peak_id[start_ramp_end_i]
                    
        #             if forward_flag:
        #                 motion_patch.append(int(np.round(start_ramp_start*end_ramp_ratio+start_ramp_end*(1-end_ramp_ratio))))
        #             else:
        #                 motion_patch.append(int(np.round(start_ramp_start*start_ramp_ratio+start_ramp_end*(1-start_ramp_ratio))))
        #         # find end
        #         end_i = patch[-1]
        #         if np.all(weld_peak_id<=end_i):
        #             motion_patch.append(end_i)
        #         else:
        #             end_ramp_start_i = np.where(weld_peak_id<=end_i)[0][-1]
        #             end_ramp_end_i = np.where(weld_peak_id>end_i)[0][0]
        #             if profile_slope[weld_peak_id[end_ramp_start_i]]<0:
        #                 end_ramp_start_i=end_ramp_start_i+1
        #                 end_ramp_end_i=end_ramp_end_i+1
        #             if profile_slope[weld_peak_id[end_ramp_end_i]]<0:
        #                 end_ramp_start_i=end_ramp_start_i-1
        #                 end_ramp_end_i=end_ramp_end_i-1
        #             end_ramp_start=weld_peak_id[end_ramp_start_i]
        #             end_ramp_end=weld_peak_id[end_ramp_end_i]
                    
        #             if forward_flag:
        #                 motion_patch.append(int(np.round(end_ramp_end*start_ramp_ratio+end_ramp_start*(1-start_ramp_ratio))))
        #             else:
        #                 motion_patch.append(int(np.round(end_ramp_end*end_ramp_ratio+end_ramp_start*(1-end_ramp_ratio))))
                
        #         if forward_flag:
        #             motion_patches.append(motion_patch[::-1])
        #         else:
        #             motion_patches.append(motion_patch)
        #     if forward_flag:
        #         motion_patches=motion_patches[::-1]
            
        #     draw_motion_patch = []
        #     for mo_pat in motion_patches:
        #         draw_motion_patch.extend(np.arange(np.min(mo_pat),np.max(mo_pat)+1))
        #     # print(draw_motion_patch)
        #     all_correction_seg.append(draw_motion_patch)

        # if i in show_layer:
        #     # plot pcd
        #     visualize_pcd([pcd])
        #     # plot height profile
        #     plt.scatter(profile_height[:,0],profile_height[:,1])
        #     plt.xlabel('x-axis')
        #     plt.ylabel('z-axis')
        #     plt.title("Height Profile of Layer "+str(i))
        #     plt.show()

        #     if plot_correction and i>2:
        #         plt.plot(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]),'o',label="Weld Height")
        #         plt.plot(profile_height[correction_index,0],profile_height[correction_index,1]-np.mean(profile_height[:,1]),'o',label="Weld Below Threshold")
                
        #         smooth_profile_height=deepcopy(profile_height)
        #         # smooth_profile_height[:,1] = moving_average(smooth_profile_height[:,1],w=5)
        #         smooth_profile_slope = np.gradient(smooth_profile_height[:,1])/np.gradient(smooth_profile_height[:,0])
                
        #         # plt.plot(profile_height[:,0],profile_slope)
        #         plt.plot(profile_height[:,0],smooth_profile_slope,label="Weld Slope")
        #         for mo_pat in motion_patches:
        #             plt.plot(profile_height[mo_pat,0],profile_height[mo_pat,1]-np.mean(profile_height[:,1]),'o',label="Corrected Motion")
        #         plt.legend()
        #         plt.show()

        forward_flag= not forward_flag

        h_mean.append(np.mean(profile_height[:,1]))
        h_std.append(np.std(profile_height[:,1]))

    i=0
    m_size=12
    for profile_height in all_profile_height:
        plt.scatter(profile_height[:,0],profile_height[:,1],s=3)
        if len(all_correction_seg)==len(all_profile_height) and dataset!='baseline':
            if i==5:
                plt.scatter(profile_height[all_correction_seg[i],0],profile_height[all_correction_seg[i],1],c='red',s=m_size,label='Correction', alpha=0.5)
            else:
                plt.scatter(profile_height[all_correction_seg[i],0],profile_height[all_correction_seg[i],1],c='red',s=m_size,alpha=0.5)
        i+=1
    plt.xlabel('x-axis')
    plt.ylabel('z-axis')
    plt.legend()
    plt.title("Height Profile")
    plt.show()

    datasets_h_mean[dataset]=np.array(h_mean)
    datasets_h_std[dataset]=np.array(h_std)

for dataset in datasets:
    plt.plot(np.arange(len(datasets_h_mean[dataset])),datasets_h_mean[dataset],'-o',label=dataset)
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Mean Height (mm)')
plt.title("Mean Height")
plt.show()

for dataset in datasets:
    plt.plot(np.arange(len(datasets_h_std[dataset])),datasets_h_std[dataset],'-o',label=dataset)
plt.axhline(y = 0.48, color = 'r', linestyle = '-')
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Height STD (mm)')
plt.title("Height STD")
plt.show()