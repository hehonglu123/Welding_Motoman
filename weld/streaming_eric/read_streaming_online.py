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
from streaming_control import *
from sklearn.cluster import DBSCAN
from StreamingSend import *

R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'

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
positioner_origin_base = deepcopy(positioner.base_H)
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# input(positioner.base_H)

#### load R1 kinematic model
# PH_data_dir='../../mocap/PH_grad_data/test'+R1_ph_dataset_date+'_R1/train_data_'
# with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
#     PH_q=pickle.load(file)
# nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
#                    [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
# nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
#                 [-1,0,0],[0,-1,0],[-1,0,0]]).T
# ph_param_r1=PH_Param(nom_P,nom_H)
# ph_param_r1.fit(PH_q,method='FBF')
ph_param_r1=None
#### load R2 kinematic model
# PH_data_dir='../../mocap/PH_grad_data/test'+R2_ph_dataset_date+'_R2/train_data_'
# with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
#     PH_q=pickle.load(file)
# nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
#                    [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
# nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
#                 [-1,0,0],[0,-1,0],[-1,0,0]]).T
# ph_param_r2=PH_Param(nom_P,nom_H)
# ph_param_r2.fit(PH_q,method='FBF')
ph_param_r2=None
#### load S1 kinematic model
# robot_weld.robot.P=deepcopy(robot_weld.calib_P)
# robot_weld.robot.H=deepcopy(robot_weld.calib_H)
# robot_scan.robot.P=deepcopy(robot_scan.calib_P)
# robot_scan.robot.H=deepcopy(robot_scan.calib_H)
# positioner.robot.P=deepcopy(positioner.calib_P)
# positioner.robot.H=deepcopy(positioner.calib_H)

dataset='circle_large/'
sliced_alg='static_spiral/'
curve_data_dir = '../../data/'+dataset+sliced_alg
data_dir=curve_data_dir+'weld_scan_2023_09_27_19_22_19'+'/'
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

with open(data_dir+'mti_scans.pickle', 'rb') as file:
    mti_recording_all=pickle.load(file)
with open(data_dir+'robot_js.pickle', 'rb') as file:
    robot_js_all=pickle.load(file)
    
## streaming parameters
point_distance=0.01			###STREAMING POINT INTERPOLATED DISTANCE
streaming_rate=125
segment_distance=0.8 ## segment d, such that ||xi_{j}-xi_{j-1}||=0.8
nominal_feedrate=160
base_nominal_slice_increment=26
base_layer_N=2
delta_h_star = 1.8
base_nominal_vd_relative=8
nominal_vd_relative=10
nominal_slice_increment=int(delta_h_star/slicing_meta['line_resolution'])
scanner_lag_bp=int(slicing_meta['scanner_lag_breakpoints'])
scanner_lag=slicing_meta['scanner_lag']
end_layer_count = 10
offset_z=0
weld_feedback_gain_K=1
SS=StreamingSend(None,None,None,streaming_rate)
scan_process = ScanProcess(robot_scan,positioner)

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
x_state_height=[]
x_state_lam = []

lam_all=[]
dh_all=[]
height_all=[]
error_all=[]

dh_layer=[]
r1v_layer=[]
r1v_planned_layer=[]
height_layer=[]
lam_layer=[]
last_r1_p = None
last_stamp=None
for layer_count in range(end_layer_count):
    base_layer=True if layer_count<base_layer_N else False # baselayer flag
    vd_relative = base_nominal_vd_relative if base_layer else nominal_vd_relative # set feedrate
    
    layer=all_layers[layer_count] # get current layer
    print("Layer:",layer,"#:",layer_count)
    
    ## the current curve, the next path curve and the next target curve
    if layer_count>0:
        last_lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count-1])
    path_curve_dense_relative=deepcopy(curve_relative_dense_all_slices[layer_count])
    lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count])
    if layer_count<end_layer_count:
        target_curve_dense_relative=curve_relative_dense_all_slices[layer_count+1]
        target_lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count+1])
    
    ## get the segment for changing velocity
    num_segbp_layer=max(2,int(lam_relative_dense[-1]/segment_distance))
    segment_bp = np.linspace(0,len(lam_relative_dense)-1,num=num_segbp_layer).astype(int)
    segment_bp_sample = (np.diff(segment_bp)/2+segment_bp[:-1])
    segment_bp_sample=segment_bp_sample.astype(int) # sample the middle point
    if len(x_state_location)==0:
        x_state_location = deepcopy(path_curve_dense_relative[segment_bp_sample][:,:3])
        x_state_location = list(x_state_location)
        x_state_dh = [[] for x in range(len(x_state_location))]
        x_state_height = [[] for x in range(len(x_state_location))]
        x_state_lam = deepcopy(lam_relative_dense[segment_bp_sample])
        x_state_lam = list(x_state_lam)

    ## streaming
    read_recording_cnt=0
    bp_status=0
    for seg_i in range(len(segment_bp)-1):
        
        # Feedback control. TODO: get the correct delta segments
        if not base_layer:
            vd_relative = weld_controller_lambda(np.mean(x_state_dh[0]),weld_feedback_gain_K,nominal_feedrate)
        vd_relative=min(vd_relative,13)
        vd_relative=max(vd_relative,5)
        ####################3
        
        ### get breakpoints for vd
        bp_start = segment_bp[seg_i]
        bp_end = segment_bp[seg_i+1]
        breakpoints=SS.get_breakpoints(lam_relative_dense[bp_start:bp_end],vd_relative)
        breakpoints=breakpoints+bp_start
        print("delta h:",np.mean(x_state_dh[0]),", vd:",vd_relative)
        # exit()
        
        ###start logging
        bp_cnt=0
        r1_v_seg = []
        for bp in breakpoints:
            #################################### READ MTI and joints####################################
            robot_timestamp=robot_js_all[layer_count][read_recording_cnt][0]
            q14=robot_js_all[layer_count][read_recording_cnt][1:]
            mti_points = mti_recording_all[layer_count][read_recording_cnt]
            read_recording_cnt+=1
            
            ## get real R1 velocity
            calib_base = deepcopy(positioner.base_H)
            positioner.base_H = deepcopy(positioner_origin_base)
            if last_r1_p is None:
                last_r1_p = positioner.fwd(q14[-2:],world=True).inv()*robot_weld.fwd(q14[:6])
                last_r1_p = last_r1_p.p
                last_stamp = robot_timestamp
            this_r1_p=positioner.fwd(q14[-2:],world=True).inv()*robot_weld.fwd(q14[:6])
            this_r1_p=this_r1_p.p
            r1_v = np.linalg.norm((this_r1_p-last_r1_p))/(robot_timestamp-last_stamp)
            # print(this_r1_p)
            # print(last_r1_p)
            # print("===")
            r1_v_seg.append(r1_v)
            
            last_r1_p=deepcopy(this_r1_p)
            last_stamp=robot_timestamp
            positioner.base_H = deepcopy(calib_base)
            #############
            
            ## Get the delta h. TODO: get the correct target p
            # if bp<len(lam_relative_dense)/2:
            if bp<np.argmin(lam_relative_dense<scanner_lag):
                target_p = deepcopy(path_curve_dense_relative[0][:3])+np.array([0,0,delta_h_star])
            else:
                target_p = deepcopy(target_curve_dense_relative[0][:3])+np.array([0,0,delta_h_star])
            delta_h,point_p = scan_process.scan2dh(deepcopy(mti_points),\
                q14[6:],target_p,crop_min=[-10,85],crop_max=[10,100],offset_z=offset_z)
            x_state_location_arr = np.array(x_state_location)
            x_state_i = np.argsort(np.linalg.norm(x_state_location_arr[:,:2]-point_p[:2],axis=1))[0] # only look at xy
            x_state_dh[x_state_i].append(delta_h)
            x_state_height[x_state_i].append(deepcopy(point_p[2]))
            ################################
            bp_cnt+=1            
        
        ### update x_state 
        x_state_location.pop(0)
        x_state_location.append(target_curve_dense_relative[segment_bp_sample[seg_i]][:3])
        this_lam = x_state_lam.pop(0)
        x_state_lam.append(target_lam_relative_dense[segment_bp_sample[seg_i]])
        this_dh = x_state_dh.pop(0)
        x_state_dh.append([])
        this_height = x_state_height.pop(0)
        x_state_height.append([])
        lam_layer.append(this_lam)
        dh_layer.append(deepcopy(this_dh))
        height_layer.append(deepcopy(this_height))
        r1v_layer.append(np.mean(r1_v_seg))
        r1v_planned_layer.append(vd_relative)
        #########################
    
    if layer_count>0:
        dh_lambda=[]
        error_lambda=[]
        height_lambda=[]
        for h_arr in dh_layer:
            dh_lambda.append(np.mean(h_arr))
            error_lambda.append(np.mean(h_arr)-delta_h_star)
        for h_arr in height_layer:
            height_lambda.append(np.mean(h_arr))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.scatter(lam_layer, dh_lambda, label='delta h')
        # if layer_count>1:
        ax2.plot(lam_layer, r1v_planned_layer, 'tab:orange',label='Lambda dot Planned')
        ax2.plot(lam_layer, r1v_layer, 'g-',label='Lambda dot Execute')
        ax1.set_ylabel("delta h (mm)")
        ax2.set_ylabel("R1 Lambda dot (mm/sec)")
        ax2.axis(ymin=0,ymax=15)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        
        # plt.scatter(lam_layer,dh_lambda)
        plt.title("Delta h and Speed of Layer "+str(layer_count))
        plt.xlabel("Lambda (mm)")
        
        plt.show()
        lam_all.append(lam_layer)
        dh_all.append(dh_lambda)
        error_all.append(error_lambda)
        height_all.append(height_lambda)
        
        last2_r1v_layer = deepcopy(last_r1v_layer)
        last2_r1v_planned_layer = deepcopy(last_r1v_planned_layer)
        
    dh_layer=[]
    lam_layer=[]
    height_layer=[]
    last_r1v_layer = deepcopy(r1v_layer)
    last_r1v_planned_layer = deepcopy(r1v_planned_layer)
    r1v_layer=[]
    r1v_planned_layer = []

## streaming final for scanner lagging
layer_count=end_layer_count
path_curve_dense_relative=curve_relative_dense_all_slices[layer_count]
lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count])
vd_relative=nominal_vd_relative
# point_stream_start_time=time.time()
lag_scan_bp=np.argmin(lam_relative_dense<scanner_lag)
lag_scan_bp = int(lag_scan_bp)
breakpoints=SS.get_breakpoints(lam_relative_dense[:lag_scan_bp],vd_relative)
read_recording_cnt=0
###start logging
for bp_idx in range(len(breakpoints)):
    #################################### READ MTI and joints####################################
    robot_timestamp=robot_js_all[layer_count][read_recording_cnt][0]
    q14=robot_js_all[layer_count][read_recording_cnt][1:]
    mti_points = mti_recording_all[layer_count][read_recording_cnt]
    read_recording_cnt+=1
    
    ## Get the delta h. TODO: get the correct target p
    target_p = deepcopy(path_curve_dense_relative[0][:3])+np.array([0,0,delta_h_star])
    delta_h,point_p = scan_process.scan2dh(deepcopy(mti_points),\
        q14[6:],target_p,crop_min=[-10,85],crop_max=[10,100],offset_z=offset_z)
    x_state_location_arr = np.array(x_state_location)
    x_state_i = np.argsort(np.linalg.norm(x_state_location_arr[:,:2]-point_p[:2],axis=1))[0] # only look at xy
    x_state_dh[x_state_i].append(delta_h)
    x_state_height[x_state_i].append(point_p[2])

dh_layer=x_state_dh
lam_layer=x_state_lam
height_layer=x_state_height
dh_lambda=[]
error_lambda=[]
height_lambda=[]
for h_arr in dh_layer:
    dh_lambda.append(np.mean(h_arr))
    error_lambda.append(np.mean(h_arr)-delta_h_star)
for h_arr in height_layer:
    height_lambda.append(np.mean(h_arr))
plt.scatter(lam_layer,dh_lambda)
plt.title("Layer "+str(layer_count))
plt.xlabel("Lambda (mm)")
plt.ylabel("delta h (mm)")
plt.show()

lam_all.append(lam_layer)
dh_all.append(dh_lambda)
height_all.append(height_lambda)
error_all.append(error_lambda)

for layer_cnt in range(len(lam_all)):
    plt.scatter(lam_all[layer_cnt],height_all[layer_cnt])
plt.xlabel("Lambda (mm)",fontsize=15)
plt.ylabel("Height (mm)",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Layers",fontsize=20)
plt.show()

height_std=[]
for h_arr in height_all:
    height_std.append(np.std(h_arr))
plt.plot(range(1,end_layer_count+1),height_std,'-o')
plt.xlabel("Layer #",fontsize=15)
plt.ylabel("Height Std (mm)",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Height Std",fontsize=20)
plt.show()

error_norm=[]
for error in error_all:
    error_norm.append(np.linalg.norm(error))
plt.plot(range(1,end_layer_count+1),error_norm,'-o')
plt.xlabel("Layer #",fontsize=15)
plt.ylabel("Error norm (mm)",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Error 2-Norm",fontsize=20)
plt.show()

exit()
if regen_pcd:
    o3d.io.write_point_cloud(data_dir+'full_pcd.pcd',all_pcd)

crop_min=tuple([-1e5,-1e5,-0.1])
crop_max=tuple([1e5,1e5,1e5])
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_min,max_bound=crop_max)
all_pcd=all_pcd.crop(bbox)
visualize_pcd([all_pcd])
