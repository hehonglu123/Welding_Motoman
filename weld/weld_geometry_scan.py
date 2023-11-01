from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
sys.path.append('../mocap/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from PH_interp import *
from weldCorrectionStrategy import *
from WeldSend import *
from weldRRSensor import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import glob
import yaml
from math import ceil,floor

def connect_failed(s, client_id, url, err):
    global mti_sub, mti_client
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
    mti_sub=RRN.SubscribeService(url)
    mti_client=mti_sub.GetDefaultClientWait(1)

def generate_mti_rr():
    
    global mti_sub,mti_client
    
    mti_sub=RRN.SubscribeService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_sub.ClientConnectFailed += connect_failed
    mti_client=mti_sub.GetDefaultClientWait(1)
    mti_client.setExposureTime("25")


R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'

zero_config=np.zeros(6)
# 0. robots.
config_dir='../config/'
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

#### change base H to calibrated ones ####
robot_scan_base = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
robot_scan.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot_weld.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# exit()

r1_nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
r1_nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
r2_nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
r2_nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                [-1,0,0],[0,-1,0],[-1,0,0]]).T
#### load R1 kinematic model
# PH_data_dir='../mocap/PH_grad_data/test'+R1_ph_dataset_date+'_R1/train_data_'
# with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
#     PH_q=pickle.load(file)
# ph_param_r1=PH_Param(r1_nom_P,r1_nom_H)
# ph_param_r1.fit(PH_q,method='FBF')
ph_param_r1=None
#### load R2 kinematic model
# PH_data_dir='../mocap/PH_grad_data/test'+R2_ph_dataset_date+'_R2/train_data_'
# with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
#     PH_q=pickle.load(file)
# ph_param_r2=PH_Param(r2_nom_P,r2_nom_H)
# ph_param_r2.fit(PH_q,method='FBF')
ph_param_r2=None
# robot_scan.robot.P=deepcopy(robot_scan.calib_P)
# robot_scan.robot.H=deepcopy(robot_scan.calib_H)
# robot_weld.robot.P=deepcopy(robot_weld.calib_P)
# robot_weld.robot.H=deepcopy(robot_weld.calib_H)
# ### load S1 kinematic model
# positioner.robot.P=deepcopy(positioner.calib_P)
# positioner.robot.H=deepcopy(positioner.calib_H)

#### data directory
# dataset='cup/'
# sliced_alg='circular_slice_shifted/'
dataset='blade0.1/'
sliced_alg='auto_slice/'
# dataset='circle_large/'
# sliced_alg='static_stepwise_zero/'
curve_data_dir = '../data/'+dataset+sliced_alg

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]

data_date = input("Use old data directory? (Enter or put time e.g. 2023_07_11_16_25_30): ")
if data_date == '':
    data_dir=curve_data_dir+'weld_scan_'+formatted_time+'/'
    # data_dir=curve_data_dir+'weld_scan_2023_10_23_15_55_05/'
else:
    data_dir=curve_data_dir+'weld_scan_'+data_date+'/'
print("Use data directory:",data_dir)

#### welding spec, goal
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)
line_resolution = slicing_meta['line_resolution']
total_layer = slicing_meta['num_layers']
total_baselayer = slicing_meta['num_baselayers']

## weldind parameters
weld_mode=100
des_job=int(200+weld_mode/10)

# des_dh = 2.3420716473455623
# des_v = round(dh2v_loglog(des_dh,weld_mode),1)
# print("The Desired speed (according to desired h",des_dh,"will be",\
#       des_v,"mm/sec")

des_v = 5
des_dh = v2dh_loglog(des_v,weld_mode)
print("The Desired height (according to desired v",des_v,"mm/sec will be",\
      des_dh,"mm")

des_dw = 4
waypoint_distance=1.625 	###waypoint separation (calculate from 40moveL/95mm, where we did the test)
# waypoint_distance=1
layer_height_num=int(des_dh/line_resolution) # preplanned
layer_width_num=int(des_dw/line_resolution) # preplanned

# weld_min_v=2.5
# weld_max_v=10
# weld_min_v=des_v/2.
weld_min_v=des_v/1.5
# weld_min_v=3.6
weld_max_v=min(des_v*2,20)
print(weld_min_v,weld_max_v)

# 2. Scanning parameters
### scan parameters
scan_speed=5 # scanning speed (mm/sec)
scan_stand_off_d = 95 ## mm
Rz_angle = np.radians(0) # point direction w.r.t welds
Ry_angle = np.radians(0) # rotate in y a bit
bounds_theta = np.radians(1) ## circular motion at start and end
all_scan_angle = np.radians([0]) ## scan angle
q_init_table=np.radians([-15,200]) ## init table
mti_Rpath = np.array([[ -1.,0.,0.],   
                        [ 0.,1.,0.],
                        [0.,0.,-1.]])

# 3. Motion Parameters
to_start_speed=7
to_home_speed=10
# R1_home = np.radians([10,0,0,0,0,0])
# R2_mid = np.radians([6,20,-10,0,0,0])
# R2_home = np.radians([70,10,-5,0,0,0])

R1_mid = np.radians([-25,0,0,0,0,0])
R1_home = np.radians([-60,0,0,0,0,0])
R2_mid = np.radians([-6,20,-10,0,0,0])
R2_home = np.radians([-30,20,-10,0,0,0])

scan_process = ScanProcess(robot_scan,positioner)
# ## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.10:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser)
# MTI connect to RR
generate_mti_rr()
###################################
start_feedback=3 # with correction
### preplanned v,height for first few layer
planned_layer=999
## 300 260 250 240 ... 100
# planned_v=np.ones(planned_layer)*8
planned_v=np.array([8,8,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v,des_v])
planned_v=np.append(planned_v,np.ones(planned_layer)*des_v)
planned_v=planned_v.astype(int)

# base ipm: 250
planned_job=np.array([225,225,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job,des_job])
planned_job=np.append(planned_job,np.ones(planned_layer)*des_job)
planned_job=planned_job.astype(int)

print_min_dh = 0.5 # mm

arc_on=True

tri_robot=True
save_weld_record=True
save_output_points=True

last_layer_curve_relative = []
last_layer_curve_height = []

layer=-1
last_layer=-1
layer_count=-1
start_weld_layer=0

# layer=1
# last_layer=0
# layer_count=2
# start_weld_layer=0

# Transz0_H=None
Transz0_H=np.array([[ 9.99977849e-01, -4.63425601e-05, -6.65580373e-03,  5.00206395e-03],
 [-4.63425601e-05,  9.99903047e-01, -1.39246294e-02,  1.04648348e-02],
 [ 6.65580373e-03,  1.39246294e-02,  9.99880895e-01, -7.51444661e-01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# try:
#     layer_count=len(glob.glob(data_dir+'layer_*_0'))+1
# except:
#     pass

manual_dh=False
correction=True
recal_dh=False

start_shift=False # if true then always add odd layers
draw_dh=False

print("Planned V (first 10):",planned_v[:10])
print("Planned Job (first 10):",planned_job[:10])
print("Start Layer:",layer)
print("Last Layer:",last_layer)
print("Layer Count:",layer_count)

mean_layer_dh=None

while True:
    print("Layer Count:",layer_count)
    if layer_count<6:
        weld_min_v=des_v/1.25
    else:
        weld_min_v=des_v/1.5

    ####### forward/backward, baselayer/regular layers #####
    if total_baselayer >0:
        baselayer=True if layer_count<total_baselayer else False
    else:
        baselayer=False
    forward=True if layer_count%2==0 else False

    ####### Load previous data if start from middle #######
    if layer_count>=0 and len(last_layer_curve_relative)==0:
        read_layer=0 if layer<0 else layer
        
        if baselayer:
            last_num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/baselayer'+str(read_layer)+'_*.csv'))
        else:
            last_num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(read_layer)+'_*.csv'))
        
        last_pcd_layer=o3d.geometry.PointCloud()
        layer_curve_dh=[]
        for x in range(last_num_sections):
            if baselayer:
                last_scan_dir=data_dir+'baselayer_'+str(read_layer)+'_'+str(x)+'/scans/'
            else:
                if layer_count==total_baselayer:
                    last_scan_dir=data_dir+'baselayer_'+str(total_baselayer-1)+'_'+str(x)+'/scans/'
                else:
                    last_scan_dir=data_dir+'layer_'+str(read_layer)+'_'+str(x)+'/scans/'
                
            # load previous curve
            last_layer_curve_relative.extend(np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(read_layer)+'_'+str(x)+'.csv',delimiter=','))
            # load previous pcd
            if layer!=-1:
                last_pcd_layer = last_pcd_layer+o3d.io.read_point_cloud(last_scan_dir+'processed_pcd.pcd')
                profile_dh = np.load(last_scan_dir+'height_profile.npy')
                layer_curve_dh.extend(profile_dh)
                
        last_layer_curve_relative=np.array(last_layer_curve_relative)
        layer_curve_dh=np.array(layer_curve_dh)
            
        # load previous height
        # if layer!=-1:
        #     layer_curve_dh = np.load(data_dir+'layer_'+str(read_layer)+'_0/scans/'+'height_profile.npy')
        # visualize_pcd([last_pcd_layer])

    ####### Decide which layer to print #######
    if baselayer:
        last_layer = layer
        layer = layer_count
    if not manual_dh and not baselayer and layer_count>=0:
        # if layer_count!=0 and layer_count<start_feedback:
        #     last_layer=layer
        #     layer+=layer_height_num
        if layer_count<total_baselayer:
            pass
        if layer_count==total_baselayer:
            layer=0
        else:
            # plt.scatter(all_profile_height[:,0],all_profile_height[:,1])
            # plt.show()
            # print(all_profile_height[:,1])
            # mean_layer_height=np.mean(last_layer_curve_height)
            # last_layer = layer # update last layer
            
            mean_layer_dh=np.mean(layer_curve_dh[:,1])
            if not start_shift:
                dlayer = int(round(mean_layer_dh/line_resolution)) # find the "delta layer" using dh
            else:
                print("enforce odd dlayer")
                dlayer = int(np.ceil(mean_layer_dh/line_resolution/2)*2-1) # enforce a odd dlayer for shift
            # dlayer=23
            dlayer = max(15,dlayer)
            dlayer = min(35,dlayer)
            if layer_count<6:
                dlayer = max(15,dlayer)
                dlayer = min(30,dlayer)
            last_layer=layer
            layer = layer+dlayer # update layer
            print("Last Mean dh:",mean_layer_dh)
            print("Last Mean dlayer:",dlayer)
            
            # layer = int(round(mean_layer_height/line_resolution))
            # print("Last Mean height:",mean_layer_height)
            
            print("Last Layer:",last_layer)
            print("This Layer:",layer)

    # if achieve total layer
    if layer>=total_layer:
        print("More than total layer")
        break
    # if close to total layer
    if layer+layer_height_num>=total_layer:
        dlayer = total_layer-layer
        if dlayer*line_resolution<print_min_dh: # too close to the total layer
            break

    print("Print Layer:",layer,"Foward",forward,"Baselayer",baselayer)
    ####################DETERMINE CURVE ORDER##############################################
    if layer<0:
        read_layer=0
    else:
        read_layer=layer
    if not baselayer:
        num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(read_layer)+'_*.csv'))
    else:
        num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/baselayer'+str(read_layer)+'_*.csv'))
        
    #### welding
    start_section=0
    weld_st=time.time()
    if layer>=start_weld_layer:
        for x in range(start_section,num_sections):
            print("Print Layer",layer,"Sec.",x)

            # 1. The curve path in "positioner tcp frame"
            # Load nominal path given the layer
            if not baselayer:
                curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                if len(curve_sliced_js)<2:
                    continue
                positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
                ir_js = np.loadtxt(curve_data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            else: # baselayer
                curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_base_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                if len(curve_sliced_js)<2:
                    continue
                positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_base_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
                ir_js = np.loadtxt(curve_data_dir+'curve_sliced_js/MA1440_base_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

            #### convert to R1 and S1 motion
            lam1=calc_lam_js(curve_sliced_js,robot_weld)
            lam2=calc_lam_js(positioner_js,positioner)
            lam_relative=calc_lam_cs(curve_sliced_relative)
            lam_relative=np.array(lam_relative)

            num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

            ## using forward/backward technique
            if forward:
                breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
            else:
                breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

            #### Correction ####
            if not correction or (layer_count<start_feedback): # no correction
                this_weld_v=np.ones(len(breakpoints)-1)*planned_v[layer_count]
                weld_job=planned_job[layer_count]
            else: # start correction after "start_feedback"
                
                if recal_dh:
                    profile_dh = scan_process.pcd2dh(last_pcd_layer,curve_sliced_relative,drawing=True)
                    layer_curve_dh = profile_dh
                    last_layer_curve_relative=deepcopy(curve_sliced_relative)

                # layer_curve_dh=np.roll(layer_curve_dh,4)

                #### correction strategy
                this_weld_v,all_dh=\
                    strategy_4(layer_curve_dh,des_dh,curve_sliced_relative,last_layer_curve_relative,breakpoints,max_v=weld_max_v,min_v=weld_min_v,ipm_mode=weld_mode)
                weld_job=des_job

                ####################
                print("dh:",all_dh)
                print("Nominal V:",des_v)
                print("Corrected V:",this_weld_v)
                print(len(curve_sliced_relative))

                # fig, ax1 = plt.subplots()
                # ax2 = ax1.twinx()
                # ax1.scatter(lam_relative[breakpoints],all_dh,c='blue',label='Height Layer '+str(layer))
                # for bp_i in range(len(breakpoints)-1):
                #     ax2.plot([lam_relative[breakpoints[bp_i]],lam_relative[breakpoints[bp_i+1]]],[this_weld_v[bp_i],this_weld_v[bp_i]],'-o')
                # ax1.set_xlabel('X-axis (Lambda) (mm)')
                # ax1.set_ylabel('dH (mm)', color='g')
                # ax2.set_ylabel('Speed (mm/sec)', color='b')
                # ax1.legend(loc=0)
                # ax2.legend(loc=0)
                # plt.title("dH and Speed, 40 MoveL")
                # plt.legend()
                # plt.ion()
                # plt.show(block=False)

                # plt.scatter(all_profile_height[:,0],all_profile_height[:,1],c='blue')
                # for bp_i in range(len(breakpoints)-1):
                #     plt.plot([lam_relative[breakpoints[bp_i]],lam_relative[breakpoints[bp_i+1]]],[this_weld_v[bp_i],this_weld_v[bp_i]],'-o')
                # plt.show()

            ## use vel=1 and times the desired speed
            s1_all,s2_all=calc_individual_speed(1,lam1,lam2,lam_relative,breakpoints)
            s1_all=np.multiply(s1_all,this_weld_v)
            s2_all=np.multiply(s2_all,this_weld_v)

            if layer>=0 and True:
                go_weld=True
            else:
                go_weld=False

            # input("Weld Move to Start")
            ###move to intermidieate waypoint for collision avoidance if multiple section
            # if num_sections!=num_sections_prev:
            waypoint_pose=robot_weld.fwd(curve_sliced_js[breakpoints[0]])
            waypoint_pose.p[-1]+=50
            try:
                q1=robot_weld.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
            except: # if use calib PH
                robot_weld.robot.P=deepcopy(r1_nom_P)
                robot_weld.robot.H=deepcopy(r1_nom_H)
                q1=robot_weld.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
                robot_weld.robot.P=deepcopy(robot_weld.calib_P)
                robot_weld.robot.H=deepcopy(robot_weld.calib_H)
            q2=positioner_js[breakpoints[0]]

            R1_mid[0]=deepcopy(q1[0])
            if go_weld:
                if tri_robot:
                    ws.jog_tri(robot_weld,positioner,robot_scan,[R1_mid,q1],q2,ir_js[breakpoints[0]],v=to_start_speed)
                else:
                    ws.jog_dual(robot_weld,positioner,[R1_mid,q1],q2,v=to_start_speed)

            ######################################################
            ########### Do welding #############
            q1_all=[curve_sliced_js[breakpoints[0]]]
            q2_all=[positioner_js[breakpoints[0]]]
            qir_all=[ir_js[breakpoints[0]]]
            v1_all=[1]
            v2_all=[10]
            primitives=['movej']
            for j in range(1,len(breakpoints)):
                q1_all.append(curve_sliced_js[breakpoints[j]])
                q2_all.append(positioner_js[breakpoints[j]])
                qir_all.append(ir_js[breakpoints[j]])
                v1_all.append(max(s1_all[j-1],0.1))
                
                positioner_w=this_weld_v[j-1]/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
                v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
                primitives.append('movel')

            # input("Start Weld")
            weld_weld_st=time.time()

            if go_weld:
                ####DATA LOGGING
                rr_sensors.start_all_sensors()
                if tri_robot:
                    rob_stamps,rob_js_exe,_,_=ws.weld_segment_tri(primitives,robot_weld,positioner,robot_scan,q1_all,q2_all,qir_all,v1_all,10*np.ones(len(v1_all)),cond_all=[weld_job],arc=arc_on)
                else:
                    rob_stamps,rob_js_exe,_,_=ws.weld_segment_dual(primitives,robot_weld,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[weld_job],arc=arc_on)
                rr_sensors.stop_all_sensors()
                print("Actual weld time:",time.time()-weld_weld_st)

                if save_weld_record:
                    Path(data_dir).mkdir(exist_ok=True)
                    if not baselayer:
                        layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
                    else:
                        layer_data_dir=data_dir+'baselayer_'+str(layer)+'_'+str(x)+'/'
                    Path(layer_data_dir).mkdir(exist_ok=True)
                    np.savetxt(layer_data_dir + 'weld_js_exe.csv',rob_js_exe,delimiter=',')
                    np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',rob_stamps,delimiter=',')
                    rr_sensors.save_all_sensors(layer_data_dir)

    ## move R1 back to home
    # print(q1_all)
    # input("Weld Move to Home")
    r1_current = ws.client.getJointAnglesMH(robot_weld.pulse2deg)
    R1_mid[0]=deepcopy(r1_current[0])
    ws.jog_single(robot_weld,[R1_mid,R1_home],v=to_home_speed)
    print("Weld time: ",time.time()-weld_st)

    #### scanning
    if True:
        scan_st=time.time()
        if layer<0:
            read_layer=0
        else:
            read_layer=layer
            
        pcd_layer=o3d.geometry.PointCloud()
        layer_curve_relative=[]
        layer_curve_dh=[]
        for x in range(0,num_sections): 
            ### scanning path module
            spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)

            if not baselayer:
                curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
                curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            else:
                curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/baselayer'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
                curve_sliced_js=np.loadtxt(curve_data_dir+'curve_sliced_js/MA2010_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                positioner_js=np.loadtxt(curve_data_dir+'curve_sliced_js/D500B_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            rob_js_plan = np.hstack((curve_sliced_js,positioner_js))

            # if len(curve_sliced_relative)<2:
            #     continue
            print("Scan Layer",layer,", Sec",x)
            # try:
            if not baselayer:
                q_out1=np.loadtxt(curve_data_dir+'curve_scan_js/MA1440_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                q_out2=np.loadtxt(curve_data_dir+'curve_scan_js/D500B_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,2))
                scan_p=np.loadtxt(curve_data_dir+'curve_scan_relative/scan_T'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            else:
                q_out1=np.loadtxt(curve_data_dir+'curve_scan_js/MA1440_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
                q_out2=np.loadtxt(curve_data_dir+'curve_scan_js/D500B_base_js'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,2))
                scan_p=np.loadtxt(curve_data_dir+'curve_scan_relative/scan_base_T'+str(read_layer)+'_'+str(x)+'.csv',delimiter=',')
            
            ## get breakpoints
            lam_relative=calc_lam_cs(scan_p)
            # scan_waypoint_distance=10 ## mm
            scan_waypoint_distance=waypoint_distance ## mm
            num_points_layer=max(2,int(lam_relative[-1]/scan_waypoint_distance))
            
            ## using forward/backward technique
            # if forward:
            #     breakpoints=np.linspace(0,len(lam_relative)-1,num=num_points_layer).astype(int)
            # else:
            #     breakpoints=np.linspace(len(lam_relative)-1,0,num=num_points_layer).astype(int)

            while True:
                ###find which end to start depending on how close to the current positioner pose
                breakpoints=np.linspace(0,len(lam_relative)-1,num=num_points_layer).astype(int)
                q_prev=ws.client.getJointAnglesDB(positioner.pulse2deg)
                if np.linalg.norm(q_prev-q_out2[0])>np.linalg.norm(q_prev-q_out2[-1]):
                    breakpoints=breakpoints[::-1]
                # generate motion program
                q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,breakpoints=breakpoints,init_sync_move=0)
                v1_all=[1]
                v2_all=[1]
                primitives=['movej']
                for j in range(1,len(breakpoints)):
                    v1_all.append(max(s1_all[j-1],0.1))
                    positioner_w=scan_speed/np.linalg.norm(scan_p[breakpoints[j]][:2])
                    v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
                    primitives.append('movel')
                #######################################

                ######## scanning motion #########
                ### execute motion ###

                # input("Scan Move to Start")
                ## move to mid point and start 
                waypoint_pose=robot_scan.fwd(q_bp1[0][0])
                waypoint_pose.p[-1]+=50
                
                try:
                    q1=robot_scan.inv(waypoint_pose.p,waypoint_pose.R,q_bp1[0][0])[0]
                except:
                    print("Use nom PH for ik")
                    robot_scan.robot.P=deepcopy(r2_nom_P)
                    robot_scan.robot.H=deepcopy(r2_nom_H)
                    q1=robot_scan.inv(waypoint_pose.p,waypoint_pose.R,q_bp1[0][0])[0]
                    robot_scan.robot.P=deepcopy(robot_scan.calib_P)
                    robot_scan.robot.H=deepcopy(robot_scan.calib_H)
                
                q2=q_bp2[0][0]
                if x==0:
                    ws.jog_dual(robot_scan,positioner,[R2_mid,q1],q2,v=to_start_speed)
                else:
                    ws.jog_dual(robot_scan,positioner,q1,q2,v=to_start_speed)
                
                # input("Start Scan")
                scan_scan_st=time.time()
                mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
                target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
                mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
                ws.client.execute_motion_program(mp)

                ## motion start
                mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
                # routine motion
                for path_i in range(1,len(q_bp1)-1):
                    target2=['MOVJ',np.degrees(q_bp2[path_i][0]),v2_all[path_i]]
                    mp.MoveL(np.degrees(q_bp1[path_i][0]), v1_all[path_i], target2=target2)
                target2=['MOVJ',np.degrees(q_bp2[-1][0]),v2_all[-1]]
                mp.MoveL(np.degrees(q_bp1[-1][0]), v1_all[-1], 0, target2=target2)

                
                mti_break_flag=False
                ws.client.execute_motion_program_nonblocking(mp)
                ###streaming
                ws.client.StartStreaming()
                start_time=time.time()
                state_flag=0
                joint_recording=[]
                robot_stamps=[]
                mti_recording=None
                mti_recording=[]
                r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
                while True:
                    if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
                        break 
                    res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
                    if res:
                        joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
                        state_flag=fb_data.controller_flags
                        joint_recording.append(joint_angle)
                        timestamp=fb_data.time
                        robot_stamps.append(timestamp)
                        ###MTI scans YZ point from tool frame
                        try:
                            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
                        except Exception as e:
                            if not mti_break_flag:
                                print(e)
                            mti_break_flag=True
                ws.client.servoMH(False)
                print("Actual scan time:",time.time()-scan_scan_st)
                if not mti_break_flag:
                    break
                print("MTI broke during robot move")
                while True:
                    try:
                        input("MTI reconnect ready?")
                        generate_mti_rr()
                        break
                    except:
                        pass
                
            
            mti_recording=np.array(mti_recording)
            joint_recording=np.array(joint_recording)
            q_out_exe=joint_recording[:,6:]
            #####################
            # exit()

            print("Total exe len:",len(q_out_exe))
            if save_output_points:
                if not baselayer:
                    layer_data_dir=data_dir+'layer_'+str(layer)+'_'+str(x)+'/'
                else:
                    layer_data_dir=data_dir+'baselayer_'+str(layer)+'_'+str(x)+'/'
                Path(data_dir).mkdir(exist_ok=True)
                Path(layer_data_dir).mkdir(exist_ok=True)
                out_scan_dir = layer_data_dir+'scans/'
                Path(out_scan_dir).mkdir(exist_ok=True)
                ## save traj
                # save poses
                np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
                np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
                with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
                    pickle.dump(mti_recording, file)
                print('Total scans:',len(mti_recording))
            ########################

            pcd_procss_st = time.time()
            #### scanning process: processing point cloud and get h
            # curve_sliced_relative=np.array(curve_sliced_relative)
            crop_extend=15
            crop_min=tuple(np.min(curve_sliced_relative[:,:3],axis=0)-crop_extend)
            crop_max=tuple(np.max(curve_sliced_relative[:,:3],axis=0)+crop_extend)
            scan_process = ScanProcess(robot_scan,positioner)
            pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,use_calib=True,ph_param=ph_param_r2)
            cluser_minp = 300
            while True:
                pcd_new = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                    min_bound=crop_min,max_bound=crop_max,outlier_remove=True,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=cluser_minp)
                # visualize_pcd([pcd_new])
                break
                while True:
                    q=input("Continue?")
                    if q=='':
                        break
                    try:
                        cluser_minp=int(q)
                        break
                    except:
                        continue
                if q=='':
                    break
            pcd=pcd_new
            
            # calibrate H
            pcd,Transz0_H = scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)
            print("Transz0_H:",Transz0_H)
            
            # record dh and curve relative
            if layer!=-1:
                # profile_dh = scan_process.pcd2dh(pcd,last_pcd_layer,curve_sliced_relative,robot_weld,rob_js_plan,ph_param=ph_param_r1,drawing=True)
                profile_dh = scan_process.pcd2dh(pcd,curve_sliced_relative,drawing=draw_dh)
            
                if len(layer_curve_dh)!=0:
                    profile_dh[:,0]=profile_dh[:,0]+layer_curve_dh[-1][0]
                layer_curve_dh.extend(profile_dh)
            layer_curve_relative.extend(curve_sliced_relative)
            
            # save dh and pcd
            if save_output_points:
                o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
                if layer!=-1:
                    np.save(out_scan_dir+'height_profile.npy',profile_dh)
            pcd_layer+=pcd

            pcd_process_dt=time.time()-pcd_procss_st
            print("PCD process dt:",pcd_process_dt)
        
        # update
        last_pcd_layer=deepcopy(pcd_layer)
        last_layer_curve_relative=np.array(layer_curve_relative)
        layer_curve_dh=np.array(layer_curve_dh)

        # curve_i=0
        # total_curve_i = len(layer_curve_dh)
        # ax = plt.figure().add_subplot()
        # for curve_i in range(total_curve_i):
        #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
        #     ax.scatter(layer_curve_dh[curve_i,0],layer_curve_dh[curve_i,1],c=color_dist)
        # ax.set_xlabel('Lambda')
        # ax.set_ylabel('dh to Layer N (mm)')
        # ax.set_title("dH Profile")
        # plt.ion()
        # plt.show(block=False)
        
        # curve_i=0
        # total_curve_i = len(layer_curve_height)
        # layer_curve_relative=np.array(layer_curve_relative)
        # lam_curve = calc_lam_cs(layer_curve_relative[:,:3])
        # ax = plt.figure().add_subplot()
        # for curve_i in range(total_curve_i):
        #     color_dist = plt.get_cmap("rainbow")(float(curve_i)/total_curve_i)
        #     ax.scatter(lam_curve[curve_i],layer_curve_height[curve_i],c=color_dist)
        # ax.set_xlabel('Lambda')
        # ax.set_ylabel('Layer N Height (mm)')
        # ax.set_title("Height Profile")
        # plt.show(block=False)

    # input("Scan Move to Home")
    # move robot to home
    ws.jog_dual(robot_scan,positioner,[R2_mid,R2_home],[0,q_prev[1]],v=to_home_speed)
    # ws.jog_single(robot_scan,R2_home,v=to_home_speed)
    print("Scan motion time:",time.time()-scan_st-pcd_process_dt)
    
    ## increase layer count
    layer_count+=1

print("Welding End!!")
exit()