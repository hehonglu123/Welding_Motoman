from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from utils import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weldRRSensor import *
from WeldSend import *
from WeldScan import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

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

def ilc_gradient(yk,yk_prime):
    pass

zero_config=np.zeros(6)

R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'
# 0. robots"
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

# print(robot_weld.fwd(zero_config))
# exit()

Table_home_T = positioner.fwd(np.radians([-15,0]))
T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

save_weld_record=True

#### motion parameters
to_start_speed=4
to_home_speed=5
# r2 pose
r1_home = np.zeros(6)
r1_mid = None
# r2_home = np.radians([38.1523,35.3321,-67.5745,145.4767,-87.9081,-96.3553])
r2_home = np.radians([60,20,-10,0,0,0])
r2_mid = np.radians([60,20,-10,0,0,0])
# positioner pose
s1_weld = np.radians([-15,0])
s1_scan = np.radians([-15,-20])

#### weld and curve parameters
ipm_weld=250
ipm_for_calculation=210
dh=2.5
weld_z_height=[0,dh] # two base layer height to first top layer
curve_start=np.array([0,-32,0])
curve_end=np.array([0,40,0])
seg_dist=1.6
seg_N=int(np.linalg.norm(curve_end-curve_start)/seg_dist)
print("Total seg:",seg_N)

#### gradient descent parameters
alpha=1
max_v=20
min_v=1.5
total_iteration=9
##################

#### data dir
current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='data/weld_scan_'+formatted_time+'/'
# data_dir='data/weld_scan_2023_11_01_17_44_58/'

#### RR drivers and all other drivers
robot_client=MotionProgramExecClient()
generate_mti_rr()
ws=WeldSend(robot_client)
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
# cam_ser=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
cam_ser=None
mic_ser=None
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser,microphone_service=mic_ser)
### test sensor (camera, microphone)
# print("Test 3 Sec.")
# rr_sensors.test_all_sensors()
# print(len(rr_sensors.ir_recording))
# exit()
weldscan=WeldScan(robot_weld,robot_scan,positioner,ws,rr_sensors,mti_client,\
                to_start_s=to_start_speed,to_home_s=to_home_speed)
###############

Transz0_H=None
# Transz0_H=np.array([[ 9.99997540e-01,  2.06703673e-06, -2.21825071e-03, -3.46701381e-03],
#  [ 2.06703673e-06,  9.99998263e-01,  1.86365986e-03,  2.91280622e-03],
#  [ 2.21825071e-03, -1.86365986e-03,  9.99995803e-01,  1.56294293e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

weld_arcon=False
wait_signal=False
draw_dh = False

uk = np.ones(seg_N)*dh2v_loglog(dh,ipm_for_calculation) ## inputs
uk_init = deepcopy(uk)
yk_d = np.ones(seg_N)*dh
baselayer_u = deepcopy(uk)
iter_start=5

use_previous=False
if use_previous:
    iter_start=3
    uk=np.loadtxt(data_dir+'iteration_'+str(iter_start-1)+'/input_uk.csv',delimiter=',')
    yk=np.loadtxt(data_dir+'iteration_'+str(iter_start-1)+'/yk.csv',delimiter=',')
    ek = yk-yk_d
    gradient_direction=deepcopy(ek)*-1 # negative direction
    print("gradient:",np.round(gradient_direction,decimals=2))
    uk = uk-alpha*gradient_direction
    print(np.round(uk,decimals=2))
    

input("Start?")
# move robot to ready position
ws.jog_dual(robot_scan,positioner,[r2_mid,r2_home],s1_weld,to_start_speed)
for iter_i in range(iter_start,total_iteration):
    
    ### first and second half in same loop
    print("Iteration",iter_i,"input u:",np.round(uk,decimals=2))
    input("Start?")
    
    Transz0_H=None
    ### first/second half
    # very first layer (always have one base layer for aluminum, learning second layer here)
    this_curve_start=[30*(iter_i%6)-75,curve_end[1],0,0,0,-1] # curve start
    this_curve_end=[30*(iter_i%6)-75,curve_start[1],0,0,0,-1] # curve end
    curve_sliced_relative=np.linspace(this_curve_start,this_curve_end,seg_N+1) # split curve to seg_N segments
    profile_dh,weld_js_exe,weld_stamps,scan_js_exe,scan_stamps,mti_recording,pcd,Transz0_H\
        =weldscan.robot_weld_scan(curve_sliced_relative,curve_sliced_relative,baselayer_u,ipm_weld,T_R1Base_S1TCP,\
                            r1_mid,r1_home,s1_weld,r2_mid,r2_home,s1_scan,\
                            arc_on=weld_arcon,Transz0_H=Transz0_H,draw_dh=draw_dh,wait_signal=wait_signal) # weld and scan
    # save data
    Path(data_dir).mkdir(exist_ok=True)
    iter_data_dir=data_dir+'iteration_'+str(iter_i)+'/'
    Path(iter_data_dir).mkdir(exist_ok=True)
    layer_data_dir=iter_data_dir+'layer_0/'
    Path(layer_data_dir).mkdir(exist_ok=True)
    # save weld record
    np.savetxt(layer_data_dir + 'curve.csv',curve_sliced_relative,delimiter=',')
    np.savetxt(layer_data_dir + 'weld_js_exe.csv',weld_js_exe,delimiter=',')
    np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',weld_stamps,delimiter=',')
    np.savetxt(layer_data_dir + 'scan_js_exe.csv',scan_js_exe,delimiter=',')
    np.savetxt(layer_data_dir + 'scan_robot_stamps.csv',scan_stamps,delimiter=',')
    with open(layer_data_dir + 'mti_scans.pickle', 'wb') as file:
        pickle.dump(mti_recording, file)
    o3d.io.write_point_cloud(layer_data_dir+'processed_pcd.pcd',pcd)
    np.save(layer_data_dir+'height_profile.npy',profile_dh)
    
    # weld first/second half
    this_curve_start=[30*(iter_i%6)-75,curve_start[1],dh,0,0,-1]
    this_curve_end=[30*(iter_i%6)-75,curve_end[1],dh,0,0,-1]
    curve_sliced_relative=np.linspace(this_curve_start,this_curve_end,seg_N+1)
    uk_input = deepcopy(uk)
    print("Input u:",np.round(uk_input,decimals=2))
    # uk_input = deepcopy(uk_init) # for testing
        
    profile_dh,weld_js_exe,weld_stamps,scan_js_exe,scan_stamps,mti_recording,pcd,Transz0_H\
        =weldscan.robot_weld_scan(curve_sliced_relative,curve_sliced_relative[::-1],uk_input,ipm_weld,T_R1Base_S1TCP,\
                            r1_mid,r1_home,s1_weld,r2_mid,r2_home,s1_scan,\
                            arc_on=weld_arcon,Transz0_H=Transz0_H,draw_dh=draw_dh,wait_signal=wait_signal) # weld and scan
    # save data
    layer_data_dir=iter_data_dir+'layer_1/'
    Path(layer_data_dir).mkdir(exist_ok=True)
    # save weld record
    np.savetxt(layer_data_dir + 'curve.csv',curve_sliced_relative,delimiter=',')
    np.savetxt(layer_data_dir + 'weld_js_exe.csv',weld_js_exe,delimiter=',')
    np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',weld_stamps,delimiter=',')
    np.savetxt(layer_data_dir + 'scan_js_exe.csv',scan_js_exe,delimiter=',')
    np.savetxt(layer_data_dir + 'scan_robot_stamps.csv',scan_stamps,delimiter=',')
    with open(layer_data_dir + 'mti_scans.pickle', 'wb') as file:
        pickle.dump(mti_recording, file)
    o3d.io.write_point_cloud(layer_data_dir+'processed_pcd.pcd',pcd)
    np.save(layer_data_dir+'height_profile.npy',profile_dh)
    
    ### output and error calculation
    yk = deepcopy(profile_dh[:,1])
    yk = moving_average(yk,n=2)
    ek = yk-yk_d
    
    # save data
    np.savetxt(iter_data_dir+'input_uk.csv',uk_input,delimiter=',')
    np.savetxt(iter_data_dir+'yk.csv',yk,delimiter=',')

    ### find gradient and update
    gradient_direction=deepcopy(ek)*-1 # negative direction
    print("gradient:",np.round(gradient_direction,decimals=2))
    uk = uk-alpha*gradient_direction
    uk=np.clip(uk,min_v,max_v)
    