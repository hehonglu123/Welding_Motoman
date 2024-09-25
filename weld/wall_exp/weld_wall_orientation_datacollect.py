from copy import deepcopy
from pathlib import Path
import pickle
import sys

sys.path.append('../')
# sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
# from robot_def import *
from motoman_def import *
from scan_utils import *
from scan_continuous import *
# from scanPathGen import *
from scanProcess import *
from weldCorrectionStrategy import *
from weldRRSensor import *
from WeldSend import *
from lambda_calc import *
from dual_robot import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import yaml

zero_config=np.zeros(6)
# 0. robots. Note use "(robot)_pose_mocapcalib.csv"
config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_flir=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

positioner_weld_js = np.radians([-15,180])
Table_home_T = positioner.fwd(np.radians([-15,180]))
S1_base_T = Transform(positioner.base_H[:3,:3],positioner.base_H[:3,-1])
T_S1TCP_R1Base = S1_base_T*Table_home_T
T_R1Base_S1TCP = T_S1TCP_R1Base.inv()

### the laser scanner is lagging about 33 mm in y direction
# print("Robot Weld TCP:",robot_weld.fwd(np.zeros(6)))
# scan_tcp = robot_scan.fwd(np.zeros(6))
# print("Robot Scan TCP:",robot_scan.fwd(np.zeros(6)))
# print("Robot Scan TCP adjust:",scan_tcp.p + scan_tcp.R[:,-1]*110)
# print(T_S1TCP_R1Base)
# p_relative = np.array([0,0,0])
# R = np.array([[0,1,0],[1,0,0],[0,0,-1]]).T
# T_test_R1 = Transform(T_S1TCP_R1Base.R@R,T_S1TCP_R1Base.R@p_relative+T_S1TCP_R1Base.p)
# print("Test R1:",T_test_R1)
# j_test = robot_weld.inv(T_test_R1.p,T_test_R1.R,zero_config)[0]
# print("Test joint:",j_test)
# T_scan_R1 = robot_scan.fwd(j_test)
# T_scan_S1 = Transform(T_R1Base_S1TCP.R@T_scan_R1.R,T_R1Base_S1TCP.R@T_scan_R1.p+T_R1Base_S1TCP.p)
# print("Test Scan:",T_scan_S1)
# print("Test Scan Adjust:",T_scan_S1.p+T_scan_S1.R[:,-1]*109)
# exit()

#### Welding Parameters ####
total_base_layer = 2
total_weld_layer = 10
weld_arcon=True

nominal_base_height = 3
nominal_weld_height = 1.2

torch_angle = 0 # 0, 10,-10
############################

#######################################ER4043########################################################
job_offset=200
vd_relative=8
feedrate_cmd=110
base_vd_relative=3
base_feedrate_cmd=300
####################################################################################################

####### Motion Parameters ########
to_start_speed=1
to_home_speed=5
# ir pose
r2_ir_q = np.radians([42.5408,35.4021,-63.2228,138.4764,-82.3411,-96.1772])
r2_mid = np.radians([43.7851,20,-10,0,0,0])
# weld bead location
shift_y = -40

scan_shift_z = 10

slice_dp=0.2
laser_lagging=33
laser_lagging_N = int(laser_lagging/slice_dp)
waypoint_distance=5
##################################

####### Data parameters #######
save_weld_record=True
test_sensor_only=False

test_meta = {'total_base_layer':total_base_layer,'total_weld_layer':total_weld_layer,'base_vd_relative':base_vd_relative,'vd_relative':vd_relative,\
             'base_feedrate_cmd':base_feedrate_cmd,'feedrate_cmd':feedrate_cmd,'nominal_base_height':nominal_base_height,'nominal_weld_height':nominal_weld_height,\
             'torch_angle':torch_angle}

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../../data/wall_weld_test/torch_ori'+str(int(torch_angle))+'_'+formatted_time+'/'
print(data_dir)
###############################

print('Total base layer:',total_base_layer)
print('Total weld layer:',total_weld_layer)
print('Base Velocity:',base_vd_relative)
print('Velocity:',vd_relative)
print('Base feedrate:',base_feedrate_cmd)
print('Feedrate:',feedrate_cmd)
print("Moving start/home speed:",to_start_speed,to_home_speed)
print('Torch angle:',torch_angle)

## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
# weld state logging
# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
weld_ser = None
cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
# cam_ser=None
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser)
fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
sub=RRN.SubscribeService(fujicam_url)
obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
scan_change=sub.SubscribeWire("lineProfile")
sub.ClientConnectFailed += connect_failed

### test sensor (camera, microphone)
if test_sensor_only:
    print("Test 3 Sec.")
    rr_sensors.test_all_sensors()
    print(len(rr_sensors.ir_recording))
    exit()
###############

mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot_weld.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
client=MotionProgramExecClient()
ws=WeldSend(client)

## pre-generate robot joint path
curve_RWeld_js_layers = []
curve_scan_js_layers = []
curve_slice_relative_layers = []
curve_scan_relative_layers = []
for base_i in range(total_base_layer):
    print('Base Layer:',base_i)
    curve_sliced_relative_start = np.array([-42.5,shift_y,base_i*nominal_base_height])
    curve_sliced_relative_end = np.array([42.5,shift_y,base_i*nominal_base_height])
    if base_i%2==1:
        curve_sliced_relative_start,curve_sliced_relative_end = curve_sliced_relative_end,curve_sliced_relative_start
    
    slice_N = int(np.linalg.norm(curve_sliced_relative_end-curve_sliced_relative_start)/slice_dp)
    curve_slice_relative_p = np.linspace(curve_sliced_relative_start,curve_sliced_relative_end,slice_N)
    # get Rotation matrix and solve joint angles
    curve_js = [zero_config]
    curve_slice_relative = []
    for i,curve_p in enumerate(curve_slice_relative_p):
        if base_i%2==1:
            if i<=laser_lagging_N:
                RyAxis = curve_slice_relative_p[0]-curve_slice_relative_p[laser_lagging_N]
            else:
                RyAxis = curve_slice_relative_p[i-laser_lagging_N]-curve_slice_relative_p[i]
        else:
            if i<slice_N-laser_lagging_N:
                RyAxis = curve_slice_relative_p[i+laser_lagging_N]-curve_slice_relative_p[i]
            else:
                RyAxis = curve_slice_relative_p[-1]-curve_slice_relative_p[-1-laser_lagging_N]
        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RzAxis = np.array([0,0,-1])
        RyAxis = RyAxis - np.dot(RyAxis,RzAxis)*RzAxis
        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RxAxis = np.cross(RyAxis,RzAxis)
        # R_S1TCP = (np.array([RxAxis,RyAxis,RzAxis]).T)@rot([1,0,0],torch_angle)
        R_S1TCP = np.array([RxAxis,RyAxis,RzAxis]).T
        Target_R1Base = Transform(T_S1TCP_R1Base.R@R_S1TCP,T_S1TCP_R1Base.R@curve_p+T_S1TCP_R1Base.p)
        this_q = robot_weld.inv(Target_R1Base.p,Target_R1Base.R,curve_js[-1])[0]
        curve_js.append(this_q)
        curve_slice_relative.append(np.append(curve_p,R2q(R_S1TCP)))
    curve_js = curve_js[1:]
    curve_RWeld_js_layers.append(np.array(curve_js))
    curve_slice_relative_layers.append(np.array(curve_slice_relative))
    curve_scan_relative_layers.append(None)
    curve_scan_js_layers.append(None)

# print(robot_weld.fwd(curve_RWeld_js_layers[-1][-1]))
for weld_i in range(total_weld_layer):
    print('Weld Layer:',weld_i)
    curve_sliced_relative_start = np.array([-32.5,shift_y,total_base_layer*nominal_base_height+weld_i*nominal_weld_height])
    curve_sliced_relative_end = np.array([32.5,shift_y,total_base_layer*nominal_base_height+weld_i*nominal_weld_height])
    if weld_i%2==1:
        curve_sliced_relative_start,curve_sliced_relative_end = curve_sliced_relative_end,curve_sliced_relative_start
    
    curve_scan_relative_start = curve_sliced_relative_end + np.array([0,0,scan_shift_z])
    curve_scan_relative_end = curve_sliced_relative_start + np.array([0,0,scan_shift_z])
    if weld_i%2==0:
        curve_scan_relative_end = curve_scan_relative_end + (curve_scan_relative_end-curve_scan_relative_start)/np.linalg.norm((curve_scan_relative_end-curve_scan_relative_start))*laser_lagging
    else:
        curve_scan_relative_start = curve_scan_relative_start + (curve_scan_relative_start-curve_scan_relative_end)/np.linalg.norm((curve_scan_relative_start-curve_scan_relative_end)) *laser_lagging

    # print('Sliced:',curve_sliced_relative_start,curve_sliced_relative_end)
    # print('Scan:',curve_scan_relative_start,curve_scan_relative_end)

    slice_N = int(np.linalg.norm(curve_sliced_relative_end-curve_sliced_relative_start)/slice_dp)
    curve_slice_relative_p = np.linspace(curve_sliced_relative_start,curve_sliced_relative_end,slice_N)
    slice_scan_N = int(np.linalg.norm(curve_scan_relative_end-curve_scan_relative_start)/slice_dp)
    curve_scan_relative_p = np.linspace(curve_scan_relative_start,curve_scan_relative_end,slice_scan_N)
    # get Rotation matrix and solve joint angles
    curve_js = [zero_config]
    curve_scan_js = [zero_config]
    curve_slice_relative = []
    curve_scan_relative = []
    for i,curve_p in enumerate(curve_slice_relative_p):
        if weld_i%2==1:
            if i<=laser_lagging_N:
                RyAxis = curve_slice_relative_p[0]-curve_slice_relative_p[laser_lagging_N]
            else:
                RyAxis = curve_slice_relative_p[i-laser_lagging_N]-curve_slice_relative_p[i]
        else:
            if i<slice_N-laser_lagging_N:
                RyAxis = curve_slice_relative_p[i+laser_lagging_N]-curve_slice_relative_p[i]
            else:
                RyAxis = curve_slice_relative_p[-1]-curve_slice_relative_p[-1-laser_lagging_N]

        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RzAxis = np.array([0,0,-1])
        RyAxis = RyAxis - np.dot(RyAxis,RzAxis)*RzAxis
        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RxAxis = np.cross(RyAxis,RzAxis)
        R_S1TCP = (np.array([RxAxis,RyAxis,RzAxis]).T)
        Target_R1Base = Transform(T_S1TCP_R1Base.R@R_S1TCP,T_S1TCP_R1Base.R@curve_p+T_S1TCP_R1Base.p)
        if weld_i % 2 == 0:
            Target_R1Base.R = Target_R1Base.R@rot([1,0,0],np.radians(torch_angle))
        else:
            Target_R1Base.R = Target_R1Base.R@rot([1,0,0],np.radians(-torch_angle))
        # print(Target_R1Base.p)
        # input(Target_R1Base.R)
        this_q = robot_weld.inv(Target_R1Base.p,Target_R1Base.R,curve_js[-1])[0]
        curve_js.append(this_q)
        curve_slice_relative.append(np.append(curve_p,R2q(R_S1TCP)))
    # scanning path
    for i,curve_p in enumerate(curve_scan_relative_p):
        if weld_i%2==1:
            if i<slice_scan_N-laser_lagging_N:
                RyAxis = curve_scan_relative_p[i+laser_lagging_N]-curve_scan_relative_p[i]
            else:
                RyAxis = curve_scan_relative_p[-1]-curve_scan_relative_p[-1-laser_lagging_N]
        else:
            if i<=laser_lagging_N:
                RyAxis = curve_scan_relative_p[0]-curve_scan_relative_p[laser_lagging_N]
            else:
                RyAxis = curve_scan_relative_p[i-laser_lagging_N]-curve_scan_relative_p[i]
        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RzAxis = np.array([0,0,-1])
        RyAxis = RyAxis - np.dot(RyAxis,RzAxis)*RzAxis
        RyAxis = RyAxis/np.linalg.norm(RyAxis)
        RxAxis = np.cross(RyAxis,RzAxis)
        R_S1TCP = np.array([RxAxis,RyAxis,RzAxis]).T
        Target_R1Base = Transform(T_S1TCP_R1Base.R@R_S1TCP,T_S1TCP_R1Base.R@curve_p+T_S1TCP_R1Base.p)
        Target_R1Base.R = Target_R1Base.R@rot([1,0,0],np.radians(-20))
        try:
            this_q = robot_weld.inv(Target_R1Base.p,Target_R1Base.R,curve_js[-1])[0]
        except:
            print("Error at:",i)
            print(Target_R1Base.p,Target_R1Base.R)
            print(curve_js[-1])
            print(robot_weld.fwd(curve_js[-1]))
            exit()
        curve_scan_js.append(this_q)
        curve_scan_relative.append(np.append(curve_p,R2q(R_S1TCP)))

    curve_js = curve_js[1:]
    curve_RWeld_js_layers.append(np.array(curve_js))
    curve_slice_relative_layers.append(np.array(curve_slice_relative))

    curve_scan_js = curve_scan_js[1:]
    curve_scan_js_layers.append(np.array(curve_scan_js))
    curve_scan_relative_layers.append(np.array(curve_scan_relative))
#############

input("Press Enter to start welding...")
# ws.jog_dual(robot_flir,positioner,r2_mid,positioner_weld_js,to_start_speed)
# ws.jog_dual(robot_flir,positioner,r2_ir_q,positioner_weld_js,to_start_speed)
# exit()
########### start welding and scanning
for layer_i in range(len(curve_js)):
    print('Layer:',layer_i)
    ########### welding
    lam1=calc_lam_js(curve_RWeld_js_layers[layer_i],robot_weld)
    positioner_js_layer = np.repeat(positioner_weld_js[np.newaxis, :], len(curve_RWeld_js_layers[layer_i]), axis=0)
    lam2=calc_lam_js(positioner_js_layer,positioner)
    lam_relative=calc_lam_cs(curve_slice_relative_layers[layer_i])

    num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
    breakpoints=np.linspace(0,len(curve_RWeld_js_layers[layer_i])-1,num=num_points_layer).astype(int)
    
    if layer_i<total_base_layer:
        s1_all,_=calc_individual_speed(base_vd_relative,lam1,lam2,lam_relative,breakpoints)
    else:
        s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

    q1_all = curve_RWeld_js_layers[layer_i][breakpoints].tolist()
    positioner_all = positioner_js_layer[breakpoints].tolist()
    v1_all = [3]+s1_all
    if layer_i<total_base_layer:
        cond_all = [0]+[int(base_feedrate_cmd/10+job_offset)]*(num_points_layer-1)
    else:
        cond_all = [0]+[int(feedrate_cmd/10+job_offset)]*(num_points_layer-1)
    primitives = ['movej']+['movel']*(num_points_layer-1)

    Tstart_hoffset = robot_weld.fwd(curve_RWeld_js_layers[layer_i][breakpoints[0]])
    qstart_hoffset = robot_weld.inv(Tstart_hoffset.p+np.array([0,0,10]),Tstart_hoffset.R,curve_RWeld_js_layers[layer_i][breakpoints[0]])[0]

    q1_all.insert(0,qstart_hoffset) # add start point
    positioner_all.insert(0,positioner_weld_js)
    v1_all.insert(0,15)
    cond_all.insert(0,0)
    primitives.insert(0,'movel')

    # start weld!
    rr_sensors.start_all_sensors()
    ws.weld_segment_dual(primitives,robot_weld,positioner,q1_all,positioner_all,v1_all,10*np.ones(len(v1_all)),cond_all,arc=weld_arcon,blocking=False)

    ### welding execution and recording
    ###Get robot joint data
    robWeld_js_exe = []
    positioner_js_exe = []
    scan_weld_exe = []
    timestamps_exe = []

    counts=0
    while True:
        res, fb_data = client.fb.try_receive_state_sync(client.controller_info, 0.001)
        if res:
            if fb_data.controller_flags & 0x08 == 0 and counts>1000:
                client.servoMH(False)
                break
            q1_cur=fb_data.group_state[0].feedback_position
            positioner_cur=fb_data.group_state[2].feedback_position
            ###get scan data >1% intensity and >50mm in Z
            wire_packet=scan_change.TryGetInValue()
            valid_indices=np.where(wire_packet[1].I_data>1)[0]
            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))

            timestamps_exe.append(time.perf_counter())
            robWeld_js_exe.append(q1_cur)
            positioner_js_exe.append(positioner_cur)
            scan_weld_exe.append(line_profile)

            counts+=1
    
    rr_sensors.stop_all_sensors()
    
    # save scan data to file
    if save_weld_record:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        if layer_i<total_base_layer:
            layer_dir = data_dir+'baselayer_'+str(layer_i)+'/'
        else:
            layer_dir = data_dir+'weldlayer_'+str(layer_i-total_base_layer)+'/'
        Path(layer_dir).mkdir(parents=True, exist_ok=True)
        np.savetxt(layer_dir+'timestamps_weld.csv',np.array(timestamps_exe),delimiter=',')
        np.savetxt(layer_dir+'robot_weld_js.csv',np.array(robWeld_js_exe),delimiter=',')
        np.savetxt(layer_dir+'positioner_js.csv',np.array(positioner_js_exe),delimiter=',')
        with open(layer_dir+'scan.pkl','wb') as f:
            pickle.dump(scan_weld_exe,f)
        rr_sensors.save_all_sensors(layer_dir)
        # save meta
        yaml.safe_dump(test_meta,open(data_dir+'meta.yaml','w'))
    ########################

    

    ######## scanning
    if layer_i<total_base_layer:
        continue
    lam1=calc_lam_js(curve_scan_js_layers[layer_i],robot_weld)
    positioner_js_layer = np.repeat(positioner_weld_js[np.newaxis, :], len(curve_scan_js_layers[layer_i]), axis=0)
    lam2=calc_lam_js(positioner_js_layer,positioner)
    lam_relative=calc_lam_cs(curve_scan_relative_layers[layer_i])

    num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
    breakpoints=np.linspace(0,len(curve_scan_js_layers[layer_i])-1,num=num_points_layer).astype(int)

    s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

    q1_all = curve_scan_js_layers[layer_i][breakpoints].tolist()
    positioner_all = positioner_js_layer[breakpoints].tolist()
    v1_all = [15]+s1_all
    cond_all = [0]*num_points_layer
    primitives = ['movel']+['movel']*(num_points_layer-1)

    ws.weld_segment_dual(primitives,robot_weld,positioner,q1_all,positioner_all,v1_all,10*np.ones(len(v1_all)),cond_all,arc=False,blocking=False)

    ### scanning execution and recording
    ###Get robot joint data
    timestamps_exe = []
    robScan_js_exe = []
    positioner_js_exe = []
    scan_scan_exe = []

    counts=0
    while True:
        res, fb_data = client.fb.try_receive_state_sync(client.controller_info, 0.001)
        if res:
            if fb_data.controller_flags & 0x08 == 0 and counts>1000:
                client.servoMH(False)
                break
            q1_cur=fb_data.group_state[0].feedback_position
            positioner_cur=fb_data.group_state[2].feedback_position
            ###get scan data >1% intensity and >50mm in Z
            wire_packet=scan_change.TryGetInValue()
            valid_indices=np.where(wire_packet[1].I_data>1)[0]
            valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
            line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))

            timestamps_exe.append(time.perf_counter())
            robScan_js_exe.append(q1_cur)
            positioner_js_exe.append(positioner_cur)
            scan_scan_exe.append(line_profile)

            counts+=1
    
    # save scan data to file
    if save_weld_record:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        layer_dir = data_dir+'weldlayer_'+str(layer_i-total_base_layer)+'/'
        Path(layer_dir).mkdir(parents=True, exist_ok=True)
        np.savetxt(layer_dir+'timestamps_scan.csv',np.array(timestamps_exe),delimiter=',')
        np.savetxt(layer_dir+'robot_scan_js.csv',np.array(robScan_js_exe),delimiter=',')
        np.savetxt(layer_dir+'positioner_js.csv',np.array(positioner_js_exe),delimiter=',')
        with open(layer_dir+'scan.pkl','wb') as f:
            pickle.dump(scan_scan_exe,f)

print("Welding End!!")