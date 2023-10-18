import sys, glob, wave, pickle
import numpy as np
from copy import deepcopy
import yaml
import datetime
from multiprocessing import Process
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from pathlib import Path
from general_robotics_toolbox import *
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
sys.path.append('../../mocap/')
from utils import *
from robot_def import *
from lambda_calc import *
from multi_robot import *
from traj_manipulation import *
from robot_def import *
from scan_utils import *
from scanProcess import *
from PH_interp import *
from dx200_motion_program_exec_client import *
from StreamingSend import *
sys.path.append('../')
from weldRRSensor import *
from streaming_control import *

def my_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

def connect_failed(s, client_id, url, err):
    global mti_sub, mti_client
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
    mti_sub=RRN.SubscribeService(url)
    mti_client=mti_sub.GetDefaultClientWait(1)

R1_ph_dataset_date='0926'
R2_ph_dataset_date='0926'
S1_ph_dataset_date='0926'

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

#### change base H to calibrated ones ####
robot_scan_base = robot_weld.T_base_basemarker.inv()*robot_scan.T_base_basemarker
robot_scan.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot_weld.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))
# robot_weld.robot.P=deepcopy(robot_weld.calib_P)
# robot_weld.robot.H=deepcopy(robot_weld.calib_H)
# robot_scan.robot.P=deepcopy(robot_scan.calib_P)
# robot_scan.robot.H=deepcopy(robot_scan.calib_H)
# positioner.robot.P=deepcopy(positioner.calib_P)
# positioner.robot.H=deepcopy(positioner.calib_H)

#### data ####
dataset='circle_large/'
sliced_alg='static_spiral/'
curve_data_dir='../../data/'+dataset+sliced_alg
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]

data_date = input("Use old data directory? (Enter or put time e.g. 2023_07_11_16_25_30): ")
if data_date == '':
    recorded_data_dir=curve_data_dir+'weld_scan_'+formatted_time+'/'
else:
    recorded_data_dir=curve_data_dir+'weld_scan_'+data_date+'/'
print("Use data directory:",recorded_data_dir)
#################################

########################################################RR FRONIUS########################################################
fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.prepare_welder()
########################################################RR STREAMING########################################################
RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.10:59945?service=robot')
RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
RR_robot = RR_robot_sub.GetDefaultClientWait(1)
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.reset_errors()
RR_robot.enable()
RR_robot.command_mode = halt_mode
time.sleep(0.1)
RR_robot.command_mode = position_mode
streaming_rate=125.
point_distance=0.01		###STREAMING POINT INTERPOLATED DISTANCE
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)
##################### mti ################
mti_sub=RRN.SubscribeService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_sub.ClientConnectFailed += connect_failed
mti_client=mti_sub.GetDefaultClientWait(1)
mti_client.setExposureTime("25")
###############################################################

###set up control parameters
job_offset=200 		###200 for Aluminum ER4043 (215,100 ipm), 300 for Steel Alloy ER70S-6 (300 ipm, 3 mm/sec), 400 for Stainless Steel ER316L
nominal_feedrate=160
base_nominal_feedrate=250
nominal_vd_relative=10
base_nominal_vd_relative=8
segment_distance=0.8 ## segment d, such that ||xi_{j}-xi_{j-1}||=0.8
base_nominal_slice_increment=26
delta_h_star = 1.8
nominal_slice_increment=int(delta_h_star/slicing_meta['line_resolution'])
base_layer_N=2
scanner_lag_bp=int(slicing_meta['scanner_lag_breakpoints'])
scanner_lag=slicing_meta['scanner_lag']
end_layer_count = 10
offset_z=0

weld_feedback_gain_K=1

welding_started=False
arc_on=False
##############################

### scan process object
scan_process = ScanProcess(robot_scan,positioner)
#######################

### get the initial robot pose
res, robot_state, _ = RR_robot_state.TryGetInValue()
q14=robot_state.joint_position
##############################

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

input("Enter to start")

### set up variables for welding and scanning
feedrate_cmd=base_nominal_feedrate
vd_relative=base_nominal_vd_relative
x_state_location=[]
x_state_dh = []

robot_logging_all=[]
weld_logging_all=[]
mti_logging_all=[]
for layer_count in range(0,end_layer_count):
    base_layer=True if layer_count<base_layer_N else False # baselayer flag
    feedrate_cmd = base_nominal_feedrate if base_layer else nominal_feedrate # set feedrate
    vd_relative = base_nominal_vd_relative if base_layer else nominal_vd_relative # set feedrate
    
    layer=all_layers[layer_count] # get current layer
    print("Layer:",layer,"#:",layer_count)
    print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)
    ###change feedrate
    fronius_client.async_set_job_number(int(feedrate_cmd/10)+job_offset, my_handler)
    
    ## the current curve, the next path curve and the next target curve
    path_curve_dense_relative=deepcopy(curve_relative_dense_all_slices[layer_count])
    lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count])
    if layer_count<end_layer_count:
        target_curve_dense_relative=curve_relative_dense_all_slices[layer_count+1]
        target_lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count+1])
    
    ## get the executed warp robot path (js)
    rob1_js=rob1_js_warp_all_slices[layer_count]
    rob2_js=rob2_js_warp_all_slices[layer_count]
    positioner_js=positioner_warp_js_all_slices[layer_count]
    
    ## get the segment for changing velocity
    num_segbp_layer=max(2,int(lam_relative_dense[-1]/segment_distance))
    segment_bp = np.linspace(0,len(lam_relative_dense)-1,num=num_segbp_layer).astype(int)
    segment_bp_sample = (np.diff(segment_bp)/2+segment_bp[:-1])
    segment_bp_sample=segment_bp_sample.astype(int) # sample the middle point
    if len(x_state_location)==0:
        x_state_location = deepcopy(path_curve_dense_relative[segment_bp_sample][:,:3])
        x_state_location = list(x_state_location)
        x_state_dh = [[] for x in range(len(x_state_location))]
    # segment_bp=segment_bp[1:] # from 1 ~ the last point (ignore 0)
    
    ###find closest %2pi
    num2p=np.round((q14[-1]-positioner_js[0,1])/(2*np.pi))
    positioner_js[:,1]=positioner_js[:,1]+num2p*2*np.pi
    
    curve_js_all_dense=interp1d(lam_relative_all_slices[layer_count],np.hstack((rob1_js,rob2_js,positioner_js)),kind='cubic',axis=0)(lam_relative_dense)

    curve_js_all_dense=np.array(curve_js_all_dense)
    
    ###start welding at the first layer, then non-stop
    if not welding_started:
        #jog above
        waypoint_pose=robot_weld.fwd(curve_js_all_dense[0,:6])
        waypoint_pose.p[-1]+=50
        waypoint_q=robot_weld.inv(waypoint_pose.p,waypoint_pose.R,curve_js_all_dense[0,:6])[0]

        # TODO: Better place for load calib
        # robot_weld.robot.P=deepcopy(robot_weld.calib_P)
        # robot_weld.robot.H=deepcopy(robot_weld.calib_H)
        # robot_scan.robot.P=deepcopy(robot_scan.calib_P)
        # robot_scan.robot.H=deepcopy(robot_scan.calib_H)
        # positioner.robot.P=deepcopy(positioner.calib_P)
        # positioner.robot.H=deepcopy(positioner.calib_H)

        # SS.jog2q(np.hstack((waypoint_q,np.radians([21,5,-39,0,-47,49]),curve_js_all_dense[0,12:])))
        SS.jog2q(np.hstack((np.radians([-50,28,7,0,-47,0]),curve_js_all_dense[0,6:])))
        # exit()
        SS.jog2q(np.hstack((waypoint_q,curve_js_all_dense[0,6:])))
        SS.jog2q(curve_js_all_dense[0])
        welding_started=True
        point_stream_start_time=time.time()
        if arc_on:
            fronius_client.start_weld()
        time.sleep(0.2)

    ## streaming
    robot_ts=[]
    robot_js=[]
    mti_recording=[]
    try:
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
            control_duration=[]
            # print(breakpoints)
            for bp in breakpoints:
                ####################################MTI PROCESSING####################################
                if bp!=len(lam_relative_dense)-1: ###no wait at last point
                    ###busy wait for accurate 8ms streaming
                    while (time.time()-point_stream_start_time)<(1/SS.streaming_rate-0.0005):
                        continue
                control_duration.append(time.time()-point_stream_start_time)
                point_stream_start_time=time.time()
                robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[bp])
                
                ###MTI scans YZ point from tool frame
                st=time.time()
                mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
                ## Get the delta h. TODO: get the correct target p
                if bp<len(curve_js_all_dense)/2:
                    target_p = deepcopy(path_curve_dense_relative[0][:3])+np.array([0,0,delta_h_star])
                else:
                    target_p = deepcopy(target_curve_dense_relative[0][:3])+np.array([0,0,delta_h_star])
                delta_h,point_p = scan_process.scan2dh(deepcopy(mti_recording[-1]),\
                    q14[6:],target_p,crop_min=[-10,85],crop_max=[10,100],offset_z=offset_z)
                x_state_location_arr = np.array(x_state_location)
                x_state_i = np.argsort(np.linalg.norm(x_state_location_arr[:,:2]-point_p[:2],axis=1))[0] # only look at xy
                x_state_dh[x_state_i].append(delta_h)
                ################################
                if time.time()-st>0.008:
                    print('MTI scan time:',time.time()-st)
                    print("What????")
                ### record data
                robot_ts.append(robot_timestamp)
                robot_js.append(q14)
            print("Average duration:",np.mean(control_duration))
            
            ### update x_state 
            x_state_location.pop(0)
            x_state_location.append(target_curve_dense_relative[segment_bp_sample[seg_i]][:3])
            x_state_dh.pop(0)
            x_state_dh.append([])
            #########################
            
        robot_ts=np.array(robot_ts)
        robot_ts=robot_ts-robot_ts[0]
        robot_js=np.array(robot_js)
        robot_logging_all.append(np.hstack((robot_ts.reshape(-1, 1),robot_js)))
        mti_logging_all.append(mti_recording)
        
    except:
        traceback.print_exc()
        fronius_client.stop_weld()
        break

if arc_on:
    fronius_client.stop_weld()

vd_relative=nominal_vd_relative
## streaming final for scanner lagging
path_curve_dense_relative=curve_relative_dense_all_slices[layer_count-1]
lam_relative_dense=deepcopy(lam_relative_dense_all_slices[layer_count-1])
## get the executed warp robot path (js)
rob1_js=rob1_js_all_slices[layer_count-1]
rob2_js=rob2_js_all_slices[layer_count-1]
positioner_js=positioner_js_all_slices[layer_count-1]
###find closest %2pi
num2p=np.round((q14[-1]-positioner_js[0,1])/(2*np.pi))
positioner_js[:,1]=positioner_js[:,1]+num2p*2*np.pi
curve_js_all_dense=interp1d(lam_relative_all_slices[layer_count-1],np.hstack((rob1_js,rob2_js,positioner_js)),kind='cubic',axis=0)(lam_relative_dense_all_slices[layer_count-1])
curve_js_all_dense=np.array(curve_js_all_dense)

robot_ts=[]
robot_js=[]
mti_recording=[]
# point_stream_start_time=time.time()
lag_scan_bp=np.argmin(lam_relative_dense<scanner_lag)
lag_scan_bp = int(lag_scan_bp+lag_scan_bp/10)
breakpoints=SS.get_breakpoints(lam_relative_dense[:lag_scan_bp],vd_relative)
try:
    ###start logging
    for bp_idx in range(len(breakpoints)):
        ####################################MTI PROCESSING####################################
        if bp_idx<len(breakpoints)-1: ###no wait at last point
            ###busy wait for accurate 8ms streaming
            while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
                continue
        ###MTI scans YZ point from tool frame
        point_stream_start_time=time.time()
        robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[breakpoints[bp_idx]])
        mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
        
        robot_ts.append(robot_timestamp)
        robot_js.append(q14)
    robot_ts=np.array(robot_ts)
    robot_ts=robot_ts-robot_ts[0]
    robot_js=np.array(robot_js)
    robot_logging_all.append(np.hstack((robot_ts.reshape(-1, 1),robot_js)))
    mti_logging_all.append(mti_recording)
except:
    traceback.print_exc()
    fronius_client.stop_weld()

# exit()
Path(recorded_data_dir).mkdir(exist_ok=True)
with open(recorded_data_dir + 'robot_js.pickle', 'wb') as file:
    pickle.dump(robot_logging_all, file)
with open(recorded_data_dir + 'mti_scans.pickle', 'wb') as file:
    pickle.dump(mti_logging_all, file)
print('Total scans:',len(mti_recording))