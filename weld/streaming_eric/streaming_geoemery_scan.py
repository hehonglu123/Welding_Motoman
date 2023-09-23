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
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from PH_interp import *
from dx200_motion_program_exec_client import *
from StreamingSend import *
sys.path.append('../')
from weldRRSensor import *

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

R1_ph_dataset_date='0801'
R2_ph_dataset_date='0801'
S1_ph_dataset_date='0801'

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

#### data ####
dataset='circle_large/'
sliced_alg='static_spiral/'
data_dir='../../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]

data_date = input("Use old data directory? (Enter or put time e.g. 2023_07_11_16_25_30): ")
if data_date == '':
    recorded_data_dir=data_dir+'weld_scan_'+formatted_time+'/'
else:
    recorded_data_dir=data_dir+'weld_scan_'+data_date+'/'
print("Use data directory:",data_dir)

########################################################RR FRONIUS########################################################
fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.prepare_welder()
########################################################RR STREAMING########################################################
RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
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
point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)
##################### mti ################
mti_sub=RRN.SubscribeService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_sub.ClientConnectFailed += connect_failed
mti_client=mti_sub.GetDefaultClientWait(1)
mti_client.setExposureTime("25")
#############################

###set up control parameters
job_offset=200 		###200 for Aluminum ER4043 (215,100 ipm), 300 for Steel Alloy ER70S-6 (300 ipm, 3 mm/sec), 400 for Stainless Steel ER316L
nominal_feedrate=170
d_feedrate = -10
nominal_vd_relative=10
# nominal_slice_increment=int(1.84/slicing_meta['line_resolution'])
nominal_slice_increment=26
scanner_lag_bp=int(slicing_meta['scanner_lag_breakpoints'])
scanner_lag=slicing_meta['scanner_lag']
slice_inc_gain=3.
end_layer_count = 7

arc_on=True

res, robot_state, _ = RR_robot_state.TryGetInValue()
q14=robot_state.joint_position

welding_started=False

####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]
lam_relative_all_slices=[]
lam_relative_dense_all_slices=[]
for i in range(0,250):
    rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
    rob2_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','))
    positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
    curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=',')
    lam_relative=calc_lam_cs(curve_sliced_relative)
    lam_relative_all_slices.append(lam_relative)
    lam_relative_dense_all_slices.append(np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance)))

input("Enter to start")
layer_count=0
slice_num=0
# feedrate_cmd=nominal_feedrate
feedrate_cmd=250
# vd_relative=nominal_vd_relative
vd_relative=8
robot_logging_all=[]
weld_logging_all=[]
mti_logging_all=[]
while slice_num<len(lam_relative_all_slices):
    print("Layer:",slice_num,"#:",layer_count)
    print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)
    ###change feedrate
    fronius_client.async_set_job_number(int(feedrate_cmd/10)+job_offset, my_handler)
    x=0
    rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
    rob2_js=copy.deepcopy(rob2_js_all_slices[slice_num])
    positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
    
    ###TRJAECTORY WARPING
    if slice_num>10:
        rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-nominal_slice_increment])
        rob2_js_prev=copy.deepcopy(rob2_js_all_slices[slice_num-nominal_slice_increment])
        positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-nominal_slice_increment])
        rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
    if slice_num<slicing_meta['num_layers']-nominal_slice_increment:
        rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+nominal_slice_increment])
        rob2_js_next=copy.deepcopy(rob2_js_all_slices[slice_num+nominal_slice_increment])
        positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+nominal_slice_increment])
        rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
    
    ###find closest %2pi
    num2p=np.round((q14[-1]-positioner_js[0,1])/(2*np.pi))
    positioner_js[:,1]=positioner_js[:,1]+num2p*2*np.pi
    
    curve_js_all_dense=interp1d(lam_relative_all_slices[slice_num],np.hstack((rob1_js,rob2_js,positioner_js)),kind='cubic',axis=0)(lam_relative_dense_all_slices[slice_num])
    ### get breakpoints for vd
    breakpoints=SS.get_breakpoints(lam_relative_dense_all_slices[slice_num],vd_relative)

    curve_js_all_dense=np.array(curve_js_all_dense)
    
    ###start welding at the first layer, then non-stop
    if not welding_started:
        #jog above
        waypoint_pose=robot_weld.fwd(curve_js_all_dense[breakpoints[0],:6])
        waypoint_pose.p[-1]+=50
        waypoint_q=robot_weld.inv(waypoint_pose.p,waypoint_pose.R,curve_js_all_dense[0,:6])[0]
        # SS.jog2q(np.hstack((waypoint_q,np.radians([21,5,-39,0,-47,49]),curve_js_all_dense[0,12:])))
        SS.jog2q(np.hstack((np.radians([-50,28,7,0,-47,0]),curve_js_all_dense[0,6:])))
        # exit()
        SS.jog2q(np.hstack((waypoint_q,curve_js_all_dense[0,6:])))
        SS.jog2q(curve_js_all_dense[breakpoints[0]])
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
        ###start logging
        for bp_idx in range(len(breakpoints)):
            ####################################MTI PROCESSING####################################
            if bp_idx<len(breakpoints)-1: ###no wait at last point
                ###busy wait for accurate 8ms streaming
                while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
                    continue
            ###MTI scans YZ point from tool frame
            st=time.time()
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
            if time.time()-st>0.008:
                print('MTI scan time:',time.time()-st)
                print("What????")
            point_stream_start_time=time.time()
            robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[breakpoints[bp_idx]])
            
            robot_ts.append(robot_timestamp)
            robot_js.append(q14)
        robot_ts=np.array(robot_ts)
        robot_ts=robot_ts-robot_ts[0]
        robot_js=np.array(robot_js)
        robot_logging_all.append(np.hstack((robot_ts.reshape(-1, 1),robot_js)))
        mti_logging_all.append(mti_recording)
        
        ####CONTROL PARAMETERS
        last_slice_num=slice_num
        
        # change feedrate and such
        if layer_count>0: # dont update for baselayer
            if layer_count<1:
                feedrate_cmd=250
                vd_relative=8
            elif layer_count==1:
                feedrate_cmd=nominal_feedrate-d_feedrate
                nominal_slice_increment=18
            feedrate_cmd = feedrate_cmd+d_feedrate
            # speed decrease in ratio, so the increment rate stays the same
            vd_relative = (feedrate_cmd/nominal_feedrate)*nominal_vd_relative 
        layer_count+=1
        # increment slice layer num
        slice_num+=int(nominal_slice_increment)
        if layer_count>=end_layer_count:
            break
        
    except:
        traceback.print_exc()
        fronius_client.stop_weld()
        break

if arc_on:
    fronius_client.stop_weld()

## streaming final for scanner lagging
robot_ts=[]
robot_js=[]
mti_recording=[]
# point_stream_start_time=time.time()
lag_scan_bp=np.argmin(lam_relative_dense_all_slices[last_slice_num]<scanner_lag)
lag_scan_bp_bp=np.argmin(breakpoints<lag_scan_bp)
curve_js_all_dense[:,13]=curve_js_all_dense[:,13]-(curve_js_all_dense[0,13]-curve_js_all_dense[-1,13])
try:
    ###start logging
    for bp_idx in range(0,lag_scan_bp):
        ####################################MTI PROCESSING####################################
        if bp_idx<len(breakpoints)-1: ###no wait at last point
            ###busy wait for accurate 8ms streaming
            while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
                continue
        ###MTI scans YZ point from tool frame
        mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
        point_stream_start_time=time.time()
        robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[breakpoints[bp_idx]])
        
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