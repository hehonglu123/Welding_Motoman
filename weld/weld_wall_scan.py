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
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

weld_timestamp=[]
weld_weld_voltage=[]
weld_weld_current=[]
weld_weld_feedrate=[]
weld_weld_energy=[]

def clean_weld_record():
    global weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy
    weld_timestamp=[]
    weld_voltage=[]
    weld_current=[]
    weld_feedrate=[]
    weld_energy=[]

def wire_cb(sub, value, ts):
    global weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy

    weld_timestamp.append(value.ts['microseconds'][0])
    weld_voltage.append(value.welding_weld_voltage)
    weld_current.append(value.welding_weld_current)
    weld_feedrate.append(value.wire_speed)
    weld_energy.append(value.welding_weld_energy)

def robot_weld_path_gen(all_layer_z,forward_flag,base_layer):
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
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
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml',tool_marker_config_file=config_dir+'scanner_marker_config.yaml')
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

forward_flag = True
base_layer = True

final_height=30
weld_z_height=[0,5,10] # two base layer height to first top layer
weld_z_height=np.append(weld_z_height,np.arange(weld_z_height[-1],final_height+1,1))
job_number=[250,250]
job_number=np.append(job_number,np.ones(len(weld_z_height)-1)*160)
weld_velocity=[5,5]
weld_v=4
for i in range(len(weld_z_height)-2):
    weld_velocity.append(weld_v)
    if weld_v==weld_velocity[-2]:
        weld_v+=2
correction_thres=1.2 ## mm
save_weld_record=True

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_scan_'+formatted_time+'/'

for i in range(len(weld_z_height)):
    if i>=2:
        base_layer=False
    this_z_height=weld_z_height[i]
    this_job_number=job_number[i]
    this_weld_v=[weld_velocity[i]]

    # 1. The curve path in "positioner tcp frame"
    ######### enter your wanted z height #######
    all_layer_z = [this_z_height]
    ###########################################
    all_path_T = robot_weld_path_gen(all_layer_z,forward_flag,base_layer) # this is your path
    path_T=all_path_T[0]
    curve_sliced_relative=[]
    for path_p in path_T:
        this_p = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.p)+T_S1TCP_R1Base[:3,-1]
        this_n = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R[:,-1])
        curve_sliced_relative.append(np.append(this_p,this_n))
    curve_sliced_relative=curve_sliced_relative[1:-1] # the start and end is for collision prevention
    R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R)

    #### Correction ####
    # TODO: Add fitering if near threshold
    h_largest=this_z_height
    if i>1:
        h_largest = np.max(profile_height[:,1])
        h_target = h_largest+1.2

        profile_slope = np.diff(profile_height[:,1])/np.diff(profile_height[:,0])
        profile_slope = np.append(profile_slope[0],profile_slope)
        # find slope peak
        peak_threshold=0.6
        weld_terrain=[]
        last_peak_i=None
        lastlast_peak_i=None
        for sample_i in range(len(profile_slope)):
            if np.fabs(profile_slope[sample_i])<peak_threshold:
                weld_terrain.append(0)
            else:
                if profile_slope[sample_i]>=peak_threshold:
                    weld_terrain.append(1)
                elif profile_slope[sample_i]<=peak_threshold:
                    weld_terrain.append(-1)
                if lastlast_peak_i:
                    if (weld_terrain[-1]==weld_terrain[lastlast_peak_i]) and (weld_terrain[-1]!=weld_terrain[last_peak_i]):
                        weld_terrain[last_peak_i]=0
                lastlast_peak_i=last_peak_i
                last_peak_i=sample_i

        weld_terrain=np.array(weld_terrain)
        weld_peak=[]
        last_peak=None
        last_peak_i=None
        flat_threshold=2.5
        for sample_i in range(len(profile_slope)):
            if weld_terrain[sample_i]!=0:
                if last_peak is None:
                    weld_peak.append(profile_height[sample_i])
                else:
                    # if the terrain change
                    if (last_peak>0 and weld_terrain[sample_i]<0) or (last_peak<0 and weld_terrain[sample_i]>0):
                        weld_peak.append(profile_height[last_peak_i])
                        weld_peak.append(profile_height[sample_i])
                    else:
                        # the terrain not change but flat too long
                        if profile_height[sample_i,0]-profile_height[last_peak_i,0]>flat_threshold:
                            weld_peak.append(profile_height[last_peak_i])
                            weld_peak.append(profile_height[sample_i])
                last_peak=deepcopy(weld_terrain[sample_i])
                last_peak_i=sample_i
        weld_peak=np.array(weld_peak)

        if forward_flag:
            weld_bp = weld_peak[np.arange(0,len(weld_peak)-1,2)+1]
        else:
            weld_bp = weld_peak[np.arange(0,len(weld_peak),2)][::-1]

        plt.scatter(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]))
        plt.plot(profile_height[:,0],profile_slope)
        plt.scatter(weld_peak[:,0],weld_peak[:,1]-np.mean(profile_height[:,1]))
        plt.scatter(weld_bp[:,0],weld_bp[:,1]-np.mean(profile_height[:,1]))
        plt.show()

        # find v
        # 140 ipm: dh=0.006477*v^2-0.2362v+3.339
        # 160 ipm: dh=0.006043*v^2-0.2234v+3.335
        this_weld_v = []
        dh = min(h_target-weld_bp[0][1],2.5) # find first v
        a=0.006477
        b=-0.2362
        c=3.339-dh
        v=-b+np.sqrt(b**2-4*a*c)/(2*a)
        this_weld_v.append(v)
        for bpi in range(len(weld_bp)):
            dh = h_target-weld_bp[bpi][1]
            a=0.006477
            b=-0.2362
            c=3.339-dh
            v=-b+np.sqrt(b**2-4*a*c)/(2*a)
            this_weld_v.append(v)
        # new curve in positioner frame
        curve_sliced_relative_correct = [curve_sliced_relative[0]]
        path_T_S1 = [Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3])]
        for bpi in range(1,len(weld_bp)):
            this_p = np.array([weld_bp[bpi][0],curve_sliced_relative[0][1],h_largest])
            curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_correct[0][3:]))
            path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
        curve_sliced_relative_correct = [curve_sliced_relative[-1]]
        path_T_S1 = [Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3])]
        curve_sliced_relative=curve_sliced_relative_correct
        # find curve in R1 frame
        path_T=[]
        for tcp_T in path_T_S1:
            this_p = T_R1Base_S1TCP[:3,:3]@tcp_T.p+T_R1Base_S1TCP[:3,-1]
            this_R = T_R1Base_S1TCP[:3,:3]@tcp_T.R
            path_T.append(Transform(this_R,this_p))
        # add path collision avoidance
        path_T.insert(0,Transform(path_T[0].R,path_T[0].p+np.array([0,0,10])))
        path_T.append(Transform(path_T[-1].R,path_T[-1].p+np.array([0,0,10])))
    ####################

    # 2. Scanning parameters
    ### scan parameters
    scan_speed=10 # scanning speed (mm/sec)
    scan_stand_off_d = 70 ## mm
    Rz_angle = np.radians(0) # point direction w.r.t welds
    Ry_angle = np.radians(0) # rotate in y a bit
    bounds_theta = np.radians(10) ## circular motion at start and end
    all_scan_angle = np.radians([0]) ## scan angle
    q_init_table=np.radians([-15,90]) ## init table
    save_output_points = True
    ### scanning path module
    spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
    mti_Rpath = np.array([[ 1.,0.,0.],   
                    [ 0.,-1.,0.],
                    [0.,0.,-1.]])
    # generate scan path
    if forward_flag:
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                        solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
    else:
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative[::-1]],[0],all_scan_angle,\
                        solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
    # generate motion program
    q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)
    # print(np.degrees(q_out1[:10]))
    # print(np.degrees(q_out1[-10:]))
    # exit()
    #######################################

    ######################################################
    ########### Do welding #############
    path_q = []
    for tcp_T in path_T:
        path_q.append(robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])
    
    input("Press Enter and move to weld starting point.")
    mp = MotionProgram(ROBOT_CHOICE='RB1', pulse2deg=robot_weld.pulse2deg)
    mp.MoveJ(np.degrees(path_q[0]), 3, 0)
    mp.MoveL(np.degrees(path_q[1]), 10, 0)

    input("Press Enter and start welding.")
    # weld state logging
    sub = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
    obj = sub.GetDefaultClientWait(3)  # connect, timeout=30s
    welder_state_sub = sub.SubscribeWire("welder_state")
    welder_state_sub.WireValueChanged += wire_cb
    mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
    mp.setArc(select, this_job_number)
    for bpi in range(len(this_weld_v)):
        if bpi!=len(this_weld_v)-1:
            mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi])
        else:
            mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi],0)
    mp.setArc(False)
    mp.MoveL(np.degrees(path_q[-1]), 10, 0)
    clean_weld_record()
    rob_stamps,rob_js_exe,_,_= robot_client.execute_motion_program(mp)

    # move home
    input("Press Enter to Move Home")
    mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
    mp.MoveL(np.zeros[6], 3, 0)
    robot_client.execute_motion_program(mp)

    if save_weld_record:
        layer_data_dir=data_dir+'layer_'+str(i)+'/'
        Path(layer_data_dir).mkdir(exist_ok=True)
        np.savetxt(layer_data_dir + 'welding.csv',
                    np.array([weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy]).T, delimiter=',',
                    header='timestamp,voltage,current,feedrate,energy', comments='')
    ######################################################

    ######## scanning motion #########
    ### execute motion ###
    robot_client=MotionProgramExecClient()
    input("Press Enter and move to scanning startint point")

    ## move to start
    to_start_speed=3
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
    mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
    robot_client.execute_motion_program(mp)

    input("Press Enter to start moving and scanning")

    ###MTI connect to RR
    mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_client.setExposureTime("25")

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    # calibration motion
    target2=['MOVJ',np.degrees(q_bp2[1][0]),10]
    mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
    # routine motion
    for path_i in range(2,len(q_bp1)-1):
        target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
        mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
    target2=['MOVJ',np.degrees(q_bp2[-1][0]),10]
    mp.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)

    robot_client.execute_motion_program_nonblocking(mp)
    ###streaming
    robot_client.StartStreaming()
    start_time=time.time()
    state_flag=0
    joint_recording=[]
    robot_stamps=[]
    mti_recording=[]
    r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
    while True:
        if state_flag & 0x08 == 0 and time.time()-start_time>1.:
            break
        res, data = robot_client.receive_from_robot(0.01)
        if res:
            joint_angle=np.radians(np.divide(np.array(data[26:34]),r_pulse2deg))
            state_flag=data[16]
            joint_recording.append(joint_angle)
            timestamp=data[0]+data[1]*1e-9
            robot_stamps.append(timestamp)
            ###MTI scans YZ point from tool frame
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
    robot_client.servoMH(False)

    mti_recording=np.array(mti_recording)
    q_out_exe=joint_recording

    print(np.degrees(joint_recording[-10:]))

    input("Press Enter to Move Home")
    # move robot to home
    q2=np.zeros(6)
    q2[0]=90
    q3=[-15,180]
    mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    mp.MoveJ(q2,5,0)
    robot_client.execute_motion_program(mp)
    mp=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=positioner.pulse2deg)
    mp.MoveJ(q3,10,0)
    robot_client.execute_motion_program(mp)
    #####################
    # exit()

    print("Total exe len:",len(q_out_exe))
    if save_output_points:
        out_scan_dir = layer_data_dir+'scans/'
        ## save traj
        Path(out_scan_dir).mkdir(exist_ok=True)
        # save poses
        np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
        np.savetxt(out_scan_dir + 'robot_stamps.csv',robot_stamps,delimiter=',')
        with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
            pickle.dump(mti_recording, file)
        print('Total scans:',len(mti_recording))

    #### scanning process: processing point cloud and get h
    scan_process = scanProcess(robot_scan,positioner,static_positioner_q=q_init_table)

    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=10,std_ratio=0.85,cluster_based_outlier_remove=False)
    z_height_start=h_largest
    profile_height = scan_process.pcd2height(pcd,z_height_start)

    if np.mean(profile_height[:,1])>final_height:
        break

    forward_flag=not forward_flag

print("Welding End!!")