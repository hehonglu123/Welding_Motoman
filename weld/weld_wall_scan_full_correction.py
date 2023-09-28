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
weld_voltage=[]
weld_current=[]
weld_feedrate=[]
weld_energy=[]

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
    weld_voltage.append(value.welding_voltage)
    weld_current.append(value.welding_current)
    weld_feedrate.append(value.wire_speed)
    weld_energy.append(value.welding_energy)

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
weld_z_height=[0,6,8] # two base layer height to first top layer
weld_z_height=np.append(weld_z_height,np.arange(weld_z_height[-1],final_height,1)+1)
# job_number=[115,115]
job_number=[215,215]
job_number=np.append(job_number,np.ones(len(weld_z_height)-2)*140)
print(weld_z_height)
print(job_number)

weld_velocity=[5,5]
weld_v=4
for i in range(len(weld_z_height)-2):
    weld_velocity.append(weld_v)
    if weld_v==weld_velocity[-2]:
        weld_v+=2
print(weld_velocity)
correction_thres=1.2 ## mm
save_weld_record=True

# exit()

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_scan_'+formatted_time+'/'

## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
# weld state logging
sub = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
obj = sub.GetDefaultClientWait(3)  # connect, timeout=30s
welder_state_sub = sub.SubscribeWire("welder_state")
welder_state_sub.WireValueChanged += wire_cb
# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")
###################################
profile_height=None
last_mean_h = None

for i in range(0,len(weld_z_height)):
    print("Layer:",i+1)
    if True:
        if i>=2:
            base_layer=False
        this_z_height=weld_z_height[i]
        this_job_number=job_number[i]
        this_weld_v=[weld_velocity[i]]
        all_dh=[]

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
        print(curve_sliced_relative)
        R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R)

        #### Correction ####
        # TODO: Add fitering if near threshold
        h_largest=this_z_height
        if i>2: # start correction from 2nd top layer
            
            if profile_height is None:
                profile_height=np.load('../data/wall_weld_test/weld_scan_2023_05_31_16_12_02/layer_6/scans/height_profile.npy')
                data_dir='../data/wall_weld_test/weld_scan_2023_05_31_16_12_02/'
            
            mean_h = np.mean(profile_height[:,1])
            h_thres = 3
            profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h>3),axis=0)
            profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h<-3),axis=0)

            h_largest = np.max(profile_height[:,1])
            
            # 1. h_target = last height point + designated dh value
            h_target = h_largest+1.2
            # 2. h_target = last mean h + last_dh
            # dh_last_layer = mean_h-last_mean_h
            # h_target = mean_h+dh_last_layer

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
                    elif profile_slope[sample_i]<=-peak_threshold:
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

            if not forward_flag:
                weld_bp = weld_peak[np.arange(0,len(weld_peak)-1,2)+1]
            else:
                weld_bp = weld_peak[np.arange(0,len(weld_peak),2)][::-1]

            plt.scatter(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]))
            plt.plot(profile_height[:,0],profile_slope)
            plt.scatter(weld_peak[:,0],weld_peak[:,1]-np.mean(profile_height[:,1]))
            plt.scatter(weld_bp[:,0],weld_bp[:,1]-np.mean(profile_height[:,1]))
            plt.show()
            # exit()

            # find v
            # 140 ipm: dh=0.006477*v^2-0.2362v+3.339
            # 160 ipm: dh=0.006043*v^2-0.2234v+3.335    
            # new curve in positioner frame
            curve_sliced_relative_correct = []
            this_p = np.array([curve_sliced_relative[0][0],curve_sliced_relative[0][1],h_largest])
            curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative[0][3:]))
            path_T_S1 = [Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3])]
            this_weld_v = []
            all_dh=[]

            for bpi in range(0,len(weld_bp)):
                if forward_flag:
                    if weld_bp[bpi][0]>curve_sliced_relative[0][0]:
                        continue
                    if weld_bp[bpi][0]<curve_sliced_relative[-1][0]:
                        break
                else:
                    if weld_bp[bpi][0]<curve_sliced_relative[0][0]:
                        continue
                    if weld_bp[bpi][0]>curve_sliced_relative[-1][0]:
                        break
                if bpi==0:
                    dh = min(h_target-weld_bp[bpi][1],2.5)
                else:
                    dh = min(h_target-weld_bp[bpi-1][1],2.5)
                all_dh.append(dh)
                a=0.006477
                b=-0.2362
                c=3.339-dh
                v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
                this_weld_v.append(v)

                this_p = np.array([weld_bp[bpi][0],curve_sliced_relative[0][1],h_largest])
                curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_correct[0][3:]))
                path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
            
            dh = min(h_target-weld_bp[bpi-1][1],2.5)
            all_dh.append(dh)
            a=0.006477
            b=-0.2362
            c=3.339-dh
            v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
            this_weld_v.append(v)
            this_p = np.array([curve_sliced_relative[-1][0],curve_sliced_relative[0][1],h_largest])
            curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative[0][3:]))
            path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
            curve_sliced_relative=deepcopy(curve_sliced_relative_correct)

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
        print("dh:",all_dh)
        print("Nominal V:",weld_velocity[i])
        print("Correct V:",this_weld_v)
        print(curve_sliced_relative)
        print(len(path_T))
        print(len(curve_sliced_relative))
        
        # 2. Scanning parameters
        ### scan parameters
        scan_speed=10 # scanning speed (mm/sec)
        scan_stand_off_d = 80 ## mm
        Rz_angle = np.radians(0) # point direction w.r.t welds
        Ry_angle = np.radians(0) # rotate in y a bit
        bounds_theta = np.radians(10) ## circular motion at start and end
        all_scan_angle = np.radians([0]) ## scan angle
        q_init_table=np.radians([-15,200]) ## init table
        save_output_points = True
        ### scanning path module
        spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
        mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
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
        # print(scan_p[0],scan_R[0])
        # print(robot_scan.fwd(q_out1[0]))
        # print(robot_scan.fwd(np.zeros(6)))
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
        
        robot_client.execute_motion_program(mp)

        input("Press Enter and start welding.")
        mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
        mp.MoveL(np.degrees(path_q[1]), 10, 0)
        mp.setArc(select, int(this_job_number))
        for bpi in range(len(this_weld_v)):
            if bpi!=len(this_weld_v)-1:
                mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi])
            else:
                mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi],0)
        mp.setArc(False)
        mp.MoveL(np.degrees(path_q[-1]), 10, 0)
        clean_weld_record()
        rob_stamps,rob_js_exe,_,_= robot_client.execute_motion_program(mp)

        if save_weld_record:
            Path(data_dir).mkdir(exist_ok=True)
            layer_data_dir=data_dir+'layer_'+str(i)+'/'
            Path(layer_data_dir).mkdir(exist_ok=True)
            np.savetxt(layer_data_dir + 'welding.csv',
                        np.array([weld_timestamp, weld_voltage, weld_current, weld_feedrate, weld_energy]).T, delimiter=',',
                        header='timestamp,voltage,current,feedrate,energy', comments='')

        # move home
        input("Press Enter to Move Home")
        mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
        mp.MoveJ(np.zeros(6), 3, 0)
        robot_client.execute_motion_program(mp)
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

        ## motion start
        mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
        # calibration motion
        target2=['MOVJ',np.degrees(q_bp2[1][0]),s2_all[0]]
        mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
        # routine motion
        for path_i in range(2,len(q_bp1)-1):
            target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
            mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
        target2=['MOVJ',np.degrees(q_bp2[-1][0]),s2_all[-1]]
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

    ### for scan testing
    # out_scan_dir=data_dir='../data/wall_weld_test/weld_scan_2023_05_31_13_05_55/layer_2/scans/'
    # q_init_table=np.radians([-15,200])
    # h_largest=12
    # q_out_exe=np.loadtxt(out_scan_dir + 'scan_js_exe.csv',delimiter=',')
    # robot_stamps=np.loadtxt(out_scan_dir + 'robot_stamps.csv',delimiter=',')
    # with open(out_scan_dir + 'mti_scans.pickle', 'rb') as file:
    #     mti_recording=pickle.load(file)
    ########################

    #### scanning process: processing point cloud and get h
    try:
        if forward_flag:
            curve_x_start = deepcopy(curve_sliced_relative[0][0])
            curve_x_end = deepcopy(curve_sliced_relative[-1][0])
        else:
            curve_x_start = deepcopy(curve_sliced_relative[-1][0])
            curve_x_end = deepcopy(curve_sliced_relative[0][0])
    except:
        curve_x_start=43
        curve_x_end=-41
    z_height_start=h_largest-1
    crop_extend=10
    crop_min=(curve_x_end-crop_extend,-30,-10)
    crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
    crop_h_min=(curve_x_end-crop_extend,-20,-10)
    crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
    print(crop_min)
    print(crop_max)
    scan_process = ScanProcess(robot_scan,positioner)
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=60,std_ratio=0.85,\
                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=False)
    profile_height = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max)
    save_output_points=True
    if save_output_points:
        o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
        np.save(out_scan_dir+'height_profile.npy',profile_height)
    visualize_pcd([pcd])
    plt.scatter(profile_height[:,0],profile_height[:,1])
    plt.show()

    if np.mean(profile_height[:,1])>final_height:
        break

    forward_flag=not forward_flag

print("Welding End!!")