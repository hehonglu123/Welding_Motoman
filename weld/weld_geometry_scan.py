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
from weldCorrectionStrategy import *
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

zero_config=np.zeros(6)
# 0. robots.
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

final_height=50
final_h_std_thres=0.48
weld_z_height=[0,6,8] # two base layer height to first top layer
weld_z_height=np.append(weld_z_height,np.arange(weld_z_height[-1],final_height,1)+1)
# job_number=[115,115]
job_number=[215,215]
job_number=np.append(job_number,np.ones(len(weld_z_height)-2)*206)
print(weld_z_height)
print(job_number)

weld_velocity=[5,5]
weld_v=18
for i in range(len(weld_z_height)-2):
    weld_velocity.append(weld_v)
    # if weld_v==weld_velocity[-2]:
    #     weld_v+=2

print(weld_velocity)
# exit()
save_weld_record=True

# exit()

dataset='cup/'
sliced_alg='circular_slice_shifted/'
curve_data_dir = '../data/'+dataset+sliced_alg

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir=curve_data_dir+'../weld_scan_'+formatted_time+'/'

# ## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
# # weld state logging
# sub = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
# obj = sub.GetDefaultClientWait(3)  # connect, timeout=30s
# welder_state_sub = sub.SubscribeWire("welder_state")
# welder_state_sub.WireValueChanged += wire_cb
# # MTI connect to RR
# mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
# mti_client.setExposureTime("25")
###################################
base_layer = False
profile_height=None
Transz0_H=np.eye(4)
curve_sliced_relative=None
last_mean_h = 0

for i in range(0,len(weld_z_height)):
    cycle_st = time.time()
    print("Layer:",i)
    #### welding
    weld_st = time.time()
    if i>=0 and False:
        weld_plan_st = time.time()
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
        print(path_T[0])
        print(curve_sliced_relative)
        R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_p.R)

        #### Correction ####
        # TODO: Add fitering if near threshold
        h_largest=this_z_height
        if (i<=2):
            if (last_mean_h == 0) and (profile_height is not None):
                last_mean_h=np.mean(profile_height[:,1])
            if profile_height is None:
                if i!=0:
                    data_dir='../data/wall_weld_test/weld_scan_2023_06_13_10_56_01/'
                    print("Using data:",data_dir)
                    last_profile_height=np.load(data_dir+'layer_16/scans/height_profile.npy')
                    last_mean_h=np.mean(last_profile_height[:,1])
                    profile_height=np.load(data_dir+'layer_17/scans/height_profile.npy')

            if (profile_height is not None) and (i>2):
                mean_h = np.mean(profile_height[:,1])
                dh_last_layer = mean_h-last_mean_h
                h_target = mean_h+dh_last_layer

                dh_direction = np.array([0,0,h_target-curve_sliced_relative[0][2]])
                dh_direction_R1 = T_R1Base_S1TCP[:3,:3]@dh_direction

                for curve_i in range(len(curve_sliced_relative)):
                    curve_sliced_relative[curve_i][2]=h_target
                
                for path_i in range(len(path_T)):
                    path_T[path_i].p=path_T[path_i].p+dh_direction_R1

                last_mean_h=mean_h

        else: # start correction from 2nd top layer
            
            if profile_height is None:
                data_dir='../data/wall_weld_test/weld_scan_2023_06_16_11_33_46/'
                print("Using data:",data_dir)
                last_profile_height=np.load(data_dir+'layer_5/scans/height_profile.npy')
                last_mean_h=np.mean(last_profile_height[:,1])
                profile_height=np.load(data_dir+'layer_6/scans/height_profile.npy')

            ## parameters
            noise_h_thres = 3
            num_l=40
            input_dh=1.7
            
            # min_v=5
            # max_v=30
            # h_std_thres=0.48

            min_v=-1
            max_v=1000
            h_std_thres=-1

            nominal_v=18
            curve_sliced_relative,path_T_S1,this_weld_v,all_dh,last_mean_h=\
                strategy_3(profile_height,input_dh,curve_sliced_relative,R_S1TCP,num_l,noise_h_thres=noise_h_thres,\
                           min_v=min_v,max_v=max_v,h_std_thres=h_std_thres,nominal_v=nominal_v)
            
            h_largest = np.max(profile_height[:,1])

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
        print("curve_sliced_relative:",curve_sliced_relative)
        print(path_T[0])
        print(len(path_T))
        print(len(curve_sliced_relative))

        print("Weld Plan time:",time.time()-weld_plan_st)

        ######################################################
        ########### Do welding #############
        weld_motion_st = time.time()
        to_start_speed=8
        path_q = []
        for tcp_T in path_T:
            path_q.append(robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])
        
        # input("Press Enter and move to weld starting point.")
        mp = MotionProgram(ROBOT_CHOICE='RB1', pulse2deg=robot_weld.pulse2deg)
        mp.MoveJ(np.degrees(path_q[0]), to_start_speed, 0)
        
        robot_client.execute_motion_program(mp)

        print("Weld to start time:",time.time()-weld_motion_st)
        
        weld_motion_weld_st = time.time()
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
            np.savetxt(layer_data_dir + 'weld_js_exe.csv',rob_js_exe,delimiter=',')
            np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',rob_stamps,delimiter=',')
        
        print("Weld actual weld time:",time.time()-weld_motion_weld_st)
        weld_to_home_st = time.time()

        # move home
        # input("Press Enter to Move Home")
        mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
        mp.MoveJ(np.zeros(6), to_start_speed, 0)
        robot_client.execute_motion_program(mp)

        print("Weld to home time:",time.time()-weld_to_home_st)
        print("Weld Motion Time:",time.time()-weld_motion_st)
        ######################################################

        print("Weld Time:",time.time()-weld_st)
    #### scanning
    if True:
        scan_st = time.time()
        if curve_sliced_relative is None:
            curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(1)+'_'+str(0)+'.csv',delimiter=',')
            
            # print(curve_sliced_relative)
            # dlam = np.linalg.norm(np.diff(curve_sliced_relative[:,:3],axis=0),axis=1)
            # print(dlam[:10])
            # plt.plot(dlam,'-o')
            # plt.show()
            # exit()

            data_dir=curve_data_dir+'../weld_scan_2023_06_13_10_56_01/'
            layer_data_dir=data_dir+'layer_'+str(i)+'/'
            
        scan_plan_st = time.time()
        # 2. Scanning parameters
        ### scan parameters
        scan_speed=10 # scanning speed (mm/sec)
        scan_stand_off_d = 95 ## mm
        Rz_angle = np.radians(0) # point direction w.r.t welds
        Ry_angle = np.radians(0) # rotate in y a bit
        bounds_theta = np.radians(1) ## circular motion at start and end
        all_scan_angle = np.radians([0]) ## scan angle
        q_init_table=np.radians([-15,200]) ## init table
        save_output_points = True
        ### scanning path module
        spg = ScanPathGen(robot_scan,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
        mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
                    [0.,0.,-1.]])
        # generate scan path
        scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                            solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
        # generate motion program
        q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

        # print(np.degrees(q_out1[:10]))
        # print(np.degrees(q_out1[-10:]))
        # plt.plot(np.degrees(q_out1))
        # plt.legend(['J1','J2','J3','J4','J5','J6'])
        # plt.show()
        # print(scan_p[0],scan_R[0])
        # print(robot_scan.fwd(q_out1[0]))
        # print(robot_scan.fwd(np.zeros(6)))
        # exit()
        #######################################

        print("Scan plan time:",time.time()-scan_plan_st)

        scan_motion_st = time.time()
        ######## scanning motion #########
        ### execute motion ###
        # input("Press Enter and move to scanning startint point")

        ## move to start
        to_start_speed=7
        mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
        target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
        mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
        robot_client.execute_motion_program(mp)

        print("Scan to home time:",time.time()-scan_motion_st)

        input("Press Enter to start moving and scanning")

        scan_motion_scan_st = time.time()

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

        print("Scan motion scan time:",time.time()-scan_motion_scan_st)
        
        scan_to_home_st = time.time()
        mti_recording=np.array(mti_recording)
        q_out_exe=joint_recording

        # input("Press Enter to Move Home")
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
            np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
            with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
                pickle.dump(mti_recording, file)
            print('Total scans:',len(mti_recording))
        
        print("Scan to home:",time.time()-scan_to_home_st)
        print("Scan motion time:",time.time()-scan_motion_st)
        
        print("Scan Time:",time.time()-scan_st)

    ### for scan testing
    # out_scan_dir=data_dir='../data/wall_weld_test/weld_scan_2023_06_06_12_43_57/layer_15/scans/'
    # q_init_table=np.radians([-15,200])
    # h_largest=20
    # q_out_exe=np.loadtxt(out_scan_dir + 'scan_js_exe.csv',delimiter=',')
    # robot_stamps=np.loadtxt(out_scan_dir + 'scan_robot_stamps.csv',delimiter=',')
    # with open(out_scan_dir + 'mti_scans.pickle', 'rb') as file:
    #     mti_recording=pickle.load(file)
    ########################

    recon_3d_st = time.time()
    #### scanning process: processing point cloud and get h
    crop_extend=10
    crop_min=tuple(np.min(curve_sliced_relative[:][:3],axis=0)-crop_extend)
    crop_max=tuple(np.max(curve_sliced_relative[:][:3],axis=0)+crop_extend)
    scan_process = ScanProcess(robot_scan,positioner)
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
    print("3D Reconstruction Time:",time.time()-recon_3d_st)
    get_h_st = time.time()
    profile_height = scan_process.pcd2dh(pcd,curve_sliced_relative)
    
    print("Get Height Time:",time.time()-get_h_st)

    save_output_points=True
    if save_output_points:
        o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
        np.save(out_scan_dir+'height_profile.npy',profile_height)
    # visualize_pcd([pcd])
    # plt.scatter(profile_height[:,0],profile_height[:,1])
    # plt.show()
    # exit()

    if np.mean(profile_height[:,1])>final_height and np.std(profile_height[:,1])<final_h_std_thres:
        break

    print("Print Cycle Time:",time.time()-cycle_st)

print("Welding End!!")