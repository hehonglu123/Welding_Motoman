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
from weldRRSensor import *
from WeldSend import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

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
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti_backup0719.csv',\
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
# final_h_std_thres=0.48
final_h_std_thres=999999999
weld_z_height=[0,6,7] # two base layer height to first top layer
weld_z_height=np.append(weld_z_height,np.arange(weld_z_height[-1],final_height,1)+1)
# job_number=[115,115]
job_number=[215,215]
job_number=np.append(job_number,np.ones(len(weld_z_height)-2)*200) # 100 ipm
# job_number=np.append(job_number,np.ones(len(weld_z_height)-2)*206) # 160 ipm
# job_number=np.append(job_number,np.ones(len(weld_z_height)-2)*212) # 220 ipm
print(weld_z_height)
print(job_number)

ipm_mode=100
weld_velocity=[5,5]
weld_v=5
print("input dh:",v2dh_loglog(weld_v,ipm_mode))
for i in range(len(weld_z_height)-2):
    weld_velocity.append(weld_v)
    # if weld_v==weld_velocity[-2]:
    #     weld_v+=2
# print(weld_velocity)
# exit()

to_start_speed=4
to_home_speed=5

save_weld_record=True

start_correction_layer=2
# start_correction_layer=99999999

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_scan_'+formatted_time+'/'

### read cmd
use_previous_cmd=False
cmd_dir = '../data/wall_weld_test/'+'moveL_100_weld_scan_2023_08_02_15_17_25/'

## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser,microphone_service=mic_ser)

### test sensor (camera, microphone)
# print("Test 3 Sec.")
# rr_sensors.test_all_sensors()
# print(len(rr_sensors.ir_recording))
# exit()
###############

# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")
###################################
base_layer = True
profile_height=None
Transz0_H=None
# Transz0_H=np.array([[ 9.99997540e-01,  2.06703673e-06, -2.21825071e-03, -3.46701381e-03],
#  [ 2.06703673e-06,  9.99998263e-01,  1.86365986e-03,  2.91280622e-03],
#  [ 2.21825071e-03, -1.86365986e-03,  9.99995803e-01,  1.56294293e+00],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
curve_sliced_relative=None
last_mean_h = 0

# ir pose
r2_ir_q = np.radians([43.3469,36.0996,-63.0900,142.5838,-83.0429,-96.0737])
r2_mid = np.radians([43.7851,20,-10,0,0,0])
# r2_ir_q = np.zeros(6)

weld_arcon=False

end_layer = len(weld_z_height)
if use_previous_cmd:
    end_layer = len(glob.glob(cmd_dir+'layer_*'))

input("Start?")
# move robot to ready position
ws.jog_dual(robot_scan,positioner,[r2_mid,r2_ir_q],np.radians([-15,180]),to_start_speed)

for i in range(0,end_layer):
    cycle_st = time.time()
    print("==================================")
    print("Layer:",i)
    if i%2==0:
        forward_flag = True
    else:
        forward_flag = False
    #### welding
    weld_st = time.time()
    if i>=0 and True:
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
        if (i<start_correction_layer):
            if (last_mean_h == 0) and (profile_height is not None):
                last_mean_h=np.mean(profile_height[:,1])
            if profile_height is None:
                if i!=0:
                    data_dir='../data/wall_weld_test/weld_scan_2023_07_05_16_58_36/'
                    print("Using data:",data_dir)
                    last_profile_height=np.load(data_dir+'layer_2/scans/height_profile.npy')
                    last_mean_h=np.mean(last_profile_height[:,1])
                    profile_height=np.load(data_dir+'layer_3/scans/height_profile.npy')

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
                data_dir='../data/wall_weld_test/weld_scan_2023_08_02_15_17_25/'
                print("Using data:",data_dir)
                last_profile_height=np.load(data_dir+'layer_21/scans/height_profile.npy')
                last_mean_h=np.mean(last_profile_height[:,1])
                profile_height=np.load(data_dir+'layer_22/scans/height_profile.npy')
                

            ## parameters
            # noise_h_thres = 3
            # peak_threshold=0.25
            # flat_threshold=2.5
            # correct_thres = 1.4 # mm
            # patch_nb = 2 # 2*0.1
            # start_ramp_ratio = 0.67
            # end_ramp_ratio = 0.33
            #############
            # curve_sliced_relative,path_T_S1,this_weld_v,all_dh,last_mean_h=\
            #     strategy_2(profile_height,last_mean_h,forward_flag,curve_sliced_relative,R_S1TCP,this_weld_v[0],\
            #             noise_h_thres=noise_h_thres,peak_threshold=peak_threshold,flat_threshold=flat_threshold,\
            #             correct_thres=correct_thres,patch_nb=patch_nb,\
            #             start_ramp_ratio=start_ramp_ratio,end_ramp_ratio=end_ramp_ratio)

            ## parameters
            noise_h_thres = 3
            num_l=40
            # input_dh=1.1624881529394444
            # input_dh=1.4018280504260527
            input_dh=v2dh_loglog(weld_v,ipm_mode)
            
            # min_v=10
            # max_v=75
            # h_std_thres=0.5

            min_v=-1
            max_v=1000
            h_std_thres=-1

            nominal_v=weld_v
            curve_sliced_relative,path_T_S1,this_weld_v,all_dh,last_mean_h=\
                strategy_3(profile_height,input_dh,curve_sliced_relative,R_S1TCP,num_l,noise_h_thres=noise_h_thres,\
                           min_v=min_v,max_v=max_v,h_std_thres=h_std_thres,nominal_v=nominal_v,ipm_mode=ipm_mode)
            
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
        
        path_q = []
        for tcp_T in path_T:
            path_q.append(robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])

        if use_previous_cmd:
            cmd_layer_data_dir=cmd_dir+'layer_'+str(i)+'/'
            breakpoints,primitives,q_bp,this_weld_v = ws.load_weld_cmd(cmd_layer_data_dir+'command1.csv')
            this_weld_v=this_weld_v[1:]
            path_q = []
            for q in q_bp:
                path_q.append(q[0])
            start_T = robot_weld.fwd(path_q[0])
            start_T.p = start_T.p+np.array([0,0,10])
            path_q.insert(0,robot_weld.inv(start_T.p,start_T.R,path_q[0])[0])
            end_T = robot_weld.fwd(path_q[-1])
            end_T.p = end_T.p+np.array([0,0,10])
            path_q.append(robot_weld.inv(end_T.p,end_T.R,path_q[-1])[0])
            print("Use cmd:",cmd_layer_data_dir)
        
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
        
        # input("Press Enter and move to weld starting point.")
        ws.jog_single(robot_weld,path_q[0],to_start_speed)
        
        weld_motion_weld_st = time.time()
        # input("Press Enter and start welding.")
        # mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
        # mp.MoveL(np.degrees(path_q[1]), 10, 0)
        # mp.setArc(select, int(this_job_number))
        # for bpi in range(len(this_weld_v)):
        #     if bpi!=len(this_weld_v)-1:
        #         mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi])
        #     else:
        #         mp.MoveL(np.degrees(path_q[bpi+2]), this_weld_v[bpi],0)
        # mp.setArc(False)
        # mp.MoveL(np.degrees(path_q[-1]), 10, 0)

        primitives=[]
        for bpi in range(len(this_weld_v)+1):
            primitives.append('movel')

        rr_sensors.start_all_sensors()
        rob_stamps,rob_js_exe,_,_=ws.weld_segment_single(primitives,robot_weld,path_q[1:-1],np.append(10,this_weld_v),cond_all=[int(this_job_number)],arc=weld_arcon)
        rr_sensors.stop_all_sensors()

        if save_weld_record:
            Path(data_dir).mkdir(exist_ok=True)
            layer_data_dir=data_dir+'layer_'+str(i)+'/'
            Path(layer_data_dir).mkdir(exist_ok=True)
            # save cmd
            q_bp=[]
            for q in path_q[1:-1]:
                q_bp.append([np.array(q)])
            ws.save_weld_cmd(layer_data_dir+'command1.csv',np.arange(len(primitives)),primitives,q_bp,np.append(10,this_weld_v))
            # save weld record
            np.savetxt(layer_data_dir + 'weld_js_exe.csv',rob_js_exe,delimiter=',')
            np.savetxt(layer_data_dir + 'weld_robot_stamps.csv',rob_stamps,delimiter=',')
            rr_sensors.save_all_sensors(layer_data_dir)
        
        print("Weld actual weld time:",time.time()-weld_motion_weld_st)
        weld_to_home_st = time.time()

        # move home
        # input("Press Enter to Move Home")
        print("Weld to home time:",time.time()-weld_to_home_st)
        ######################################################

        print("Weld Time:",time.time()-weld_st)
    # exit()
    ws.jog_single(robot_weld,np.zeros(6),to_home_speed)
    #### scanning
    if True:
        scan_st = time.time()
        if curve_sliced_relative is None:
            data_dir='../data/wall_weld_test/weld_scan_2023_08_02_17_07_02/'
            last_profile_height=np.load('../data/wall_weld_test/weld_scan_2023_08_02_17_07_02/layer_17/scans/height_profile.npy')
            last_mean_h=np.mean(last_profile_height[:,1])
            h_largest=np.max(last_profile_height[:,1])
            layer_data_dir=data_dir+'layer_'+str(i)+'/'
            curve_sliced_relative=[np.array([ 3.30446707e+01,  1.72700000e+00,  4.36704154e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01]), np.array([-3.19476273e+01,  1.72700000e+00,  4.36704154e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01])]
            input("Start Scanning")

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
        if forward_flag:
            scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                            solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
        else:
            scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative[::-1]],[0],all_scan_angle,\
                            solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
        # generate motion program
        q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)
        #######################################

        print("Scan plan time:",time.time()-scan_plan_st)

        scan_motion_st = time.time()
        ######## scanning motion #########
        ### execute motion ###
        robot_client=MotionProgramExecClient()
        # input("Press Enter and move to scanning startint point")

        ## move to start
        ws.jog_dual(robot_scan,positioner,[r2_mid,q_bp1[0][0]],q_bp2[0][0],to_start_speed)

        # input("Press Enter to start moving and scanning")
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

        ws.client.execute_motion_program_nonblocking(mp)
        ###streaming
        ws.client.StartStreaming()
        start_time=time.time()
        state_flag=0
        joint_recording=[]
        robot_stamps=[]
        mti_recording=[]
        r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
        while True:
            if state_flag & 0x08 == 0 and time.time()-start_time>1.:
                break
            res, data = ws.client.receive_from_robot(0.01)
            if res:
                joint_angle=np.radians(np.divide(np.array(data[26:34]),r_pulse2deg))
                state_flag=data[16]
                joint_recording.append(joint_angle)
                timestamp=data[0]+data[1]*1e-9
                robot_stamps.append(timestamp)
                ###MTI scans YZ point from tool frame
                mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
        ws.client.servoMH(False)

        print("Scan motion scan time:",time.time()-scan_motion_scan_st)
        
        scan_to_home_st = time.time()
        mti_recording=np.array(mti_recording)
        q_out_exe=joint_recording

        # input("Press Enter to Move Home")
        # move robot to home
        # q2=np.zeros(6)
        # q2[0]=90
        q2=deepcopy(r2_ir_q)
        q3=np.radians([-15,180])
        ws.jog_dual(robot_scan,positioner,[r2_mid,r2_ir_q],q3,to_home_speed)
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
    # Transz0_H=np.array([[ 9.99974559e-01, -7.29664987e-06, -7.13309345e-03, -1.06461758e-02],
    #                     [-7.29664987e-06,  9.99997907e-01, -2.04583032e-03, -3.05341146e-03],
    #                     [ 7.13309345e-03,  2.04583032e-03,  9.99972466e-01,  1.49246365e+00],
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    z_height_start=h_largest-3
    crop_extend=10
    crop_min=(curve_x_end-crop_extend,-30,-10)
    crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
    crop_h_min=(curve_x_end-crop_extend,-20,-10)
    crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
    scan_process = ScanProcess(robot_scan,positioner)
    pcd=None
    pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
    pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                        min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
    print("3D Reconstruction:",time.time()-recon_3d_st)
    get_h_st = time.time()
    profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
    print("Transz0_H:",Transz0_H)
    
    print("Get Height:",time.time()-get_h_st)

    save_output_points=True
    if save_output_points:
        o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
        np.save(out_scan_dir+'height_profile.npy',profile_height)
    # visualize_pcd([pcd])
    plt.scatter(profile_height[:,0],profile_height[:,1])
    plt.show()
    # exit()

    if np.mean(profile_height[:,1])>final_height and np.std(profile_height[:,1])<final_h_std_thres and (not use_previous_cmd):
        break

    forward_flag=not forward_flag

    print("Print Cycle Time:",time.time()-cycle_st)

print("Welding End!!")