import sys, time
import numpy as np
import matplotlib.pyplot as plt
from dx200_motion_program_exec_client import *
sys.path.append('../../toolbox/')
from robot_def import *
from WeldSend import *
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from scan_utils import *
from scanPathGen import *
from scanProcess import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15,\
    base_marker_config_file='../config/MA2010_marker_config.yaml',tool_marker_config_file='../config/weldgun_marker_config.yaml')
robot2_mti=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv',\
    base_marker_config_file='../config/MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv',\
    base_marker_config_file='../config/D500B_marker_config.yaml',tool_marker_config_file='../config/positioner_tcp_marker_config.yaml')

#### change base H to calibrated ones ####
robot_scan_base = robot.T_base_basemarker.inv()*robot2.T_base_basemarker
robot2.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
robot2_mti.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

positioner_pose=positioner.fwd(np.radians([-15,0]))
H_positioner_pose=H_from_RT(positioner_pose.R,positioner_pose.T)



R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
x_start=1630
x_end=1700
x_all=np.linspace(x_start,x_end,5)
y_start=-780
y_end=-840
z=-260

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)

###base layers welding
for x in x_all:
    p1=np.array([x,y_start,z])
    p2=np.array([x,y_end,z])
    q_init=robot.inv(p1,R,q_seed)[0]
    q_end=robot.inv(p2,R,q_seed)[0]
    q_all=[q_init,q_end]
    v_all=[1,15]
    cond_all=[0,215]
    primitives=['movej','movel']
    ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=False)

###first layers welding
for x in x_all:
    p1=np.array([x,y_start,z+2])
    p2=np.array([x,y_end,z+2])
    q_init=robot.inv(p1,R,q_seed)[0]
    q_end=robot.inv(p2,R,q_seed)[0]
    q_all=[q_init,q_end]
    v_all=[1,15]
    cond_all=[0,215]
    primitives=['movej','movel']
    ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=False)


######## scanning ##########
for x in x_all:
    ws.jog_single(robot,np.array([-8.135922244967886741e-01,7.096733413840118354e-01,3.570605700073341549e-01,1.795958126158156976e-01,-8.661845429601626734e-01,-4.639865155930678053e-01]),v=3)
    p1_relative=(H_positioner_pose.T@positioner.base_H.T@np.array([x,y_start,z+2,1]))[:3]
    p2_relative=(H_positioner_pose.T@positioner.base_H.T@np.array([x,y_end,z+2,1]))[:3]

    curve_sliced_relative=np.linspace(p1_relative,p2_relative,100)

    scan_speed=10 # scanning speed (mm/sec)
    scan_stand_off_d = 95 ## mm
    Rz_angle = np.radians(0) # point direction w.r.t welds
    Ry_angle = np.radians(0) # rotate in y a bit
    bounds_theta = np.radians(1) ## circular motion at start and end
    all_scan_angle = np.radians([0]) ## scan angle
    q_init_table=np.radians([-15,20]) ## init table
    save_output_points = True
    ### scanning path module
    spg = ScanPathGen(robot2_mti,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
    mti_Rpath = np.array([[ -1.,0.,0.],   
                [ 0.,1.,0.],
                [0.,0.,-1.]])
    # mti_Rpath = np.eye(3)
    scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                        solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
    q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

    # to_start_speed=7
    # mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    # target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
    # mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
    # robot_client.execute_motion_program(mp)
    ws.jog_dual(robot2_mti,positioner,q_bp1[0][0],q_bp2[0][0],v=3)

    scan_motion_scan_st = time.time()

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2_mti.pulse2deg,pulse2deg_2=positioner.pulse2deg)
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
    r_pulse2deg = np.append(robot2_mti.pulse2deg,positioner.pulse2deg)
    while True:
        if state_flag & 0x08 == 0 and time.time()-start_time>1.:
            print("break")
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
    robot_stamps=np.array(robot_stamps)-robot_stamps[0]
    ws.client.servoMH(False)
    mti_recording=np.array(mti_recording)
    q_out_exe=joint_recording

    # move robot to home
    q2=np.array([-2.704468035842202411e-01,9.330144509521144380e-01,-3.376213326477142118e-01,-1.228474839331376023e+00,-1.395732731587226549e+00,2.846773448527548656e+00])
    q3=np.radians([-15,90])
    # ## move to home
    # to_home_speed=7
    # mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2_mti.pulse2deg,pulse2deg_2=positioner.pulse2deg)
    # target2=['MOVJ',q3,to_home_speed]
    # mp.MoveJ(q2, to_home_speed, 0, target2=target2)
    # robot_client.execute_motion_program(mp)
    ws.jog_dual(robot2_mti,positioner,q2,q3,v=3)
    #####################

    print("Total exe len:",len(q_out_exe))
    out_scan_dir = layer_data_dir+'scans/'
    ## save traj
    Path(out_scan_dir).mkdir(exist_ok=True)
    # save poses
    np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
    np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
    with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
        pickle.dump(mti_recording, file)
    print('Total scans:',len(mti_recording))

    curve_x_start = deepcopy(curve_sliced_relative[0][0])
    curve_x_end = deepcopy(curve_sliced_relative[-1][0])
    # Transz0_H=np.array([[ 9.99974559e-01, -7.29664987e-06, -7.13309345e-03, -1.06461758e-02],
    #                     [-7.29664987e-06,  9.99997907e-01, -2.04583032e-03, -3.05341146e-03],
    #                     [ 7.13309345e-03,  2.04583032e-03,  9.99972466e-01,  1.49246365e+00],
    #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    z_height_start=h_largest-5
    crop_extend=10
    crop_min=(curve_x_start-crop_extend,-30,-10)
    crop_max=(curve_x_end+crop_extend,30,z_height_start+30)
    crop_h_min=(curve_x_start-crop_extend,-20,-10)
    crop_h_max=(curve_x_end+crop_extend,20,z_height_start+30)

    try:
        scan_process = ScanProcess(robot2_mti,positioner)
        pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
        pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                            min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
        profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
        print("Transz0_H:",Transz0_H)

        save_output_points=True
        if save_output_points:
            o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
            np.save(out_scan_dir+'height_profile.npy',profile_height)
        # visualize_pcd([pcd])
        plt.scatter(profile_height[:,0],profile_height[:,1])
        plt.show()
        h_largest=np.max(profile_height[:,1])
        h_mean=np.mean(profile_height[:,1])
        print("H largest:",h_largest)
        print("H mean:",h_mean)
    except Exception as e:
        print(e)
        h_largest = curve_sliced_relative[0][2]+1.8