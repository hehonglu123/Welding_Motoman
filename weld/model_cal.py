from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from utils import *
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weld_dh2v import *
from scipy.optimize import curve_fit
from weld_dh2v import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import open3d as o3d
import math
from scipy.optimize import curve_fit
from scipy import stats
# def moving_average(a,w=3):
    
#     if w%2==0:
#         w+=1

#     ## add padding
#     padd_n = int((w-1)/2)
#     a = np.append(np.ones(padd_n)*a[0],a)
#     a = np.append(a,np.ones(padd_n)*a[-1])
    
#     ret = np.cumsum(a, dtype=float)
#     ret[w:] = ret[w:] - ret[:-w]
#     return ret[w - 1:] / w
def power_law(x, a, b):
    return np.log(a) + b * np.log(x)
def func(x, a, b):
    return a * x**b
def is_odd(num):
    return num % 2 != 0
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

path_R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
R_S1TCP = np.matmul(T_S1TCP_R1Base[:3,:3],path_R)

build_height_profile=False
plot_correction=True
# show_layer = [12]
show_layer = [12,29,30]

x_lower = -99999
x_upper = 999999

# datasets=['baseline','full_test']
datasets=['full_test']
datasets_h_mean={}
datasets_h_std={}
for dataset in datasets:

    if dataset=='baseline':
        data_dir = '../data/wall_weld_test/baseline_weld_scan_2023_06_06_15_28_31/'
    elif dataset=='full_test':
        data_dir = '../data/wall_weld_test/ER4043_model_130ipm_2023_10_08_07_52_48/'
    print('data_dir',data_dir)
    forward_flag=False
    all_profile_height=[]
    all_correction_layer=[]

    all_h_mean=[]
    all_h_std=[]
    for i in range(0,50):
        try:
            weld_dir=data_dir+'layer_'+str(i)+'/'
            weld_q=np.loadtxt(weld_dir+'weld_js_exe.csv',delimiter=',')
            weld_stamp=np.loadtxt(weld_dir+'weld_robot_stamps.csv',delimiter=',')
            scan_dir=weld_dir+'scans/'
            pcd = o3d.io.read_point_cloud(scan_dir+'processed_pcd.pcd')
            profile_height = np.load(scan_dir+'height_profile.npy')
            q_out_exe=np.loadtxt(scan_dir+'scan_js_exe.csv',delimiter=',')
            robot_stamps=np.loadtxt(scan_dir+'scan_robot_stamps.csv',delimiter=',')
            with open(scan_dir+'mti_scans.pickle', 'rb') as file:
                mti_recording=pickle.load(file)
            
            # q_out_exe=np.loadtxt(data_dir +'scan_js_exe.csv',delimiter=',')
            # robot_stamps=np.loadtxt(data_dir +'robot_stamps.csv',delimiter=',')
            # with open(data_dir +'mti_scans.pickle', 'rb') as file:
            #     mti_recording=pickle.load(file)
        except:
            continue
        print("Layer",i)
        print("Forward:",not forward_flag)

        if build_height_profile:
            curve_x_start=43
            curve_x_end=-41
            crop_extend=10
            z_height_start=20
            scan_process = ScanProcess(robot_scan,positioner)
            crop_min=(curve_x_end-crop_extend,-30,-10)
            crop_max=(curve_x_start+crop_extend,30,z_height_start+30)
            crop_h_min=(curve_x_end-crop_extend,-20,-10)
            crop_h_max=(curve_x_start+crop_extend,20,z_height_start+30)
            q_init_table=np.radians([-15,200])
            pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
            # visualize_pcd([pcd])
            pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
            # visualize_pcd([pcd])
            profile_height = scan_process.pcd2height(deepcopy(pcd),-1)

        ### ignore x smaller and larger
        profile_height=np.delete(profile_height,np.where(profile_height[:,0]>x_upper),axis=0)
        profile_height=np.delete(profile_height,np.where(profile_height[:,0]<x_lower),axis=0)

        all_profile_height.append(profile_height)

        h_std_thres=0.48
        h_std = np.std(profile_height[:,1])
        if i>2 and h_std>h_std_thres:
            all_correction_layer.append(i)

        if (plot_correction and (i in show_layer)):
            # weld path
            if forward_flag:
                curve_sliced_relative=[np.array([ 3.30445152e+01,  1.72700000e+00,  3.31751393e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01]), np.array([-3.19477829e+01,  1.72700000e+00,  3.31751393e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01])]
            else:
                curve_sliced_relative=[np.array([-3.19477829e+01,  1.72700000e+00,  3.31751393e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01]), np.array([ 3.30445152e+01,  1.72700000e+00,  3.31751393e+01,  1.55554573e-04,
       -6.31394918e-20, -9.99881509e-01])]

            ## parameters
            noise_h_thres = 3
            min_v=5
            max_v=30
            h_std_thres=h_std_thres
            nominal_v=18
            input_dh=1.7
            num_l=40
            ############

            ### delete noise
            mean_h = np.mean(profile_height[:,1])
            profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h>noise_h_thres),axis=0)
            profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h<-noise_h_thres),axis=0)
            ###
            mean_h = np.mean(profile_height[:,1])
            h_largest = np.max(profile_height[:,1])
            # 3. h_target = mean_h + designated dh value
            h_target = mean_h+input_dh

            h_std = np.std(profile_height[:,1])
            print("H STD:",h_std)
            if h_std<=h_std_thres:
                print("H std smaller than threshold.")
                
                nominal_dh = v2dh_loglog(nominal_v,mode=160)
                print("Change target dh to:",nominal_dh)
                h_target = mean_h+nominal_dh
                
                curve_sliced_relative_correct = []
                path_T_S1 = []
                this_weld_v = []
                all_dh=[]
                for curve_i in range(len(curve_sliced_relative)):
                    this_p = np.array([curve_sliced_relative[curve_i][0],curve_sliced_relative[curve_i][1],h_target])
                    curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative[curve_i][3:]))
                    path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
                    this_weld_v.append(nominal_v)
                    all_dh.append(nominal_dh)
                this_weld_v.pop()
                all_dh.pop()

                # plt.scatter(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]))
                # plt.show()

                all_profile=[profile_height]

            else:
                # chop curve
                curve_sliced_relative_chop = np.linspace(curve_sliced_relative[0],curve_sliced_relative[-1],num_l+1)

                # find v  
                # new curve in positioner frame
                curve_sliced_relative_correct = []
                this_p = np.array([curve_sliced_relative_chop[0][0],curve_sliced_relative_chop[0][1],h_target])
                curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_chop[0][3:]))
                path_T_S1 = [Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3])]
                this_weld_v = []
                seg_mean_h = []
                all_dh= []
                all_profile=[]
                for l in range(1,num_l+1):
                    # path
                    this_p = np.array([curve_sliced_relative_chop[l][0],curve_sliced_relative_chop[l][1],h_target])
                    curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_chop[l][3:]))
                    path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))

                    # velocity
                    min_x = min(curve_sliced_relative_chop[l-1][0],curve_sliced_relative_chop[l][0])
                    max_x = max(curve_sliced_relative_chop[l-1][0],curve_sliced_relative_chop[l][0])

                    this_profile = deepcopy(profile_height[np.where(profile_height[:,0]>=min_x)[0]])
                    this_profile = this_profile[np.where(max_x>=this_profile[:,0])[0]]


                    all_profile.append(this_profile)
                    this_mean_h=np.mean(this_profile[:,1])
                    ## using model to get the velocity
                    this_dh = h_target-this_mean_h
                    this_dh=max(0.01,this_dh) # to prevent inf or nan

                    this_v = dh2v_loglog(this_dh,mode=160)
                    this_v = min(max(min_v,this_v),max_v)

                    this_weld_v.append(this_v)
                    all_dh.append(this_dh)
                    seg_mean_h.append(this_mean_h)
                
                # print("Mean H:",mean_h)
                # print("Target H:",h_target)
                # # print("Seg mean h:",seg_mean_h)
                # print("dh:",all_dh)
                # print("v:",this_weld_v)

                # visualize_pcd([pcd])
                # plt.scatter(profile_height[:,0],profile_height[:,1]-np.mean(profile_height[:,1]))
                # for p in all_profile:
                #     plt.scatter(p[:,0],p[:,1]-np.mean(profile_height[:,1]))
                # plt.show()

            ### plot velocity and actual velocity
            try:
                next_weld_dir=data_dir+'layer_'+str(i+1)+'/'
                next_weld_q=np.loadtxt(next_weld_dir+'weld_js_exe.csv',delimiter=',')
                next_weld_stamp=np.loadtxt(next_weld_dir+'weld_robot_stamps.csv',delimiter=',')
                next_scan_dir=next_weld_dir+'scans/'
                next_profile_height = np.load(next_scan_dir+'height_profile.npy')
                
                robot_p_S1TCP=[]
                for q in next_weld_q:
                    r_tcp=robot_weld.fwd(q[:6])
                    r_S1TCP=Transform(T_S1TCP_R1Base[:3,:3],T_S1TCP_R1Base[:3,-1])*r_tcp
                    robot_p_S1TCP.append(r_S1TCP.p)
                robot_p_S1TCP=np.array(robot_p_S1TCP)

                # start_idx=0
                # end_idx=-2
                plan_x = np.array(curve_sliced_relative)[:,0]
                plan_y = np.array(curve_sliced_relative)[:,1]
                # plt.plot(plan_x,plan_y)
                # plt.plot(robot_p_S1TCP[:,0],robot_p_S1TCP[:,1])
                # plt.show()
                robot_v_S1TCP=np.linalg.norm(np.diff(robot_p_S1TCP,axis=0),2,1)/np.diff(next_weld_stamp)
                robot_v_S1TCP=np.append(robot_v_S1TCP[0],robot_v_S1TCP)
                robot_v_S1TCP=moving_average(robot_v_S1TCP,padding=True)
                robot_a_S1TCP=np.gradient(robot_v_S1TCP)/np.gradient(next_weld_stamp)
                robot_a_S1TCP=moving_average(robot_a_S1TCP,padding=True)

                start_idx=300
                end_idx=np.where(robot_v_S1TCP==0)[0][-1]
                robot_p_S1TCP=robot_p_S1TCP[start_idx:end_idx+1]
                next_weld_stamp=next_weld_stamp[start_idx:end_idx+1]
                robot_v_S1TCP=robot_v_S1TCP[start_idx:end_idx+1]
                robot_a_S1TCP=robot_a_S1TCP[start_idx:end_idx+1]

                # plt.plot(robot_v_S1TCP)
                # plt.plot(robot_p_S1TCP[:,0],robot_v_S1TCP)
                # plt.show()

                dh_in_layer = []
                for curve_i in range(len(next_profile_height)):
                    px=next_profile_height[curve_i,0]
                    this_l_id=np.argmin(np.fabs(profile_height[:,0]-px))
                    dh_in_layer.append([px,next_profile_height[curve_i,1]-profile_height[this_l_id,1]])
                dh_in_layer=np.array(dh_in_layer)

                motion_dh = []
                for curve_i in range(len(robot_p_S1TCP)):
                    px=robot_p_S1TCP[curve_i,0]
                    this_l_id=np.argmin(np.fabs(profile_height[:,0]-px))
                    motion_dh.append([px,robot_p_S1TCP[curve_i,2]-profile_height[this_l_id,1]])
                motion_dh=np.array(motion_dh)

                all_profile_plot=[]
                all_profile_v_plot=[]
                for seg_i in range(len(all_profile)):
                    p=deepcopy(all_profile[seg_i])
                    all_profile_plot.extend(all_profile[seg_i])
                    all_profile_v_plot.extend(np.array([p[:,0],np.repeat(this_weld_v[seg_i],len(p[:,0]))]).T)
                all_profile_plot=np.array(all_profile_plot)
                all_profile_v_plot=np.array(all_profile_v_plot)
                
                # fig, ax1 = plt.subplots()
                # ax2 = ax1.twinx()
                # ax1.scatter(profile_height[:,0],profile_height[:,1],label='Height Layer'+str(i))
                # ax1.scatter(next_profile_height[:,0],next_profile_height[:,1],label='Height Layer'+str(i+1))
                # ax2.plot(all_profile_v_plot[:,0],all_profile_v_plot[:,1],label='Planned Corrected Speed')
                # ax2.plot(robot_p_S1TCP[:,0],robot_v_S1TCP,label='Actual Cartesian Speed')
                # ax1.set_xlabel('X-axis (Lambda) (mm)')
                # ax1.set_ylabel('Height (mm)', color='g')
                # ax2.set_ylabel('Speed (mm/sec)', color='b')
                # ax1.legend(loc=0)
                # ax2.legend(loc=0)
                # plt.title("Height and Speed, 40 MoveL")
                # plt.legend()
                # plt.show()

                # fig, ax1 = plt.subplots()
                # ax2 = ax1.twinx()
                # ax1.scatter(dh_in_layer[:,0],dh_in_layer[:,1],label='dH L'+str(i)+'L'+str(i+1))
                # ax2.plot(robot_p_S1TCP[:,0],robot_v_S1TCP,label='Actual Cartesian Speed')
                # ax1.set_xlabel('X-axis (Lambda) (mm)')
                # ax1.set_ylabel('dH (mm)', color='g')
                # ax2.set_ylabel('Speed (mm/sec)', color='b')
                # ax1.legend(loc=0)
                # ax2.legend(loc=0)
                # plt.title("dH and Speed, 40 MoveL")
                # plt.legend()
                # plt.show()

                # fig, ax1 = plt.subplots()
                # ax2 = ax1.twinx()
                # ax1.scatter(dh_in_layer[:,0],dh_in_layer[:,1],label='dH L'+str(i)+'L'+str(i+1))
                # ax2.plot(robot_p_S1TCP[:,0],robot_a_S1TCP,label='Actual Cart Acc')
                # ax1.set_xlabel('X-axis (Lambda) (mm)')
                # ax1.set_ylabel('dH (mm)', color='g')
                # ax2.set_ylabel('Acceleration (mm/sec^2)', color='b')
                # ax1.legend(loc=2)
                # ax2.legend(loc=1)
                # plt.title("dH and Acceleration, 40 MoveL")
                # plt.show()
            except:
                continue
            
        forward_flag= not forward_flag

        all_h_mean.append(np.mean(profile_height[:,1]))
        all_h_std.append(np.std(profile_height[:,1]))

    i=0
    m_size=12
    # print('profile_height',profile_height)
    print('all_profile_height',all_profile_height)
    weld_v=2

    print(len(all_profile_height))
    for profile_height in all_profile_height:
        if i==0:
            plt.scatter(profile_height[:,0],profile_height[:,1],s=3,c='tab:blue',label='Forward (Right to Left)')
        elif i==1:
            plt.scatter(profile_height[:,0],profile_height[:,1],s=3,c='tab:orange',label='Backward (Left to Right)')
        elif i%2==0:
            plt.scatter(profile_height[:,0],profile_height[:,1],s=3,c='tab:blue')
        else:
            plt.scatter(profile_height[:,0],profile_height[:,1],s=3,c='tab:orange')
        i+=1
    plt.xlabel('X-axis (Lambda) (mm)')
    plt.ylabel('height (mm)')
    plt.legend()
    plt.title("Height Profile")
    plt.show()
    datasets_h_mean[dataset]=np.array(all_h_mean)

    datasets_h_std[dataset]=np.array(all_h_std)

for dataset in datasets:
    plt.plot(np.arange(len(datasets_h_mean[dataset])),datasets_h_mean[dataset],'-o',label=dataset)
plt.legend()
plt.xlabel('Layer')
plt.ylabel('Mean Height (mm)')
plt.title("Mean Height")
plt.show()
print('datasets_h_mean[dataset]',datasets_h_mean[dataset])
datasets_without_baselayer = datasets_h_mean[dataset][2:]
diff_arr = np.diff(datasets_without_baselayer)
weld_velocity=[5,5]
for i in range(len(diff_arr)):
    weld_velocity.append(weld_v)
    if weld_v==weld_velocity[-2]:
        weld_v+=2
weld_velocity = weld_velocity[2:]
print("weld_velocity",len(weld_velocity))
print("diff_arr",len(diff_arr))
weld_velocity_odd = weld_velocity[::2]
diff_arr_odd = diff_arr[::2]
print("weld_velocity",weld_velocity)
print("diff_arr",diff_arr)
print(len(datasets_without_baselayer))
# 定义两个数组
x = np.array(weld_velocity)
y = np.array(diff_arr)
print('len(x)',len(x))
print('len(y)',len(y))
# 使用 polyfit 函数进行二次拟合，参数 2 是拟合函数的次数
coeffs = np.polyfit(x, y, 2)
print('coeffs',coeffs)
# 创建多项式对象，以便进行函数的评估
p = np.poly1d(coeffs)
# 使用 linspace 函数创建用于绘图的 x 值数组
x_fit = np.linspace(x.min(), x.max(), 500)
# 使用多项式对象计算出 y 值
y_fit = p(x_fit)
# 绘图：原始数据点和拟合曲线
plt.plot(x, y, 'o')
plt.plot(x_fit, y_fit, '-')
plt.show()
plt.close()

log_x = np.log(x)
log_y = np.log(y)
print('log_x',log_x)
print('log_y',log_y)
coeffs = np.polyfit(log_x, log_y, 1)
p = np.poly1d(coeffs)
print('loglog-coeffs',coeffs)
# 使用 linspace 函数创建用于绘图的 x 值数组
x_fit = np.linspace(log_x.min(), log_x.max(), 500)
# 使用多项式对象计算出 y 值
y_fit = p(x_fit)
plt.plot(log_x, log_y, 'o',label='data_points')
plt.plot(x_fit, y_fit, '-', label = 'loglog_fit_model')
plt.legend()
plt.xlabel('log(v) (log(mm/s))')
plt.ylabel('log(delta_h) (log(mm))')
plt.title(f"loglog_plot of {data_dir}")
plt.show()
plt.close()



