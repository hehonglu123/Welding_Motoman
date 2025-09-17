from copy import deepcopy
import numpy as np
import yaml
from pathlib import Path
from matplotlib import pyplot as plt
from motoman_def import *
from redundancy_resolution_dual import *
from redundancy_resolution import *
import open3d as o3d
sys.path.append('../scan/scan_process/')
sys.path.append('../scan/scan_tools/')
from scan_utils import *

def get_scanner_ori(Rz_vec, Rx_vec):
    Rz_vec = Rz_vec/np.linalg.norm(Rz_vec)
    Rx_vec = Rx_vec/np.linalg.norm(Rx_vec)
    Rx_vec = Rx_vec - np.dot(Rx_vec,Rz_vec)*Rz_vec
    Rx_vec = Rx_vec/np.linalg.norm(Rx_vec)
    Ry_vec = np.cross(Rz_vec,Rx_vec) # right hand rule
    return np.array([Rx_vec,Ry_vec,Rz_vec]).T

def get_torch_scanner_ori(Rz_vec, layer_weld_scan_vec, rotate_y_direction):
    Rz_vec = Rz_vec/np.linalg.norm(Rz_vec)
    Ry_vec = -rot(Rz_vec, rotate_y_direction)@layer_weld_scan_vec
    Ry_vec = Ry_vec/np.linalg.norm(Ry_vec)
    Ry_vec = Ry_vec - np.dot(Ry_vec,Rz_vec)*Rz_vec
    Ry_vec = Ry_vec/np.linalg.norm(Ry_vec)
    Rx_vec = np.cross(Ry_vec,Rz_vec) # right hand rule
    return np.array([Rx_vec,Ry_vec,Rz_vec]).T

def get_robot_init_from_positioner(robot,positioner,q_positioner,position,orientation,zero_config=np.zeros(6)):
    T_start = Transform(orientation,position) # starting transfomation in the positioner tip frame
    T_positioner_start = positioner.fwd(q_positioner,world=True) # positioner starting transformation in the world frame
    T_start_robot = T_positioner_start*T_start # starting transformation in the robot base frame
    q_init=robot.inv(T_start_robot.p,T_start_robot.R,zero_config)[0]
    return q_init

def main():

    ## define the robot
    zero_config = np.zeros(6)
    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_weld.robot.p_tool = np.array([-49.29784692 ,  2.8512889,  476.90722611])
    robot_weld.p_tool = deepcopy(robot_weld.robot.p_tool)
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    # get fujicam standoff distance
    # Move the fujicam frame along the z-axis with the distance of the standoff distance
    # will locate the frame onto 
    # the plane perpendicular to the weldgun axis and passing through the weldgun TCP
    T_weldgun = robot_weld.fwd(np.zeros(6))
    T_scanner = robot_scan.fwd(np.zeros(6))
    print("T_weldgun:\n", T_weldgun)
    print("T_scanner:\n", T_scanner)
    print("T_weld_scan:\n", T_weldgun.inv()*T_scanner)
    print("Weld vs Scan angle:", np.degrees(np.arccos(np.dot(-T_weldgun.R[:3,2],T_scanner.R[:3,2]))))
    fujicam_standoff_d = np.dot((T_weldgun.p-T_scanner.p),T_weldgun.R[:3,2])/np.dot(T_scanner.R[:3,2],T_weldgun.R[:3,2])
    robot_scan_motion=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=fujicam_standoff_d,tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_thermal=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')
    r_weld_z = robot_weld.fwd(np.zeros(6))
    r_scan_z = robot_scan_motion.fwd(np.zeros(6))
    T_weld_scan = r_weld_z.inv()*r_scan_z
    print("T_weld_scan_motion:\n", T_weld_scan)
    weld_scan_vec = T_weld_scan.p/np.linalg.norm(T_weld_scan.p)
    rotate_y_direction = subproblem1(weld_scan_vec, np.array([0,1,0]), np.array([0,0,1]))
    dist_weld_scan = np.linalg.norm(T_weld_scan.p)
    torch_z_shift = np.dot(T_weld_scan.p,T_weld_scan.R[:3,2])
    torch_z_shift /= 2

    ## planning parameters
    R1_w = 0.001
    R2_w = 0.05
    R1_w_scan = 0.001
    R2_w_scan = 0.05
    thermal_distance=400
    scanning_extend_distance = 20 # mm

    ## always plan for lagging
    ## then plan for both forward and backward
    data_dir = '../data/wall_weld_test/'
    # data_dir = '../data/casing_scaled/'

    ## read curve data meta data
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)

    #### visualize the path in 3D with equal aspect ratio ####
    # pcd = o3d.geometry.PointCloud()
    # curve_points = []
    # points_per_layer = []
    # for layer_n in range(meta_data['layer_num']):
    #     if layer_n%10!=0:
    #         continue
    #     print("Reading layer:", layer_n,flush=True)
    #     curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')
    #     curve_points.extend(curve[:,:3])
    #     points_per_layer.append(len(curve))

    #     # if layer_n>=1169:
    #     #     print("layer:", layer_n)
    #     #     plt.plot(curve[:,0],curve[:,1], '-o')
    #     #     plt.axis('equal')
    #     #     plt.show()
    # points_per_layer = np.cumsum(points_per_layer)
    # pcd.points = o3d.utility.Vector3dVector(np.array(curve_points))
    # path_dl_all = np.linalg.norm(np.diff(np.array(curve_points),axis=0),axis=1)
    # print("Path dl:", np.mean(path_dl_all), "std:", np.std(path_dl_all), "min:", np.min(path_dl_all), "max:", np.max(path_dl_all))
    # # find the max path dl points
    # max_dl_index = np.argmax(path_dl_all)
    # layer_max_dl = np.where(points_per_layer > max_dl_index)[0][0]
    # print("max dl index:", max_dl_index, "max dl:", path_dl_all[max_dl_index], "at layer:", layer_max_dl, "index:", max_dl_index-points_per_layer[layer_max_dl-1])
    # visualize_pcd([pcd])
    # # find the point with max z
    # curve_points = np.array(curve_points)
    # max_z_index = np.argmax(curve_points[:,2])
    # print("max z index:", max_z_index, "max z p:", curve_points[max_z_index])
    # exit()
    
    ## get the index (distance) where the scanner is on the layer
    path_dl = meta_data['path_dl']
    dist_weld_scan_index = np.round(dist_weld_scan/path_dl).astype(int)
    print(f'dist_weld_scan_index: {dist_weld_scan_index}, dist_weld_scan: {dist_weld_scan}')

    layers_name = ['baselayer','layer']
    # layers_name = ['layer']
    for layer_name in layers_name:
        if layer_name == 'baselayer':
            layer_num = meta_data['baselayer_num']
        else:
            layer_num = meta_data['layer_num']

        st = time.time()
        start_layer = 0
        for layer_n in range(start_layer,layer_num):
            if layer_n<layer_num-2:
                continue # test the last two layers first
            ##### read curve data #####
            if layer_name == 'baselayer':
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{layer_n}_0.csv',delimiter=',')
            else:
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')

            # positioner is always at [-15, *]
            po_lower_limit = deepcopy(positioner.lower_limit)
            po_upper_limit = deepcopy(positioner.upper_limit)
            po_lower_limit = np.radians([-15-0.01,-190])
            po_upper_limit = np.radians([-15+0.01,190])
            positioner.lower_limit = po_lower_limit
            positioner.robot.joint_lower_limit = po_lower_limit
            positioner.upper_limit = po_upper_limit
            positioner.robot.joint_upper_limit = po_upper_limit

            # robot scan limits
            rob_upper_limit = deepcopy(robot_scan_motion.upper_limit)
            rob_upper_limit[2] = np.radians(65)
            robot_weld.upper_limit = rob_upper_limit
            robot_weld.robot.joint_upper_limit = rob_upper_limit
            robot_scan_motion.upper_limit = rob_upper_limit
            robot_scan_motion.robot.joint_upper_limit = rob_upper_limit
            
            ##### generate robot js ######
            if layer_n%2 == 0:
                all_cases = ['forward']
            else:
                all_cases = ['backward']
            all_cases = ['forward','backward']

            for cases in all_cases:
                ### forward case (+x direction)
                ### backward case (-x direction)
                if cases == 'backward':
                    curve = curve[::-1]

                rWeld_js = []
                rScan_js = []
                positioner_js = []
                poScan_js = []
                ## get the first point where both the torch and scanner are on the layer
                layer_weld_scan_vec = curve[dist_weld_scan_index,:3]-curve[0,:3]
                orientation_start = get_torch_scanner_ori(curve[dist_weld_scan_index,3:], layer_weld_scan_vec, rotate_y_direction)
                if cases == 'forward':
                    # positioner_j2_start = np.degrees(-1*(np.radians(180)-np.arctan2(curve[dist_weld_scan_index,1],curve[dist_weld_scan_index,0])))
                    positioner_j2_start = -90
                else:
                    positioner_j2_start = 90
                ## solve ik when the scanner is NOT on the layer yet
                curve_part = deepcopy(curve[:dist_weld_scan_index+1])
                curve_part = curve_part[::-1]
                rrd=redundancy_resolution_dual(robot_weld,positioner,curve_part[:,:3],curve_part[:,3:])
                q_init_table = np.radians([-15, positioner_j2_start])
                q_init = get_robot_init_from_positioner(robot_weld,positioner,q_init_table,curve_part[0,:3],orientation_start,zero_config=zero_config)
                q_out1, q_out2 = rrd.dual_arm_5dof_stepwise(q_init,q_init_table,w1=R1_w,w2=R2_w)
                rWeld_js.extend(q_out1[::-1])
                positioner_js.extend(q_out2[::-1])
                ## solve ik when the torch and scanner is both on the layer
                # get orientation (R) for curve using scanner position
                curve_R = []
                for i in range(dist_weld_scan_index+1,len(curve)):
                    curve_R.append(get_torch_scanner_ori(curve[i,3:], layer_weld_scan_vec, rotate_y_direction))
                assert len(curve[dist_weld_scan_index+1:,:3]) == len(curve_R), 'curve and curve_R length mismatched'
                rrd=redundancy_resolution_dual(robot_weld,positioner,curve[dist_weld_scan_index+1:,:3],curve_R)
                q_init_table = positioner_js[-1]
                q_init = rWeld_js[-1]
                q_out1, q_out2 = rrd.dual_arm_6dof_stepwise(q_init,q_init_table,w1=R1_w,w2=R2_w)
                rWeld_js.extend(q_out1)
                positioner_js.extend(q_out2)
                # solve ik for robot with thermal when torch part is done
                rr_thermal = redundancy_resolution(robot_weld,positioner,curve)
                rThermal_js = rr_thermal.rob2_flir_resolution([[rWeld_js]],robot_thermal,measure_distance=thermal_distance,rotate_angle=np.radians(15),y_direction=np.array([0,0,-1]))[0][0]
                ## visualize the js
                # plt.plot(np.degrees(rWeld_js), '-o')
                # plt.legend(['j1','j2','j3','j4','j5','j6'])
                # plt.title('weld robot js')
                # plt.show()

                # plt.plot(np.degrees(rThermal_js), '-o')
                # plt.legend(['j1','j2','j3','j4','j5','j6'])
                # plt.title('thermal robot js')
                # plt.show()
                ## solve ik when the torch is NOT on the layer (leaving the layer)
                curve_part = deepcopy(curve[-dist_weld_scan_index-1:])
                curve_part = curve_part[:,:3]
                # extend the scanning at the end uniformly
                extend_vec = curve_part[-1,:3]-curve_part[-2,:3]
                extend_vec = extend_vec/np.linalg.norm(extend_vec)
                curve_extend = np.linspace(curve_part[-1,:3],curve_part[-1,:3]+scanning_extend_distance*extend_vec,int(scanning_extend_distance//path_dl+1))
                curve_part = np.vstack((curve_part,curve_extend[1:]))
                # get orientation using scanner orientation
                T_robot = robot_scan_motion.fwd(rWeld_js[-1],world=True)
                T_positioner = positioner.fwd(positioner_js[-1],world=True)
                T_robot_positioner = T_positioner.inv()*T_robot
                curve_R_start = T_robot_positioner.R # starting scanner orientation in the positioner tip frame
                if cases == 'forward':
                # if cases == 'backward':
                    curve_R_final = np.array([[-1,0,0],\
                                            [0,-1,0],\
                                            [0,0,1]]) # target ending scanner orientation in the positioner tip frame
                else:
                    curve_R_final = np.array([[1,0,0],\
                                            [0,1,0],\
                                            [0,0,1]]) # target ending scanner orientation in the positioner tip frame
                rot_k, rot_theta = R2rot(curve_R_start.T@curve_R_final)
                rot_theta /= 2
                print("scanner rotation angle:",np.degrees(rot_theta))
                curve_R = []
                curve_quat = []
                for i in range(1,len(curve_part)):
                    curve_R.append(curve_R_start@rot(rot_k,rot_theta*i/(len(curve_part)-1)))
                    curve_quat.append(R2q(curve_R[-1]))
                    curve_part[i,2] = curve_part[i,2] + torch_z_shift*(i/(len(curve_part)-1))
                curve_part = curve_part[1:]
                assert len(curve_part) == len(curve_R), 'curve and curve_R length mismatched'
                curve_scan = np.hstack((curve_part[:,:3],curve_quat))
                rrd=redundancy_resolution_dual(robot_scan_motion,positioner,curve_part[:,:3],curve_R)
                q_init_table = positioner_js[-1]
                q_init = rWeld_js[-1]
                q_out1, q_out2 = rrd.dual_arm_6dof_stepwise(q_init,q_init_table,w1=R1_w_scan,w2=R2_w_scan)
                rScan_js.extend(q_out1)
                poScan_js.extend(q_out2)


                Path(data_dir+'curve_sliced_js').mkdir(parents=True, exist_ok=True)
                if layer_name == 'baselayer':
                    robot_output_data_dir = data_dir+f'curve_sliced_js/MA2010_base_js{layer_n}_0_{cases}'
                    robot_thermal_output_data_dir = data_dir+f'curve_sliced_js/MA1440_base_js{layer_n}_0_{cases}'
                    positioner_output_data_dir = data_dir+f'curve_sliced_js/D500B_base_js{layer_n}_0_{cases}'
                    robot_scan_output_data_dir = data_dir+f'curve_sliced_js/MA2010_base_js{layer_n}_0_scan_{cases}'
                    positioner_scan_output_data_dir = data_dir+f'curve_sliced_js/D500B_base_js{layer_n}_0_scan_{cases}'
                    curve_scan_output_data_dir = data_dir+f'curve_sliced_relative/baselayer{layer_n}_0_scan_{cases}.csv'
                else:
                    robot_output_data_dir = data_dir+f'curve_sliced_js/MA2010_js{layer_n}_0_{cases}'
                    robot_thermal_output_data_dir = data_dir+f'curve_sliced_js/MA1440_js{layer_n}_0_{cases}'
                    positioner_output_data_dir = data_dir+f'curve_sliced_js/D500B_js{layer_n}_0_{cases}'
                    robot_scan_output_data_dir = data_dir+f'curve_sliced_js/MA2010_js{layer_n}_0_scan_{cases}'
                    positioner_scan_output_data_dir = data_dir+f'curve_sliced_js/D500B_js{layer_n}_0_scan_{cases}'
                    curve_scan_output_data_dir = data_dir+f'curve_sliced_relative/slice{layer_n}_0_scan_{cases}.csv'
                np.savetxt(robot_output_data_dir+'.csv',np.array(rWeld_js),delimiter=',')
                np.savetxt(robot_thermal_output_data_dir+'.csv',np.array(rThermal_js),delimiter=',')
                np.savetxt(positioner_output_data_dir+'.csv',np.array(positioner_js),delimiter=',')
                np.savetxt(robot_scan_output_data_dir+'.csv',np.array(rScan_js),delimiter=',')
                np.savetxt(positioner_scan_output_data_dir+'.csv',np.array(poScan_js),delimiter=',')
                np.savetxt(curve_scan_output_data_dir,curve_scan,delimiter=',')
                print(f'{layer_name} {layer_n} {cases} done')
                print(f'Estimated time left h/m/s: {((time.time()-st)/(layer_n-start_layer+1)*(layer_num-layer_n-1))//3600}h {((time.time()-st)/(layer_n-start_layer+1)*(layer_num-layer_n-1))%3600//60}m {((time.time()-st)/(layer_n-start_layer+1)*(layer_num-layer_n-1))%60}s')
                print('---------------------------------')

if __name__ == "__main__":
    main()