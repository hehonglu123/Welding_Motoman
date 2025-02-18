from copy import deepcopy
import numpy as np
import yaml
from pathlib import Path
from motoman_def import *
from redundancy_resolution_dual import *

def get_torch_scanner_ori(Rz_vec, layer_weld_scan_vec, rotate_y_direction):
    Ry_vec = rot(Rz_vec, rotate_y_direction)@layer_weld_scan_vec
    Ry_vec = Ry_vec/np.linalg.norm(Ry_vec)
    Ry_vec = Ry_vec - np.dot(Ry_vec,Rz_vec)*Rz_vec
    Ry_vec = Ry_vec/np.linalg.norm(Ry_vec)
    Rx_vec = np.cross(Ry_vec,Rz_vec) # right hand rule
    return np.array([Rx_vec,Ry_vec,Rz_vec]).T

def main():

    ## define the robot
    zero_config = np.zeros(6)
    config_dir='../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    # get fujicam standoff distance
    # Move the fujicam frame along the z-axis with the distance of the standoff distance
    # will locate the frame onto 
    # the plane perpendicular to the weldgun axis and passing through the weldgun TCP
    T_weldgun = robot_weld.fwd(np.zeros(6))
    T_scanner = robot_scan.fwd(np.zeros(6))
    fujicam_standoff_d = np.dot((T_weldgun.p-T_scanner.p),T_weldgun.R[:3,2])/np.dot(T_scanner.R[:3,2],T_weldgun.R[:3,2])
    robot_scan_motion=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=fujicam_standoff_d,tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')
    r_weld_z = robot_weld.fwd(np.zeros(6))
    r_scan_z = robot_scan_motion.fwd(np.zeros(6))
    T_weld_scan = r_weld_z.inv()*r_scan_z
    weld_scan_vec = T_weld_scan.p/np.linalg.norm(T_weld_scan.p)
    rotate_y_direction = subproblem1(weld_scan_vec, np.array([0,1,0]), np.array([0,0,1]))
    dist_weld_scan = np.linalg.norm(T_weld_scan.p)

    ## planning parameters
    R1_w = 0.01
    R2_w = 0.01

    ## always plan for lagging
    ## then plan for both forward and backward

    data_dir = '../data/wall_weld_test/'
    ## read curve data meta data
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    path_dl = meta_data['path_dl']

    layers_name = ['baselayer','layer']
    for layer_name in layers_name:
        if layer_name == 'baselayer':
            layer_num = meta_data['baselayer_num']
        else:
            layer_num = meta_data['layer_num']

        for layer_n in range(layer_num):
            ##### read curve data #####
            if layer_name == 'baselayer':
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{layer_n}_0.csv',delimiter=',')
            else:
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')

            # positioner is always at [-15, *]
            po_lower_limit = deepcopy(positioner.lower_limit)
            po_upper_limit = deepcopy(positioner.upper_limit)
            po_lower_limit[0] = np.radians(-15-0.001)
            po_upper_limit[0] = np.radians(-15+0.001)
            positioner.lower_limit = po_lower_limit
            positioner.robot.joint_lower_limit = po_lower_limit
            positioner.upper_limit = po_upper_limit
            positioner.robot.joint_upper_limit = po_upper_limit
            
            ##### generate robot js ######
            ### forward case (+x direction)
            ## get the first point where both the torch and scanner are on the layer
            layer_weld_scan_vec = curve[dist_weld_scan_index,:3]-curve[0,:3]
            orientation_start = get_torch_scanner_ori(curve[dist_weld_scan_index,3:], layer_weld_scan_vec, rotate_y_direction)
            positioner_j2_start = -1*(np.radians(180)-np.arctan2(curve[dist_weld_scan_index,1],curve[dist_weld_scan_index,0]))
            ## solve ik when the scanner is NOT on the layer yet
            curve_part = deepcopy(curve[:dist_weld_scan_index+1])
            curve_part = curve_part[::-1]
            rrd=redundancy_resolution_dual(robot_weld,positioner,curve_part[:,:3],curve_part[:,3:])
            q_init_table = np.radians([-15, positioner_j2_start])
            T_start = Transform(orientation_start,curve_part[0,:3]) # starting transfomation in the positioner tip frame
            T_positioner_start = positioner.fwd(np.radians([-15, positioner_j2_start]),world=True) # positioner starting transformation in the world frame
            T_start_robot = T_positioner_start*T_start # starting transformation in the robot base frame
            q_init=robot_weld.inv(T_start_robot.p,T_start_robot.R,zero_config)[0]
            q_out1, q_out2 = rrd.dual_arm_5dof_stepwise(q_init,q_init_table,w1=R1_w,w2=R2_w)
            ## solve ik then the torch and scanner is both on the layer


            ## get orientation (R) for curve using scanner position
            curve_R = []
            for i in range(len(curve)):
                dist_weld_scan_index = np.round(dist_weld_scan/path_dl).astype(int)
                




                # get Rz vector
                Rz_vec = curve[i,3:]
                Rz_vec = Rz_vec/np.linalg.norm(Rz_vec)
                # get Ry vector by using scanner position
                
                if lead_lag_flag==0:
                    if dist_weld_scan_index+i<len(curve):
                        layer_weld_scan_vec = curve[dist_weld_scan_index+i,:3]-curve[i,:3]
                    else:
                        if i!=len(curve)-1:
                            layer_weld_scan_vec = curve[-1,:3]-curve[i,:3]
                        else:
                            layer_weld_scan_vec = curve[-1,:3]-curve[-2,:3]
                else:
                    if i-dist_weld_scan_index>=0:
                        layer_weld_scan_vec = curve[i-dist_weld_scan_index,:3]-curve[i,:3]
                    else:
                        if i!=0:
                            layer_weld_scan_vec = curve[0,:3]-curve[i,:3]
                        else:
                            layer_weld_scan_vec = curve[1,:3]-curve[0,:3]
                
                curve_R.append(np.vstack((Rx_vec,Ry_vec,Rz_vec)).T)
            ## solve ik
            rrd=redundancy_resolution_dual(robot_weld,positioner,curve[:,:3],curve_R)
            q_init_table = np.radians([-15, 270])
            q_init=robot_weld.inv(scan_p_R2Base[0],scan_R_R2Base[0],zero_config)[0]
            q_out1, q_out2 = rrd.dual_arm_6dof_stepwise(q_init,q_init_table,w1=R1_w,w2=R2_w)


            Path('curve_sliced_js').mkdir(parents=True, exist_ok=True)
            data_dir = 'curve_sliced_js/'

if __name__ == "__main__":
    main()