import time, os, copy, sys, yaml
import glob
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from motoman_def import *
from robotics_utils import *
sys.path.append('../../scan/scan_process/')
sys.path.append('../../scan/scan_tools/')
from scan_utils import *
from scanProcess import *

def main():

    ############## Robot definition ##############
    config_dir='../../config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

    positioner_joints = np.radians([-15,180])

    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'
    with open(data_dir+'sliced_meta.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    
    base_layer_num= meta_data['baselayernum']
    baselayer_resolution= meta_data['baselayer_resolution']
    layer_num = meta_data['layer_num']
    layer_resolution = meta_data['layer_resolution']

    logdata_dir = data_dir+'weld_fujiscan_2025_02_07_13_43_40/'

    for weld_parts in ['layer']:
        if weld_parts == 'base':
            total_layers_name = glob.glob(logdata_dir+'baselayer*')
        else:
            total_layers_name = glob.glob(logdata_dir+'layer*')
        # get printed layer number
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))

        print('Layer numbers:',layer_nums)

        # build layers from bottom to top by layers
        Transz0_H = None
        for layer_n in layer_nums:
            
            # read layer curve data
            if weld_parts == 'base':
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{layer_n}_0.csv',delimiter=',')
            else:
                curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')

            # read logged data
            if weld_parts == 'base':
                layer_name = 'baselayer'+str(layer_n)
            else:
                layer_name = 'layer'+str(layer_n)
            print('Processing layer:',layer_name)
            this_layer_dir = logdata_dir+layer_name+'/'+layer_name
            robot_stamps = np.loadtxt(this_layer_dir+'_timestamps_exe.csv',delimiter=',')
            weld_js_exe = np.loadtxt(this_layer_dir+'_weld_js_exe.csv',delimiter=',')
            weld_js_exe = weld_js_exe[:,:6] # get only robot 1 joints
            with open(this_layer_dir+'_scan_exe.pickle', 'rb') as f:
                scan_exe = pickle.load(f)
            
            assert len(weld_js_exe) == len(scan_exe), 'Weld joint and scan data length mismatched'
        
            # processing the scans
            scan_process = ScanProcess(robot_scan,positioner)
            pcd=None
            pcd = scan_process.pcd_register_mti(scan_exe,weld_js_exe,robot_stamps,static_positioner_q=positioner_joints,flip=True,scanner='fuji')

            # cropping the point cloud
            curve_planned_z = np.mean(curve[:,2])
            curve_x_end = np.min(curve[:,0])
            curve_x_start = np.max(curve[:,0])
            z_height_start=curve_planned_z-3
            # z_height_start = -3
            print(z_height_start)
            crop_extend=10
            crop_min=(curve_x_end-crop_extend,-30,-30)
            crop_max=(curve_x_start+crop_extend,30,z_height_start+8)
            crop_h_min=(curve_x_end-crop_extend,-20,-30)
            crop_h_max=(curve_x_start+crop_extend,20,z_height_start+8)
            pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                                min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
            profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
            print("Transz0_H:",Transz0_H)

            # save processed profile height and point cloud
            np.savetxt(this_layer_dir+'_profile_height.csv',profile_height,delimiter=',')
            o3d.io.write_point_cloud(this_layer_dir+'_pcd.pcd',pcd)

            # visualize the reconstructed point cloud
            visualize_pcd([pcd])
            plt.scatter(profile_height[:,0],profile_height[:,1])
            plt.show()

if __name__ == '__main__':
    main()