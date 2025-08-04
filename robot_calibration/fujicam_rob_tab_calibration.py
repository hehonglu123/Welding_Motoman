import time, os, copy, sys, yaml, pathlib, pickle
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from motoman_def import *
from robotics_utils import *
sys.path.append('../scan/scan_process/')
sys.path.append('../scan/scan_tools/')
from scan_utils import *
from scanProcess import *

def main():

    ############## Robot definition ##############
    config_dir='../config/'
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')
    positioner_H = deepcopy(positioner.base_H)
    
    data_dir = 'turntable_calibration/'
    all_data_names = ['angle_0_0', 'angle_0_1', 'angle_0_2', 'angle_1_0']
    color_code = np.linspace([1,0,0],[0,1,0],len(all_data_names))
    positioner_shift_x = 0 # shift the positioner to the right for better visibility

    #### read raw scan and js data ####

    all_pcds = []
    for data_name,this_color in zip(all_data_names,color_code):
        print(f"Processing data: {data_name}")
        positioner.base_H[0,3] += positioner_shift_x

        with open(f'{data_dir}{data_name}_scan_exe.pickle', 'rb') as f:
            scan_exe = pickle.load(f)
        scan_js_exe = np.loadtxt(f'{data_dir}{data_name}_weld_js_exe.csv', delimiter=',')

        # processing the scans
        scan_process = ScanProcess(robot_scan,positioner)
        if os.path.exists(f'{data_dir}{data_name}_scan_exe_noise_remove.pickle'):
            with open(f'{data_dir}{data_name}_scan_exe_noise_remove.pickle', 'rb') as f:
                scan_exe_noise_remove = pickle.load(f)
        else:
            scan_exe_noise_remove = []
            skip_id = []
            count_id = 0
            for (weld_js,scan) in zip(scan_js_exe,scan_exe):
                try:
                    scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                    scan_exe_noise_remove.append(scan_noise_remove)
                except ValueError as e:
                    skip_id.append(count_id)
                count_id += 1
            scan_js_exe = np.delete(scan_js_exe, skip_id, axis=0)
            np.savetxt(f'{data_dir}{data_name}_scan_js_exe_noise_remove.csv', scan_js_exe, delimiter=',')
            with open(f'{data_dir}{data_name}_scan_exe_noise_remove.pickle', 'wb') as f:
                pickle.dump(scan_exe_noise_remove, f)
            print(f"Total scans without points: {len(skip_id)}, out of {len(scan_exe)} scans.")
        
        scan_js_exe = np.loadtxt(f'{data_dir}{data_name}_scan_js_exe_noise_remove.csv', delimiter=',')
        robot_stamps = scan_js_exe[:, 0]
        scan_js_exe = scan_js_exe[:, [1,2,3,4,5,6,13,14]]  # select only the relevant joints (robot 1 and positioner)

        pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,scan_js_exe,robot_stamps,flip=True,scanner='fuji')
        pcd = scan_process.pcd_noise_remove(pcd,voxel_size=0.1,crop_flag=False,outlier_remove=True,cluster_based_outlier_remove=True)
        visualize_pcd([pcd])
        pcd.paint_uniform_color(this_color)
        all_pcds.append(pcd)
        print("==============")

    visualize_pcd(all_pcds)


if __name__ == '__main__':
    main()