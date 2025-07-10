import time, os, copy, sys, yaml, inspect
import glob
from copy import deepcopy
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
import open3d as o3d
import cv2 as cv
from motoman_def import *
from robotics_utils import *
from flir_toolbox import *
from ultralytics import YOLO
sys.path.append('../../scan/scan_process/')
sys.path.append('../../scan/scan_tools/')
from scan_utils import *
from scanProcess import *
from animation_3d import *

# # feat_detector = cv.ORB_create()
# feat_detector = cv.SIFT_create()
# # bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
# bf_matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

def main():

    ############## Robot definition ##############
    config_dir='../../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_thermal=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

    # positioner_joints = np.radians([-15,180])

    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'

    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_17_39_17/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_17_39_17/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/']
    # logdata_dir_all = ['weld_fujicontrol_2025_03_12_18_27_33/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujicontrol_2025_03_12_18_27_33/']
    # logdata_dir_all = ['weld_fujiscan_2025_06_11_14_12_44/']

    # material ER316L (stainless steel)
    # logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/']
    # logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
    #                    'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
    #                    'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']
    logdata_dir_all = ['weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
    # logdata_dir_all = ['weld_fujiscan_2025_06_11_18_14_56/']
    
    ### skip data directories
    skip_data_dir_all = []
    #skip_data_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/']

    # to increase robustness of capturing thermal reading
    # since the camera is following the torch
    # if the torch is not detected, use the last few frames' centroid
    thermal_centroid_record = []

    run_code_again_flag = True # For scanner leading case, need to generate all profile height before actually get dh.
    create_transform = False
    for logdata_dir_name in logdata_dir_all:
        print('Processing:',logdata_dir_name)
        if logdata_dir_name in skip_data_dir_all:
            print("Skipping...")
            continue

        ## determine if the scanner is leading or lagging
        scanner_lagging= False
        if 'scan' in logdata_dir_name:
            scanner_lagging= True
        
        logdata_dir = data_dir+logdata_dir_name
        
        with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
            meta_data = yaml.safe_load(f)

        last_profile_height = None
        # build layers from bottom to top by layers
        all_pcd_transform = []
        all_profile_height = []
        if create_transform or scanner_lagging:
            Transz0_H_odd = None
            Transz0_H_even = None
            Transicp_H_odd2even = None
        else:
            Transz0_H_odd = np.loadtxt(logdata_dir+'Transz0_H_odd.csv',delimiter=',')
            Transz0_H_even = np.loadtxt(logdata_dir+'Transz0_H_even.csv',delimiter=',')
            Transicp_H_odd2even = np.loadtxt(logdata_dir+'Trans_icp_odd2even.csv',delimiter=',')
            Transz0_H_odd = Transz0_H_odd @ Transicp_H_odd2even
        for weld_parts in ['base','layer']:
        # for weld_parts in ['layer']:
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
            layer_nums = np.sort(layer_nums)

            # for layer_n in [layer_nums[-1],layer_nums[-2]]:
            for layer_n_id, layer_n in enumerate(layer_nums):
                # if layer_n_id<7:
                #     continue
                # read layer curve data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{layer_n}_0.csv',delimiter=',')
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')

                # read logged data
                if weld_parts == 'base':
                    layer_name = 'baselayer'+str(layer_n)
                    next_layer_name = 'baselayer'+str(layer_nums[layer_n_id+1]) if layer_n_id+1 < len(layer_nums) else 'layer9'
                else:
                    layer_name = 'layer'+str(layer_n)
                    next_layer_name = 'layer'+str(layer_nums[layer_n_id+1]) if layer_n_id+1 < len(layer_nums) else ''
                print('Processing layer:',layer_name)
                this_layer_dir = logdata_dir+layer_name+'/'
                next_layer_dir = logdata_dir+next_layer_name+'/'
                rob_js_exe = np.loadtxt(this_layer_dir+'weld_js_exe.csv',delimiter=',')
                # get js at index 1~6 and 13 14
                rob2_js_exe = rob_js_exe[:,7:13]
                rob_js_exe = rob_js_exe[:,[0,1,2,3,4,5,6,13,14]]
                robot_stamps = rob_js_exe[:,0]
                with open(this_layer_dir+'scan_exe.pickle', 'rb') as f:
                    scan_exe = pickle.load(f)
                
                assert len(rob_js_exe) == len(scan_exe), 'Weld joint and scan data length mismatched'

                ############### get welding commands #####################
                weld_cmd = np.loadtxt(this_layer_dir+'weld_cmd.csv',delimiter=',')

                ############### get welding js ####################
                stamps_diff_sorted = np.argsort(np.diff(robot_stamps))[::-1]
                for stamp_diff_id in stamps_diff_sorted:
                    # make sure to find the time jump after the welding command
                    if robot_stamps[stamp_diff_id] > weld_cmd[-1,0] and robot_stamps[stamp_diff_id] < weld_cmd[-1,0]+3:
                        weld_split_id = stamp_diff_id
                        break

                # weld_split_id = np.argmax(np.diff(robot_stamps))
                scan_js_exe = deepcopy(rob_js_exe)
                if scanner_lagging:
                    weld_js_exe = rob_js_exe[:weld_split_id+1,:]
                else:
                    weld_js_exe = rob_js_exe[weld_split_id+1:,:]

                ############### get welding status ##############
                print("Getting welding status...")
                welding_status = np.loadtxt(this_layer_dir+'welding.csv',delimiter=',',skiprows=1)

                ############### get welding current status ########
                print("Getting welding current status...")
                welding_current_exe = np.loadtxt(this_layer_dir+'current.csv',delimiter=',',skiprows=1)
                # print(welding_current_exe)
                # plt.plot(welding_current_exe[:,0]-welding_current_exe[0,0], welding_current_exe[:,1], label='Welding Current')
                # plt.title('Welding Current vs Time')
                # plt.xlabel('Time (s)')
                # plt.ylabel('Welding Current (A)')
                # plt.grid()
                # plt.show()

                if len(welding_current_exe) > 0:
                    # Find peaks
                    peaks, _ = find_peaks(welding_current_exe[:,1], height=75)  # height=0 filters out very low peaks
                    # find the timestamp of the first peak
                    first_strike_time = welding_current_exe[peaks[0], 0]
                    print(f"First peak time: {first_strike_time:.2f} seconds")
                    # plt.plot(welding_current_exe[:,0]-welding_current_exe[0,0], welding_current_exe[:,1], label='Welding Current')
                    # plt.plot(welding_current_exe[peaks,0]-welding_current_exe[0,0], welding_current_exe[peaks,1], "x")
                    # plt.title("Detected Peaks")
                    # plt.show()
                else:
                    print("No welding current data found.")

                ############### get thermal readings ##############
                print("Getting thermal readings...")
                try:
                    thermal_reading = np.loadtxt(this_layer_dir+'thermal.csv',delimiter=',')
                    with open(this_layer_dir+'thermal_pixel_trace.pickle', 'rb') as f:
                        pass
                except FileNotFoundError:
                    print("No thermal readings found, using IR camera to get thermal readings...")
                    with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
                        ir_exe = pickle.load(f)
                    ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')
                    horizontal_offset=0
                    vertical_offset=3
                    ir_pixel_window_size=5
                    cam_pixel_moving_ratio = 1.77 # 1.77 pixel per mm
                    flame_centroid_history=[]
                    thermal_reading = []
                    thermal_stamp = []

                    thermal_pixel_trace = np.array([])
                    thermal_workpiece_x_trace = []
                    thermal_trace = []
                    thermal_trace_stamp = []
                    trace_stamps = []
                    trace_dxdy = []
                    last_trace_stamp = 0
                    des_pre = None
                    last_T_thermal_cam = None
                    # for (ir_image_raw,stamp) in zip(ir_exe,ir_stamp):
                    for (ir_id,ir_image_raw, stamp) in zip(range(len(ir_exe)), ir_exe, ir_stamp):
                        if ir_id % int(len(ir_exe)/10) == 0:
                            print("Processing IR image:", ir_id, "at time", stamp-ir_stamp[0], 'total images:', len(ir_exe))
                        # plt.imshow(np.clip(ir_image_raw, 7000, 9200), cmap='inferno', aspect='equal')
                        # plt.show()
                        # ir_image = np.rot90(ir_image_raw, k=-1)
                        ir_image = deepcopy(ir_image_raw)
                        img_height, img_width = ir_image.shape

                        # robot movement from the last collected thermal reading
                        # find thermal camera pose
                        closest_idx = np.argmin(np.abs(robot_stamps - stamp))
                        T_table = positioner.fwd(rob_js_exe[closest_idx,-2:], world=True)
                        T_thermal_cam = robot_thermal.fwd(rob2_js_exe[closest_idx,:6], world=True)
                        T_table_thermal = T_table.inv() * T_thermal_cam
                        T_torch = robot_weld.fwd(rob_js_exe[closest_idx,1:7], world=True)
                        T_table_torch = T_table.inv() * T_torch
                        if last_T_thermal_cam is not None:
                            # Compute the relative transformation
                            T_rel = last_T_thermal_cam.inv() * T_table_thermal
                            # Extract the translation vector
                            rob_translation = T_rel.p
                            # print(f"Robot moved: {rob_translation}")

                            ##### pixel tracing moving #####
                            moving_dx = -rob_translation[0] * cam_pixel_moving_ratio
                            moving_dy = rob_translation[2] * cam_pixel_moving_ratio
                            # add trace dxdy and stamp to the list
                            trace_dxdy.append(np.array([moving_dx, moving_dy]))
                            trace_stamps.append(stamp)
                        last_T_thermal_cam = deepcopy(T_table_thermal)

                        # move tracing pixel and add thermal reading
                        if len(thermal_pixel_trace) > 0:
                            thermal_pixel_trace = thermal_pixel_trace + np.array([moving_dx, moving_dy])
                            # add the thermal status to the traced pixel trace
                            # if stamp - last_trace_stamp > 0.1: # if more than dt second has passed since the last trace
                            for trace_coord_id, trace_coord in enumerate(thermal_pixel_trace):
                                # if pixel within the image
                                trace_coord_round = np.round(trace_coord).astype(int)
                                if 0+ir_pixel_window_size//2 <= trace_coord_round[0] < img_width- ir_pixel_window_size//2 and 0 <= trace_coord_round[1] < img_height:
                                    # update the thermal status
                                    thermal_trace[trace_coord_id].append(get_pixel_value(ir_image, trace_coord_round, ir_pixel_window_size))
                                    thermal_trace_stamp[trace_coord_id].append(stamp)
                                    thermal_workpiece_x_trace[trace_coord_id].append(thermal_workpiece_x_trace[trace_coord_id][0])
                            last_trace_stamp = stamp    

                        # centroid, bbox, torch_centroid, torch_bbox=weld_detection_aluminum(ir_image,torch_model,percentage_threshold=0.8)
                        centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_wire_model)

                        # plt.imshow(ir_image, cmap='inferno', aspect='equal')
                        # plt.show()
                        # plt.plot(welding_current_exe[:,0]-welding_current_exe[0,0], welding_current_exe[:,1], label='Welding Current')
                        # # draw a vertical line at the current time
                        # plt.axvline(x=stamp-welding_current_exe[0,0], color='r', linestyle='--', label='Current Time')
                        # plt.title('Welding Current vs Time')
                        # plt.xlabel('Time (s)')
                        # plt.ylabel('Welding Current (A)')
                        # plt.xlim(-0.1,0.5)
                        # plt.grid()
                        # plt.show()

                        # find max pixel value in ir_image
                        # centroid = np.unravel_index(np.argmax(ir_image, axis=None), ir_image.shape)
                        if centroid is None:
                            if len(thermal_centroid_record) == 0:
                                print(f"No flame detected in image at {stamp}")
                                # plt.clf()
                                # plt.imshow(ir_image, cmap='inferno', aspect='equal')
                                # plt.pause(0.1)
                                continue
                            else:
                                # use the last N recorded centroid
                                centroid = np.mean(thermal_centroid_record[-5:], axis=0)

                        thermal_centroid_record.append(centroid) # record centroid for debugging

                        # plt.clf()
                        # plt.imshow(ir_image, cmap='inferno', aspect='equal')
                        # plt.scatter(centroid[0], centroid[1], c='r', s=5, label='Flame centroid')
                        # plt.show()

                        #find average pixel value 
                        centroid = np.round(centroid).astype(int)
                        # if ir_image[centroid] >= 1e4:
                        # if stamp > first_strike_time # only collect thermal reading after the first strike
                        pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
                        # pixel_coord = pixel_coord[::-1]
                        flame_reading=get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size)
                        thermal_reading.append(flame_reading)
                        thermal_stamp.append(stamp)
                        # print(flame_reading, centroid)    

                        # add trace pixels and thermal readings
                        # if ir_image[centroid] >= 1e4:
                        # print("Total pixels:", len(thermal_pixel_trace), "current id:", ir_id)
                        # print("T_table_torch p", T_table_torch.p)
                        if len(thermal_pixel_trace) == 0 or np.abs(T_table_torch.p[0]-thermal_workpiece_x_trace[-1][0]) > 0.001: # 1 mm away from the previous traced pixel
                            # if stamp - last_trace_stamp > 1:
                            # add pixel to the traced pixel trace
                            try:
                                thermal_pixel_trace = np.vstack((thermal_pixel_trace, pixel_coord))
                            except ValueError:
                                thermal_pixel_trace = np.array([pixel_coord])
                            # add the thermal status to the traced pixel trace
                            thermal_trace.append([flame_reading])
                            # add the thermal timestamp
                            thermal_trace_stamp.append([stamp])
                            # add the current torch x
                            thermal_workpiece_x_trace.append([T_table_torch.p[0]])
                            assert len(thermal_pixel_trace) == len(thermal_trace), "Thermal pixel trace and thermal trace length mismatch"
                            # last_trace_stamp = stamp

                            # add thermal status and stamp backward in earlier images (before the welding command)
                            tracing_xy = np.array(pixel_coord)

                            # print("weld point temperature:", flame_reading, "at time", stamp-ir_stamp[0])
                            for (backward_id, move_dxdy, move_stamp) in zip(range(ir_id-1,-1,-1), trace_dxdy[::-1], trace_stamps[::-1]):
                                assert ir_stamp[backward_id+1] == move_stamp, "Trace stamp mismatch"
                                tracing_xy = tracing_xy - move_dxdy
                                tracing_xy_round = np.round(tracing_xy).astype(int)

                                # if pixel within the image
                                if 0+ir_pixel_window_size//2 <= tracing_xy_round[0] < img_width- ir_pixel_window_size//2 and 0 <= tracing_xy_round[1] < img_height:
                                    # this_image = np.rot90(ir_exe[backward_id], k=-1)
                                    this_image = ir_exe[backward_id]
                                    thermal_reading_at_stamp = get_pixel_value(this_image, tracing_xy_round, ir_pixel_window_size)
                                    if np.isnan(thermal_reading_at_stamp):
                                        print("NaN thermal reading at stamp", move_stamp, "for pixel", tracing_xy_round)
                                        input("Press Enter to continue...")
                                    thermal_trace[-1].insert(0, thermal_reading_at_stamp)
                                    thermal_trace_stamp[-1].insert(0, move_stamp)
                                    thermal_workpiece_x_trace[-1].insert(0, thermal_workpiece_x_trace[-1][0])

                        # plt.clf()
                        # # plt.imshow(np.clip(ir_image,7000,15000), cmap='inferno', aspect='equal')
                        # plt.imshow(np.log10(ir_image), cmap='inferno', aspect='equal')
                        # plt.scatter(pixel_coord[0], pixel_coord[1], c='r', s=7, label='Flame centroid')
                        # # plot tracing pixel
                        # cmap_trace = plt.get_cmap('tab10')
                        # for trace_id, trace in enumerate(thermal_pixel_trace):
                        #     if trace_id % 10 == 0:
                        #         # if pixel within the image
                        #         if 0 <= trace[0] < img_width-1 and 0 <= trace[1] < img_height-1:
                        #             plt.scatter(trace[0], trace[1], c=cmap_trace(trace_id % 10), s=10)
                        # plt.colorbar(format='%.2f')
                        # plt.pause(0.1)
                    
                    # print("Centroid mean:", np.mean(thermal_centroid_record, axis=0))
                    # print("Centroid x pixel min max:", np.min(thermal_centroid_record, axis=0)[0], np.max(thermal_centroid_record, axis=0)[0])
                    # print("Centroid y pixel min max:", np.min(thermal_centroid_record, axis=0)[1], np.max(thermal_centroid_record, axis=0)[1])

                    # print("Collected thermal pixel trace:", len(thermal_pixel_trace))
                    thermal_trace_stamp_full = []
                    thermal_workpiece_x_trace_full = []
                    thermal_trace_full = []
                    for (trace_st, trace_x, trace_t) in zip(thermal_trace_stamp, thermal_workpiece_x_trace, thermal_trace):
                        if trace_x[0] <= -35 or trace_x[0] > 45:  # only plot traces with x > 10 mm
                            continue

                        trace_t_smooth = moving_average(trace_t,n=5,padding=True)
                        len(trace_t_smooth) == len(trace_st) == len(trace_x), "Trace length mismatch"
                        thermal_trace_stamp_full.extend(trace_st)
                        thermal_workpiece_x_trace_full.extend(trace_x)
                        # thermal_trace_full.extend(trace_t)
                        thermal_trace_full.extend(trace_t_smooth)

                        # if trace_x[0]>-25:
                        #     # plt.plot(trace_st-ir_stamp[0], trace_t, '-o')
                        #     plt.plot(trace_st-ir_stamp[0], trace_t_smooth, '-o')
                        #     plt.title('Pixel Value vs Time at x='+str(round(trace_x[0],1))+' mm', fontsize=24)
                        #     plt.xlabel('Time (s)', fontsize=18)
                        #     plt.ylabel('Pixel Value (Counts)', fontsize=18)
                        #     plt.xticks(fontsize=16)
                        #     plt.yticks(fontsize=16)
                        #     plt.legend()
                        #     plt.show()

                    # plot_skip = 2
                    # plt.clf()
                    # fig = plt.figure()
                    # ax = plt.axes(projection='3d')
                    # # surf = ax.plot_trisurf(ts_all, pixel_all, counts_all, linewidth=0, antialiased=False, label='-')
                    # surf = ax.plot_trisurf(thermal_trace_stamp_full[::plot_skip]-ir_stamp[0], thermal_workpiece_x_trace_full[::plot_skip], thermal_trace_full[::plot_skip], linewidth=0, antialiased=False, label='-')
                    # # for (trace_st, trace_x, trace_t) in zip(thermal_trace_stamp, thermal_workpiece_x_trace, thermal_trace):
                    # #     if len(trace_st) > 0:
                    # #         ax.plot(trace_st, trace_x, trace_t, linewidth=1, label='-')
                    # plt.title('Pixel Value vs Time')
                    # ax.set_xlabel('Time (s)')
                    # ax.set_ylabel('x pos (mm)')
                    # ax.set_zlabel('Pixel Value (Counts)')
                    # plt.show()

                    # save thermal readings
                    thermal_reading = np.vstack((thermal_stamp,thermal_reading)).T
                    np.savetxt(this_layer_dir+'thermal.csv',thermal_reading,delimiter=',')

                    # save thermal pixel trace
                    trace_dict = {}
                    for (trace_st, trace_x, trace_t) in zip(thermal_trace_stamp, thermal_workpiece_x_trace, thermal_trace):
                        trace_dict[trace_x[0]] = {
                            'time': trace_st,
                            'value': trace_t
                        }
                    with open(this_layer_dir+'thermal_pixel_trace.pickle', 'wb') as f:
                        pickle.dump(trace_dict, f)

                ################ get speed ##############
                print("Getting speed...")
                try:
                    weld_relative_exe = np.loadtxt(this_layer_dir+'weld_relative_exe',delimiter=',')
                    weld_relative_v_exe = np.loadtxt(this_layer_dir+'weld_relative_v_exe.csv',delimiter=',')
                except FileNotFoundError:
                    weld_relative_exe = []
                    weld_relative_v_exe = []
                    for i in range(len(weld_js_exe)):
                        t1_world = robot_weld.fwd(weld_js_exe[i,1:7])
                        t2_world = positioner.fwd(weld_js_exe[i,7:],world=True)
                        t1_t2 = t2_world.inv()*t1_world
                        weld_relative_exe.append(t1_t2.p)
                    weld_relative_exe = np.array(weld_relative_exe)
                    weld_relative_v_exe=np.linalg.norm(np.diff(weld_relative_exe,axis=0),2,1)/np.diff(weld_js_exe[:,0])
                    weld_relative_v_exe=np.append(weld_relative_v_exe[0],weld_relative_v_exe)
                    weld_relative_v_exe=moving_average(weld_relative_v_exe,padding=True)
                    weld_relative_v_exe=moving_average(weld_relative_v_exe,padding=True)
                    np.savetxt(this_layer_dir+'weld_relative_exe.csv',weld_relative_exe,delimiter=',')
                    np.savetxt(this_layer_dir+'weld_relative_v_exe.csv',weld_relative_v_exe,delimiter=',')
                #############################################
                
                ############### get height and width ##############
                print("Getting height and width...")
                try:
                    profile_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',')
                    profile_width = np.loadtxt(this_layer_dir+'profile_width.csv',delimiter=',')
                    all_profile_height.append(profile_height)
                    # pcd = o3d.io.read_point_cloud(this_layer_dir+'pcd.pcd')
                    pcd_denoise = o3d.io.read_point_cloud(this_layer_dir+'pcd_denoise.pcd')

                    # Transz0_H = deepcopy(Transz0_H_even) if layer_n_id % 2 == 0 else deepcopy(Transz0_H_odd)
                    # pcd_denoise_trans = deepcopy(pcd_denoise)
                    # pcd_denoise_trans.transform(Transz0_H)
                    # all_pcd_transform.append(pcd_denoise_trans)

                except FileNotFoundError:
                    # processing the scans
                    scan_process = ScanProcess(robot_scan,positioner)

                    # Single scan 2D reconstruction
                    try:
                        with open(this_layer_dir+'scan_exe_noise_remove.pickle', 'rb') as f:
                            scan_exe_noise_remove = pickle.load(f)
                    except FileNotFoundError:
                        scan_exe_noise_remove = []
                        duration_list = []
                        for (weld_js,scan) in zip(scan_js_exe,scan_exe):
                            st = time.time()
                            scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                            scan_exe_noise_remove.append(scan_noise_remove)
                            duration_list.append(time.time()-st)
                        # plt.plot(duration_list)
                        # plt.show()
                        # print("Average single scan 2D reconstruction time:",np.mean(duration_list))
                        # print("Max single scan 2D reconstruction time:",np.max(duration_list))
                        with open(this_layer_dir+'scan_exe_noise_remove.pickle', 'wb') as f:
                            pickle.dump(scan_exe_noise_remove, f)

                    # whole layer 3D reconstruction
                    pcd=None
                    # pcd = scan_process.pcd_register_mti(scan_exe,scan_js_exe[:,:6],robot_stamps,static_positioner_q=positioner_joints,flip=True,scanner='fuji')
                    pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,scan_js_exe[:,1:],robot_stamps,flip=True,scanner='fuji')
                    # visualize_pcd([pcd])
                    # move pcd_noise_preremoved in y direction
                    # pcd_noise_preremoved = pcd_noise_preremoved.translate((0,200,0))
                    # visualize_pcd([pcd,pcd_noise_preremoved])

                    # cropping the point cloud
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    if scanner_lagging:
                        z_height_start=curve_planned_z+0.1
                    else:
                        z_height_start=curve_planned_z-5
                        if layer_n == layer_nums[-1] and weld_parts == 'layer':
                            curve_prev = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_nums[layer_n_id-1]}_0.csv',delimiter=',')
                            curve_prev_z = np.mean(curve_prev[:,2])
                            z_height_start = curve_prev_z
                    # z_height_start = 0
                    # print(z_height_start)
                    # print(curve_y)
                    crop_extend_x=10
                    crop_extend_z=20
                    crop_min=(curve_x_end-crop_extend_x,curve_y-30,-30)
                    crop_max=(curve_x_start+crop_extend_x,curve_y+30,z_height_start+crop_extend_z)
                    crop_h_min=(curve_x_end-crop_extend_x,curve_y-20,-30)
                    crop_h_max=(curve_x_start+crop_extend_x,curve_y+20,z_height_start+crop_extend_z)
                    # profile_height_noise, profile_width_noise,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    pcd = scan_process.pcd_noise_remove(pcd,min_bound=crop_min,max_bound=crop_max,outlier_remove=False,cluster_based_outlier_remove=False)
                    pcd_denoise = scan_process.pcd_noise_remove(pcd,crop_flag=False,outlier_remove=False,nb_neighbors=40,std_ratio=1.5,min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
                    # visualize_pcd([pcd])
                    # Transz0_H = None
                    Transz0_H = deepcopy(Transz0_H_even) if layer_n_id % 2 == 0 else deepcopy(Transz0_H_odd)
                    if last_profile_height is None:
                        profile_height, _,Transz0_H = scan_process.pcd2height(deepcopy(pcd_denoise),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                        _, profile_width,_ = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    else:
                        profile_height, _,Transz0_H = scan_process.pcd2height(deepcopy(pcd_denoise),last_profile_height,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                        _, profile_width,_ = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    
                    # for visualization
                    pcd_denoise_trans = deepcopy(pcd_denoise)
                    pcd_denoise_trans.transform(Transz0_H)
                    all_pcd_transform.append(pcd_denoise_trans)
                    all_profile_height.append(profile_height)

                    if scanner_lagging:
                        Transz0_H_even = deepcopy(Transz0_H)
                        Transz0_H_odd = deepcopy(Transz0_H)
                    elif create_transform:
                        if layer_n_id % 2 == 0:
                            Transz0_H_even = deepcopy(Transz0_H)
                            np.savetxt(logdata_dir+'Transz0_H_even.csv',Transz0_H_even,delimiter=',')
                        else:
                            Transz0_H_odd = deepcopy(Transz0_H)
                            if layer_n_id == 1 and weld_parts == 'base':
                                Transz0_H_odd[0,-1] -= 4
                                Transz0_H_odd[1,-1] += 1
                            np.savetxt(logdata_dir+'Transz0_H_odd.csv',Transz0_H_odd,delimiter=',')
                        if 'control' in logdata_dir_name and layer_n_id == 1 and weld_parts == 'layer':
                            print("Transforming pcd using icp")
                            threshold = 1
                            for pcd_i in [-2,-1]:
                                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,0),max_bound=(1e5,1e5,1e5))
                                all_pcd_transform[pcd_i]=all_pcd_transform[pcd_i].crop(bbox)
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                        all_pcd_transform[-1], all_pcd_transform[-2], threshold, np.eye(4),
                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                            np.savetxt(logdata_dir+'Trans_icp_odd2even.csv',reg_p2p.transformation,delimiter=',')

                    # visualize_pcd([pcd_denoise_trans])
                    # plt.plot(profile_height[:,0],profile_height[:,1],'-o')
                    # plt.show()
                    # if len(all_pcd_transform) != 0:
                    #     cmap = plt.get_cmap('jet')
                    #     color = cmap(np.linspace(0, 1, len(all_pcd_transform)))
                    #     for i in range(len(all_pcd_transform)):
                    #         all_pcd_transform[i].paint_uniform_color(color[i][:3])
                    #     visualize_pcd(all_pcd_transform)

                    # apply 1D smoother to profile_width
                    profile_width[:,1] = np.convolve(profile_width[:,1], np.ones(5)/5, mode='same')
                    # profile_width_noise[:,1] = np.convolve(profile_width_noise[:,1], np.ones(5)/5, mode='same')

                    # save processed profile height and point cloud
                    np.savetxt(this_layer_dir+'profile_height.csv',profile_height,delimiter=',')
                    np.savetxt(this_layer_dir+'profile_width.csv',profile_width,delimiter=',')
                    o3d.io.write_point_cloud(this_layer_dir+'pcd.pcd',pcd)
                    o3d.io.write_point_cloud(this_layer_dir+'pcd_denoise.pcd',pcd_denoise)
                    #############################################

                # compensating the observed shifting
                shift_x = 5.685
                profile_height[:,0] = profile_height[:,0] + shift_x
                profile_width[:,0] = profile_width[:,0] + shift_x

                ################ combine everything in one array ##############
                if not scanner_lagging:
                    if layer_n_id == len(layer_nums)-1 and weld_parts == 'layer':
                        # if the scanner is leading, the last layer is not welding anything.
                        break
                    try:
                        next_scan_height = np.loadtxt(next_layer_dir+'profile_height.csv',delimiter=',')
                        next_scan_width = np.loadtxt(next_layer_dir+'profile_width.csv',delimiter=',')
                    except FileNotFoundError:
                        next_scan_height = deepcopy(profile_height)
                        next_scan_width = deepcopy(profile_width)
                        run_code_again_flag = True

                profile_welding = []
                for js_id,x in enumerate(weld_relative_exe[:,0]):
                # for x_id, x in enumerate(profile_height[:,0]):
                    # find closest x in weld_relative_exe
                    # js_id = np.argmin(np.abs(weld_relative_exe[:,0]-x))
                    if np.min(np.abs(profile_height[:,0]-x)) > 0.3:
                        continue

                    # time at the same x
                    this_t = weld_js_exe[js_id,0]
                    # weld command right before this time
                    cmd_idx = np.where(weld_cmd[:,0]<=this_t)[0]
                    if len(cmd_idx) == 0:
                        cmd_idx = 0
                    else:
                        cmd_idx = cmd_idx[-1]
                    this_cmd_v = weld_cmd[cmd_idx,2]
                    this_cmd_fr = weld_cmd[cmd_idx,3]
                    # velocity at the same x
                    this_v = weld_relative_v_exe[js_id]
                    # height and width at the same x
                    if scanner_lagging:
                        this_height = profile_height[np.argmin(np.abs(profile_height[:,0]-x)),1]
                        if last_profile_height is not None:
                            last_height = last_profile_height[np.argmin(np.abs(last_profile_height[:,0]-x)),1]
                        else:
                            last_height = 0
                    else:
                        this_height = next_scan_height[np.argmin(np.abs(next_scan_height[:,0]-x)),1]
                        last_height = profile_height[np.argmin(np.abs(profile_height[:,0]-x)),1]
                    this_dh = this_height - last_height                    
                    this_width = profile_width[np.argmin(np.abs(profile_width[:,0]-x)),1] if scanner_lagging else next_scan_width[np.argmin(np.abs(next_scan_width[:,0]-x)),1]
                    
                    # torch height
                    torch_height = weld_relative_exe[js_id,2] - last_height
                    # welding status at time t
                    welding_status_idx=np.where(welding_status[:,0]>=this_t)[0]
                    welding_status_idx = -1 if len(welding_status_idx) == 0 else welding_status_idx[0]
                    ratio=(this_t-welding_status[:,0][welding_status_idx-1])/(welding_status[:,0][welding_status_idx]-welding_status[:,0][welding_status_idx-1])
                    this_welding_status=welding_status[:,1:][welding_status_idx-1]*(1-ratio)+welding_status[:,1:][welding_status_idx]*ratio

                    # thermal reading at time t
                    thermal_reading_idx = np.where(thermal_reading[:,0]>=this_t)[0]
                    thermal_reading_idx = -1 if len(thermal_reading_idx) == 0 else thermal_reading_idx[0]
                    ratio=(this_t-thermal_reading[:,0][thermal_reading_idx-1])/(thermal_reading[:,0][thermal_reading_idx]-thermal_reading[:,0][thermal_reading_idx-1])
                    this_thermal_reading=thermal_reading[:,1][thermal_reading_idx-1]*(1-ratio)+thermal_reading[:,1][thermal_reading_idx]*ratio

                    this_welding_profile = np.array([this_t,x,this_cmd_v,this_cmd_fr,this_height,this_dh,torch_height,this_width,this_v,this_thermal_reading])
                    this_welding_profile = np.append(this_welding_profile,this_welding_status)
                    profile_welding.append(this_welding_profile)
                
                profile_welding = np.array(profile_welding)
                # save profile welding with header
                header = 'time,x,cmd_v,cmd_feedrate,height,dheight,torch_height,width,v,thermal,voltage,current,feedrate,energy'
                np.savetxt(this_layer_dir+'profile_welding.csv',profile_welding,delimiter=',',header=header)
                last_profile_height = profile_height

                print("Finished processing layer:",layer_name)
                print("=====================================")

        # fig, ax = plt.subplots()
        # ax.set_title('Profile height')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # for i in range(len(all_profile_height)):
        #     plot_indeces = all_profile_height[i][:,1]>np.mean(all_profile_height[i][:,1])-5
        #     ax.plot(all_profile_height[i][plot_indeces,0],all_profile_height[i][plot_indeces,1],label='Layer '+str(i))
        # # ax.legend()
        # plt.show()

        # if len(all_pcd_transform) != 0:
        #     cmap = plt.get_cmap('tab10')
        #     for i in range(len(all_pcd_transform)):
        #         all_pcd_transform[i].paint_uniform_color(cmap(i%10)[:3])
        #     visualize_pcd(all_pcd_transform)
        #     animation_mesh(all_pcd_transform)
        

    if run_code_again_flag:
        print("********** You need to run the code again **********")

if __name__ == '__main__':
    main()