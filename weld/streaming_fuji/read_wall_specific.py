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

torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

def main():

    test_current = False
    test_thermal = False
    test_geometry = True

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
    
    ############## choose data directory ##############
    data_dir = '../../data/wall_weld_test/'
    logdata_dir_name = 'weld_fujiscan_2025_06_11_16_27_41/'
    logdata_dir = data_dir+logdata_dir_name

    ##### layer, basic infos ####
    last_layer_n = 117
    layer_n = 129
    layer_name = 'layer'+str(layer_n)
    last_layer_name = 'layer'+str(last_layer_n)
    this_layer_dir = logdata_dir+layer_name+'/'
    last_layer_dir = logdata_dir+last_layer_name+'/'
    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')
    rob_js_exe = np.loadtxt(this_layer_dir+'weld_js_exe.csv',delimiter=',')
    rob2_js_exe = rob_js_exe[:,7:13] # robot 2 js exe
    robot_stamps = rob_js_exe[:,0] # robot stamps

    ###### viz current #####
    if test_current:
        welding_current_exe = np.loadtxt(this_layer_dir+'current.csv',delimiter=',',skiprows=1)
        welding_fronius = np.loadtxt(this_layer_dir+'welding.csv',delimiter=',',skiprows=1)
        # print(welding_current_exe)
        plt.plot(welding_current_exe[:,0]-welding_current_exe[0,0], welding_current_exe[:,1], label='Welding Current')
        plt.plot(welding_fronius[:,0]-welding_current_exe[0,0], welding_fronius[:,2], label='Fronius Current')
        plt.title('Welding Current vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Welding Current (A)')
        plt.grid()
        plt.show()

    ###### viz thermal #####
    if test_thermal:
        with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
            ir_exe = pickle.load(f)
        ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')

        horizontal_offset=0
        vertical_offset=3
        ir_pixel_window_size=5
        thermal_pixel_trace = np.array([])
        thermal_surface_pixel_trace = np.array([])
        trace_dxdy = []
        trace_stamps = []
        thermal_workpiece_x_trace = []
        thermal_trace_surface = []
        thermal_trace = []
        thermal_trace_stamp = []
        thermal_centroid_record = []
        cam_pixel_moving_ratio = 1.77 # 1.77 pixel per mm
        last_T_thermal_cam = None
        for (ir_id,ir_image_raw, stamp) in zip(range(len(ir_exe)), ir_exe, ir_stamp):
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
                moving_dy = rob_translation[1] * cam_pixel_moving_ratio
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

            plt.clf()
            # plt.imshow(np.clip(ir_image,7000,10000), cmap='hot', aspect='equal')
            plt.imshow(np.log10(ir_image), cmap='hot', aspect='equal')
            # plt.imshow(ir_image, cmap='hot', aspect='equal')
            plt.scatter(pixel_coord[0], pixel_coord[1], c='r', s=7, label='Flame centroid')
            # plot tracing pixel
            # cmap_trace = plt.get_cmap('tab10')
            # for trace_id, trace in enumerate(thermal_pixel_trace):
            #     if trace_id % 4 == 0:
            #         # if pixel within the image
            #         if 0 <= trace[0] < img_width-1 and 0 <= trace[1] < img_height-1:
            #             plt.scatter(trace[0], trace[1], c=cmap_trace(trace_id % 10), s=10)
            plt.scatter(thermal_pixel_trace[::4,0], thermal_pixel_trace[::4,1], c='b', s=5, label='Traced pixels')
            # plt.colorbar(format='%.2f')
            plt.pause(0.000001)

            # input('')

        thermal_trace_stamp_full = []
        thermal_workpiece_x_trace_full = []
        thermal_trace_full = []
        for (trace_st, trace_x, trace_t) in zip(thermal_trace_stamp, thermal_workpiece_x_trace, thermal_trace):
            # if trace_x[0] <= -35 or trace_x[0] > 45:  # only plot traces with x > 10 mm
            #     continue

            trace_t_smooth = moving_average(trace_t,n=5,padding=True)
            len(trace_t_smooth) == len(trace_st) == len(trace_x), "Trace length mismatch"
            thermal_trace_stamp_full.extend(trace_st)
            thermal_workpiece_x_trace_full.extend(trace_x)
            # thermal_trace_full.extend(trace_t)
            thermal_trace_full.extend(trace_t_smooth)
        thermal_trace_stamp_full = np.array(thermal_trace_stamp_full)
        thermal_workpiece_x_trace_full = np.array(thermal_workpiece_x_trace_full)
        thermal_trace_full = np.array(thermal_trace_full)

        # find the mid number of thermal_trace_stamp_full
        thermal_trace_stamp_full_sort = np.sort(thermal_trace_stamp_full)
        mid_stamp = thermal_trace_stamp_full_sort[len(thermal_trace_stamp_full_sort)//2]
        plt.plot(thermal_workpiece_x_trace_full[thermal_trace_stamp_full==mid_stamp], thermal_trace_full[thermal_trace_stamp_full==mid_stamp], '-o')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Pixel Value (Counts)')
        plt.title('Thermal Trace at Timestamps around {:.2f} s'.format(mid_stamp-ir_stamp[0]))
        plt.grid()
        plt.legend()
        plt.show()

        plot_skip = 2
        plt.clf()
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # surf = ax.plot_trisurf(ts_all, pixel_all, counts_all, linewidth=0, antialiased=False, label='-')
        surf = ax.plot_trisurf(thermal_trace_stamp_full[::plot_skip]-ir_stamp[0], thermal_workpiece_x_trace_full[::plot_skip], thermal_trace_full[::plot_skip], linewidth=0, antialiased=False, label='-')
        # for (trace_st, trace_x, trace_t) in zip(thermal_trace_stamp, thermal_workpiece_x_trace, thermal_trace):
        #     if len(trace_st) > 0:
        #         ax.plot(trace_st, trace_x, trace_t, linewidth=1, label='-')
        plt.title('Pixel Value vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('x pos (mm)')
        ax.set_zlabel('Pixel Value (Counts)')
        plt.show()

    if test_geometry:
        # profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
        # last_profile_welding = np.loadtxt(last_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
        # plt.plot(profile_welding[:,1], profile_welding[:,4], '-o', label='Welding Height')
        # plt.plot(last_profile_welding[:,1], last_profile_welding[:,4], '-o', label='Last Welding Height')
        # plt.grid()
        # plt.show()

        edge_exclude = 7.5 # mm
        start_x = -60 # mm
        end_x = 50
        sample_rate = 30
        shift_x = 5.685

        dh_sample = []
        width_sample = []

        test_dir = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']
        # test_dir = ['weld_fujiscan_2025_06_11_14_12_44/','weld_fujiscan_2025_06_11_16_27_41/']
        # test_dir = ['weld_fujiscan_2025_06_11_16_52_36/']
        
        
        color_viz = []
        color_map = plt.get_cmap('tab10')
        for dir_cnt,logdata_dir_name in enumerate(test_dir):
            ## data to visualize
            profile_welding_viz = []
            height_viz = []

            ## directory to process
            print(f"Processing directory: {logdata_dir_name}")
            logdata_dir = data_dir + logdata_dir_name
            total_layers_name = glob.glob(logdata_dir+'layer*')
            # get printed layer number
            layer_nums = []
            for layer_name in total_layers_name:
                this_layer = layer_name.split('\\')[-1]
                this_layer = this_layer.split('r')[-1]
                layer_nums.append(int(this_layer))
            layer_nums = np.sort(layer_nums)
            for layer_n_id, layer_n in enumerate(layer_nums):
                print(f"Processing layer {layer_n} ({layer_n_id+1}/{len(layer_nums)})")
                # if layer_n_id % 3 != 0:
                #     print(f"Skipping layer {layer_n} due to odd index")
                #     continue
                # if layer_n != 129:
                #     continue
                this_layer_dir = logdata_dir + 'layer' + str(layer_n) + '/'

                profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
                profile_welding_viz.append(profile_welding[:,[1,4]])
                # # exclude the first and last edge_exclude mm of the profile
                # if profile_welding[0,1] < profile_welding[-1,1]:
                #     profile_welding = profile_welding[profile_welding[:,1] >= start_x + edge_exclude]
                #     profile_welding = profile_welding[profile_welding[:,1] <= end_x - edge_exclude]
                # else:
                #     profile_welding = profile_welding[profile_welding[:,1] <= end_x - edge_exclude]
                #     profile_welding = profile_welding[profile_welding[:,1] >= start_x + edge_exclude]
                
                # try:
                #     timestamps_sample = np.arange(profile_welding[0,0], profile_welding[-1,0], 1/sample_rate)
                # except IndexError:
                #     print(f"Skipping layer {layer_n} due to empty profile_welding")
                #     continue
                # for stamp_i, stamp in enumerate(timestamps_sample):
                #     # find the closest timestamp in profile_welding smaller than the current timestamp
                #     time_id_last = np.where(profile_welding[:,0] <= stamp)[0][-1]+1
                #     time_id_first = np.where(profile_welding[:,0] > stamp-1/sample_rate)[0][0]

                #     if time_id_first >= time_id_last:
                #         print(f"Skipping timestamp {stamp} at layer {layer_n} due to no valid data")
                #         continue

                #     dh_sample.append(np.mean(profile_welding[time_id_first:time_id_last,5]))
                #     width_sample.append(np.mean(profile_welding[time_id_first:time_id_last,7]))
                
                ## height in lambda
                this_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',',skiprows=1)
                height_viz.append(this_height)
                color_viz.append(color_map(dir_cnt/len(test_dir)))
            
            # visualize the height
            for profile_cnt,height_profile in enumerate(height_viz):
                # plt.plot(height_profile[:,0], height_profile[:,1]+40, '-o', color=color_viz[profile_cnt], label='Height Profile')
                plt.plot(profile_welding_viz[profile_cnt][:,0], profile_welding_viz[profile_cnt][:,1], 'o', label='Welding Profile')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Height (mm)')
            plt.grid()
            plt.show()

        dh_sample = np.array(dh_sample)
        width_sample = np.array(width_sample)
        print(f"dh_sample: {dh_sample.shape}, width_sample: {width_sample.shape}")

        dh_width_sample = np.vstack((dh_sample, width_sample)).T
        dh_width_sample_flatten = dh_width_sample.flatten()

        # dh_width_sample_flatten = deepcopy(width_sample)

        plt.clf()
        for k in [120]:
            output_dim = 2 # dh and width
            T = len(dh_width_sample_flatten)//output_dim
            # k=120
            num_rows = k*output_dim # the "k"
            num_cols = T - k + 1
            # np.array([a[i:i+cols*2:2] for i in range(rows)])
            big_Y = np.array([dh_width_sample_flatten[i:i+num_cols*output_dim:output_dim] for i in range(num_rows)]) # the Hankel matrix
            # for i in range(len(dh_width_sample_flatten)//2):
            #     big_Y.append(np.roll(dh_width_sample_flatten, i*2))
            # big_Y = np.array(big_Y).T
            print(f"big_Y shape: {big_Y.shape}")

            U, s, Vt = np.linalg.svd(big_Y, full_matrices=False)
            s_sort = np.sort(s)[::-1]
            print("10 largest singular values:", s_sort[:10])
            # plot singular values
            # plt.clf()
            plt.plot(np.log10(s/s[0]), '-o')
        # use latex equation
        # plt.title(f'SVD Singular Values ($\Delta h$)')
        # plt.title(f'SVD Singular Values (width)')
        plt.title(f'SVD Singular Values ($\Delta h$ and width)')
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log10 scale)')
        plt.grid()
        plt.show()

        # plt.plot(profile_welding[:,0]-profile_welding[0,0], profile_welding[:,4], '-o', label='Welding Height')
        # plt.show()

if __name__ == "__main__":
    
    main()