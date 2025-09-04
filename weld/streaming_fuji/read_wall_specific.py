import time, os, copy, sys, yaml, inspect
import glob
from copy import deepcopy
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from matplotlib import pyplot as plt
import open3d as o3d
import cv2 as cv
from motoman_def import *
from robotics_utils import *
from flir_toolbox import *
from ultralytics import YOLO

from controlModelFunction import *
sys.path.append('../../scan/scan_process/')
sys.path.append('../../scan/scan_tools/')
from scan_utils import *
from scanProcess import *
from animation_3d import *

def grid_from_data(log_v, log_omega, n=30, pad=0.05):
    """
    Build a meshgrid spanning the data range with a small padding.
    """
    xv_min, xv_max = np.min(log_v), np.max(log_v)
    xo_min, xo_max = np.min(log_omega), np.max(log_omega)
    # small padding to avoid tight cropping
    dxv = (xv_max - xv_min) or 1.0
    dxo = (xo_max - xo_min) or 1.0
    xv = np.linspace(xv_min - pad*dxv, xv_max + pad*dxv, n)
    xo = np.linspace(xo_min - pad*dxo, xo_max + pad*dxo, n)
    Xv, Xo = np.meshgrid(xv, xo)
    return Xv, Xo

def get_weld_shift_x(profile_height):
    profile_x = np.arange(np.min(profile_height[:,0]), np.max(profile_height[:,0])+0.1, 0.1)
    height_approx_func = CubicSpline(profile_height[:,0], profile_height[:,1])
    profile_height_aug = np.column_stack((profile_x, height_approx_func(profile_x)))

    reference_height = 3.5
    profile_height_closed_arg = np.argsort(np.abs(profile_height_aug[:,1]-reference_height))
    left_x = None
    right_x = None
    for profile_idx in profile_height_closed_arg:
        if profile_height_aug[profile_idx,0]<0 and left_x is None:
            left_x = profile_height_aug[profile_idx,0]
        if profile_height_aug[profile_idx,0]>0 and right_x is None:
            right_x = profile_height_aug[profile_idx,0]
        if left_x is not None and right_x is not None:
            break
    shift_x = -1*(left_x+right_x)/2

    # # visualize the height
    # plt.figure(figsize=(16, 5))
    # plt.plot(profile_height_aug[:, 0], profile_height_aug[:, 1], '-o', label='Profile Height')
    # # draw a vertical line at left_x and right_x
    # plt.axvline(x=left_x, color='r', linestyle='--', label='Left Shift Point')
    # plt.axvline(x=right_x, color='g', linestyle='--', label='Right Shift Point')
    # plt.title('Profile Height Visualization')
    # plt.xlabel('X Position (mm)')
    # plt.ylabel('Height (mm)')
    # plt.legend()
    # plt.grid()
    # plt.show()
    
    return shift_x

def get_sum_profile(profile,sample_id):

    profile_sum = np.concatenate(([0.0],np.cumsum(profile, dtype=np.float64)))
    starts, ends = sample_id[:-1], sample_id[1:]
    sums = profile_sum[ends] - profile_sum[starts]
    counts = ends - starts
    means = sums / counts
    return means

# for plotting
xy_label_size = 18
xy_tick_size = 16
legend_size = 16
title_size = 20
sup_title_size = 20

torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")
tip_wire_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/tip_wire.pt")

def main():

    test_current = False
    test_thermal = False
    test_thermal_collected = False
    test_pcd = False
    test_geometry = False
    test_weld_shift = False
    get_statistics = False
    test_loglog = False
    test_read_thermal = False
    reverse_thermal_pixel_trace = True

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
    logdata_dir_name = 'weld_fujiscan_2025_06_12_15_33_03/'
    logdata_dir = data_dir+logdata_dir_name

    ##### layer, basic infos ####
    last_layer_n = 293
    layer_n = 311
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

            
            # find the thermal distribution at the same z level
            thermal_dist_torch_pixel_x = np.arange(0, img_width, 1)
            thermal_dist_torch_table_x = -1*(thermal_dist_torch_pixel_x-pixel_coord[0])*(1/cam_pixel_moving_ratio) + T_table_torch.p[0]
            # only keep the pixels with x >= -55 mm, <= 55 mm
            thermal_dist_torch_pixel_x = thermal_dist_torch_pixel_x[(thermal_dist_torch_table_x >= -55)&(thermal_dist_torch_table_x <= 55)]
            thermal_dist_torch_pixel = np.vstack((thermal_dist_torch_pixel_x, np.full_like(thermal_dist_torch_pixel_x, pixel_coord[1]))).T

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
            # plt.scatter(thermal_pixel_trace[::4,0], thermal_pixel_trace[::4,1], c='b', s=5, label='Traced pixels')
            plt.scatter(thermal_dist_torch_pixel[:,0], thermal_dist_torch_pixel[:,1], c='b', s=5, label='Thermal distribution at torch x')
            # plt.colorbar(format='%.2f')
            plt.pause(0.000001)

            input('')

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

    ###### viz thermal collected ####
    if test_thermal_collected:
        profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
        with open(this_layer_dir+'thermal_pixel_trace.pickle', 'rb') as f:
            thermal_dist = pickle.load(f)
        
        thermal_sample_x = thermal_dist.keys()
        thermal_sample_t = []
        thermal_sample_x_full = []
        thermal_sample_t_full = []
        thermal_reading_full = []
        for thermal_x in thermal_sample_x:
            thermal_sample_x_full.extend(np.repeat(thermal_x, len(thermal_dist[thermal_x]['time'])))
            thermal_sample_t_full.extend(thermal_dist[thermal_x]['time'])
            thermal_reading_full.extend(thermal_dist[thermal_x]['value'])
            thermal_sample_t.extend(np.setdiff1d(thermal_dist[thermal_x]['time'], thermal_sample_t))
        thermal_sample_x_full = np.array(thermal_sample_x_full)
        thermal_sample_t_full = np.array(thermal_sample_t_full)
        thermal_reading_full = np.array(thermal_reading_full)
        thermal_sample_t = np.sort(np.array(thermal_sample_t))

        
        # Define radial basis functions (Gaussians)
        centers = np.linspace(-60, 60, 20)  # 10 basis functions
        width = np.diff(centers)[0]/np.sqrt(2*np.log(2))  # width of each RBF
        rbf = lambda x, c, w: np.exp(-((x - c) ** 2) / (2 * w ** 2))

        for vix_i,viz_t in enumerate(thermal_sample_t):
            if viz_t < profile_welding[0,0]:
                continue

            # find the closest timestamp in profile_welding
            time_id = np.where(profile_welding[:,0] <= viz_t)[0][-1]
            profile_welding_x = profile_welding[time_id,1]
            # find all thermal readings and x
            this_x = thermal_sample_x_full[thermal_sample_t_full == viz_t]
            this_reading = thermal_reading_full[thermal_sample_t_full == viz_t]
            x_sorted_id = np.argsort(this_x)
            this_x = this_x[x_sorted_id]
            this_reading = this_reading[x_sorted_id]
            # viz in inertial frame
            this_x = this_x - profile_welding_x
            
            this_reading = this_reading[(this_x >= -55) & (this_x <= 55)]  # limit to the range of interest
            this_x = this_x[(this_x >= -55) & (this_x <= 55)]  # limit to the range of interest

            # Solve least squares to find projection coefficients
            # Construct RBF matrix
            Phi = np.array([rbf(this_x, c, width) for c in centers]).T  # shape: (len(x), num_centers)
            coeffs, _, _, _ = np.linalg.lstsq(Phi, this_reading, rcond=None)

            # Reconstruct the projected signal
            projection = Phi @ coeffs

            plt.clf()
            plt.plot(this_x, this_reading, 'o')
            plt.plot(this_x, projection, label="RBF Projection", linestyle="--")
            plt.title(f'Thermal Reading at {viz_t-profile_welding[0,0]:.2f} s')
            plt.xlim(-57, 57)
            plt.ylim(8000, 26500)
            plt.xlabel('X Position (mm)')
            plt.ylabel('Pixel Value (Counts)')
            plt.grid()
            plt.pause(0.1)

    ###### test shift detection ####
    if test_weld_shift:
        # test_dir = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
        #                'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
        #                'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/',\
        #                 'weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
        # test_dir = ['weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
        test_dir = ['weld_fujicontrol_2025_08_14_11_19_59/', 'weld_fujicontrol_2025_08_13_14_57_52/']
        shift_x_all = []
        for dir_cnt,logdata_dir_name in enumerate(test_dir):
            ## data to visualize
            profile_welding_viz = []
            height_viz = []

            ## directory to process
            print(f"Processing directory: {logdata_dir_name}")
            logdata_dir = data_dir + logdata_dir_name
            total_layers_name = glob.glob(logdata_dir+'baselayer*')
            # get printed layer number
            layer_nums = []
            for layer_name in total_layers_name:
                this_layer = layer_name.split('\\')[-1]
                this_layer = this_layer.split('r')[-1]
                layer_nums.append(int(this_layer))
            layer_nums = np.sort(layer_nums)
            show_pcd_list = []
            for layer_n_id, layer_n in enumerate(layer_nums):
                this_layer_dir = logdata_dir + 'baselayer' + str(layer_n) + '/'

                profile_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',')
                profile_x = np.arange(np.min(profile_height[:,0]), np.max(profile_height[:,0])+0.1, 0.1)
                # profile_height_aug = np.interp(profile_x, profile_height[:,0], profile_height[:,1])
                # profile_height_aug = np.column_stack((profile_x, profile_height_aug))
                height_approx_func = CubicSpline(profile_height[:,0], profile_height[:,1])
                profile_height_aug = np.column_stack((profile_x, height_approx_func(profile_x)))
                # plt.plot(profile_height_aug[:,0], profile_height_aug[:,1], '-o', label=f'Layer {layer_n}')
                # plt.show()

                height_viz.append(profile_height)

                if layer_n_id == 1:
                    # scan_N = 200
                    # span_N = 5
                    # threshold = 0.1
                    # diff_points_1 = []
                    # height_diff = np.diff(profile_height_aug[:,1])
                    # for point_i, point in enumerate(profile_height_aug[0:scan_N+1]):
                    #     diff_right = np.mean(height_diff[point_i:point_i+span_N])

                    #     diff_points_1.append(diff_right)
                    # # find the first diff points > 0.1
                    # plt.plot(profile_height_aug[:,0], profile_height_aug[:,1], '-o', label=f'Layer {layer_n}')
                    # plt.show()
                    # plt.plot(diff_points_1, '-o', label='Error Points 1')
                    # plt.show()
                    # left_point = np.argwhere(np.array(diff_points_1) > threshold).flatten()[0]+int(span_N/2)
                    # diff_points_2 = []
                    # for point_i, point in enumerate(profile_height_aug[::-1][0:scan_N+1]):
                    #     diff_right = np.mean(height_diff[::-1][point_i:point_i+span_N])
                    #     diff_points_2.append(diff_right)
                    # # find the first diff points < -0.1
                    # right_point = np.argwhere(np.array(diff_points_2) < -threshold).flatten()[0]+int(span_N/2)

                    # left_x = np.mean(profile_height_aug[left_point:left_point+2, 0])
                    # right_x = np.mean(profile_height_aug[::-1][right_point:right_point+2, 0])
                    # shift_x = -1*(left_x+right_x)/2
                    # shift_x_all.append(shift_x)

                    # print(f"Left point: {left_x:.2f}, Right point: {right_x:.2f}")
                    # # plt.plot(diff_points_1, '-o', label='Error Points 1')
                    # # plt.plot(diff_points_2, '-o', label='Error Points 2')
                    # # plt.show()
                    # # plt.figure(figsize=(16, 5))
                    # # plt.plot(profile_height[:,0],profile_height[:,1], '-o', label=f'Layer {layer_n}')
                    # # plt.plot(profile_height_aug[:,0],profile_height_aug[:,1], '--', label=f'Layer {layer_n} (Augmented)')
                    # # plt.axvline(x=left_x, color='r', linestyle='--', label='Left Shift Point')
                    # # plt.axvline(x=right_x, color='g', linestyle='--', label='Right Shift Point')
                    # # plt.xlabel('X Position (mm)')
                    # # plt.ylabel('Height (mm)')
                    # # plt.title(f'Profile Height - Layer {layer_n}')
                    # # plt.legend()
                    # # plt.grid()
                    # # plt.show()

                    profile_height_closed_arg = np.argsort(np.abs(profile_height_aug[:,1]-3.5))
                    left_x = None
                    right_x = None
                    for profile_idx in profile_height_closed_arg:
                        if profile_height_aug[profile_idx,0]<0 and left_x is None:
                            left_x = profile_height_aug[profile_idx,0]
                        if profile_height_aug[profile_idx,0]>0 and right_x is None:
                            right_x = profile_height_aug[profile_idx,0]
                        if left_x is not None and right_x is not None:
                            break
                    shift_x = -1*(left_x+right_x)/2
                    shift_x_all.append(shift_x)

            # visualize the height
            plt.figure(figsize=(16, 5))
            for i, h in enumerate(height_viz):
                plt.plot(h[:, 0], h[:, 1], '-o', label=f'Layer {layer_nums[i]}')
            # draw a vertical line at left_x and right_x
            plt.axvline(x=left_x, color='r', linestyle='--', label='Left Shift Point')
            plt.axvline(x=right_x, color='g', linestyle='--', label='Right Shift Point')
            plt.title('Profile Height Visualization')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Height (mm)')
            plt.legend()
            plt.grid()
            plt.show()

        print(f"Mean shift: {np.mean(shift_x_all):.2f}")
        print(f"Std shift: {np.std(shift_x_all):.4f}")
        print(f"Min shift: {np.min(shift_x_all):.2f}, Max shift: {np.max(shift_x_all):.2f}")
        print(f"Max shift diff: {np.max(shift_x_all) - np.mean(shift_x_all):.4f}")
    
    ###### test pcd and profile height width ####
    if test_pcd:
        scan_process = ScanProcess(robot_scan,positioner)
        pcd = o3d.io.read_point_cloud(this_layer_dir+'pcd.pcd')
        pcd_denoise = o3d.io.read_point_cloud(this_layer_dir+'pcd_denoise.pcd')
        pcd_base_denoise = o3d.io.read_point_cloud(logdata_dir+'baselayer0/'+'pcd_denoise.pcd')
        last_profile_height = np.loadtxt(last_layer_dir+'profile_height.csv',delimiter=',')

        baselayer1_profile_height = np.loadtxt(logdata_dir+'baselayer1/profile_height.csv',delimiter=',')
        shift_x = get_weld_shift_x(baselayer1_profile_height)

        # cropping the point cloud
        curve_planned_z = np.mean(curve[:,2])
        curve_x_end = np.min(curve[:,0])
        curve_x_start = np.max(curve[:,0])
        curve_y = np.mean(curve[:,1])
        z_height_start=curve_planned_z+0.1
        crop_extend_x=20
        crop_extend_z=20
        crop_min=(curve_x_end-crop_extend_x,curve_y-30,-30)
        crop_max=(curve_x_start+crop_extend_x,curve_y+30,z_height_start+crop_extend_z)
        crop_h_min=(curve_x_end-crop_extend_x,curve_y-20,-30)
        crop_h_max=(curve_x_start+crop_extend_x,curve_y+20,z_height_start+crop_extend_z)
        
        _, _,Transz0_H = scan_process.pcd2height(deepcopy(pcd_base_denoise),0.1,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=None,return_width=True)
        
        profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd_denoise),last_profile_height,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=False)
        # _, profile_width,_ = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
        _, profile_width,_ = scan_process.pcd2height(deepcopy(pcd),last_profile_height,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)

        fig,ax = plt.subplots(2,1,figsize=(12,6))
        ax[0].plot(profile_height[:,0]+shift_x,profile_height[:,1],'-o',label=f'height')
        ax[1].plot(profile_width[:,0]+shift_x,profile_width[:,1],'-o',label=f'width')
        plt.legend()
        plt.show()

        visualize_pcd([pcd])

    ###### get std and other statistics
    if get_statistics:
        data_dir = '../../data/wall_weld_test/'
        test_dir = ['weld_fujicontrol_2025_08_13_14_57_52/','weld_fujicontrol_2025_08_14_11_19_59/']
        test_labels = ['Baseline','Control']
        test_dimension = ['Height', 'Width']
        test_statistics = ['STD']

        # start_x={'Baseline':-45,'Control':-45}
        # end_x={'Baseline':45,'Control':45}

        # start_x={'Baseline':-55,'Control':-60}
        # end_x={'Baseline':55,'Control':60}

        start_x={'Baseline':-60,'Control':-60}
        end_x={'Baseline':60,'Control':60}

        test_results={}
        for logdata_dir_name,dat_label in zip(test_dir,test_labels):
            test_results[dat_label] = {}
            ## directory to process
            print(f"Processing directory: {logdata_dir_name}")
            logdata_dir = data_dir + logdata_dir_name
            ## get weld meta file
            with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
                weld_meta = yaml.safe_load(f)
            if 'correction_start_layer' in weld_meta.keys():
                correction_start_layer = weld_meta['correction_start_layer'] 
            else:
                correction_start_layer = 99999999 if dat_label == 'Baseline' else 2

            ## get shift x
            baselayer1_profile_height = np.loadtxt(logdata_dir+'baselayer1/profile_height.csv',delimiter=',')
            shift_x = get_weld_shift_x(baselayer1_profile_height)
            print(f"Shift X: {shift_x:.2f}")

            ### loop through all layers to get height width statistics
            total_layers_name = glob.glob(logdata_dir+'layer*')
            # get printed layer number
            layer_nums = []
            for layer_name in total_layers_name:
                this_layer = layer_name.split('\\')[-1]
                this_layer = this_layer.split('r')[-1]
                layer_nums.append(int(this_layer))
            layer_nums = np.sort(layer_nums)
            
            # collect statistics
            height_std = []
            width_std = []
            control_error_dh = []
            control_error_width = []
            prediction_error_dh = []
            prediction_error_width = []
            last_profile_height = np.loadtxt(logdata_dir + f'baselayer1/profile_height.csv',delimiter=',')
            last_profile_height[:,0] += shift_x
            for layer_n_id, layer_n in enumerate(layer_nums):
                this_layer_dir = logdata_dir + f'layer{layer_n}/'
                profile_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',')
                profile_width = np.loadtxt(this_layer_dir+'profile_width.csv',delimiter=',')
                # shift profiles
                profile_height[:,0] += shift_x
                profile_width[:,0] += shift_x
                
                # get std between -55 and 55 mm
                height_std.append(np.std(profile_height[(profile_height[:,0] >= start_x[dat_label]) & (profile_height[:,0] <= end_x[dat_label] )& (profile_height[:,1]> 6), 1]))
                width_std.append(np.std(profile_width[(profile_width[:,0] >= start_x[dat_label]) & (profile_width[:,0] <= end_x[dat_label]), 1]))

                if layer_n_id == len(layer_nums)-1:
                    test_results[dat_label]['Height Viz'] = profile_height
                    test_results[dat_label]['Width Viz'] = profile_width
                
                # get target dh dw vs control/prediction dh dw vs actual dh dw
                if layer_n_id >= correction_start_layer:

                    control_status_log = np.loadtxt(this_layer_dir+'control_status_log.csv',delimiter=',')
                    control_status_log_actual_height = \
                        np.interp(control_status_log[:,1],profile_height[:,0],profile_height[:,1])
                    control_status_log_actual_height_last_layer = \
                        np.interp(control_status_log[:,1],last_profile_height[:,0],last_profile_height[:,1])
                    control_status_log_actual_dh = control_status_log_actual_height - control_status_log_actual_height_last_layer
                    control_status_log_actual_width = \
                        np.interp(control_status_log[:,1],profile_width[:,0],profile_width[:,1])
                    control_status_log = np.column_stack((control_status_log[:,:-1], control_status_log_actual_dh, control_status_log_actual_width, control_status_log[:,-1][:,None]))

                    # remove edges
                    start_end_location = 47.5
                    control_status_log = control_status_log[control_status_log[:,1]>=-start_end_location]
                    control_status_log = control_status_log[control_status_log[:,1]<=start_end_location]

                    cmd_updated_id = np.where(control_status_log[:,-1]!=0)[0]
                    # exlude the last segments if too short
                    if cmd_updated_id[-1] < len(control_status_log)-8:
                        control_status_log = control_status_log[:cmd_updated_id[-1],:]
                        cmd_updated_id = cmd_updated_id[:-1]
                    control_status_log_smooth = []
                    for profile in control_status_log[:,:-1].T:
                        control_status_log_smooth.append(get_sum_profile(profile, cmd_updated_id))
                    control_status_log_smooth = np.array(control_status_log_smooth).T

                    target_vs_control_dh_error = control_status_log_smooth[:,4] - control_status_log_smooth[:,6]
                    target_vs_control_width_error = control_status_log_smooth[:,5] - control_status_log_smooth[:,7]
                    actual_vs_predict_dh_error = control_status_log_smooth[:,8] - control_status_log_smooth[:,6]
                    actual_vs_predict_width_error = control_status_log_smooth[:,9] - control_status_log_smooth[:,7]

                    control_error_dh.append(target_vs_control_dh_error)
                    control_error_width.append(target_vs_control_width_error)
                    prediction_error_dh.append(actual_vs_predict_dh_error)
                    prediction_error_width.append(actual_vs_predict_width_error)

                last_profile_height = deepcopy(profile_height)

            # collect statistics
            test_results[dat_label]['Height'] = {}
            test_results[dat_label]['Width'] = {}
            test_results[dat_label]['Height']['STD'] = height_std
            test_results[dat_label]['Width']['STD'] = width_std
            test_results[dat_label]['Height']['Control Error'] = control_error_dh
            test_results[dat_label]['Width']['Control Error'] = control_error_width
            test_results[dat_label]['Height']['Prediction Error'] = prediction_error_dh
            test_results[dat_label]['Width']['Prediction Error'] = prediction_error_width

        for stat in test_statistics:
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            for dim_i,dim in enumerate(test_dimension):
                for dat_label in test_labels:
                    ax[dim_i].plot(test_results[dat_label][dim][stat], '-o', label=f"{dat_label}")
                ax[dim_i].set_xlabel('Layer Number', fontsize=xy_label_size)
                ax[dim_i].tick_params(axis='both', which='major', labelsize=xy_tick_size)
                ax[dim_i].set_title(f"{dim} {stat} (mm)", fontsize=title_size)
                ax[dim_i].legend(fontsize=legend_size)
                ax[dim_i].grid()
        plt.show()

        fig, ax = plt.subplots(2, 1, figsize=(12, 7))
        for dim_i,dim in enumerate(test_dimension):
            for dat_label in test_labels:
                valid_index = np.where((test_results[dat_label][dim+' Viz'][:,0] >= start_x[dat_label]) & (test_results[dat_label][dim+' Viz'][:,0] <= end_x[dat_label]))[0]
                if dim == 'Height':
                    ax[dim_i].plot(test_results[dat_label][dim+' Viz'][valid_index,0],test_results[dat_label][dim+' Viz'][valid_index,1]-np.mean(test_results[dat_label][dim+' Viz'][valid_index,1]), '-o', label=f"{dat_label}")
                else:
                    ax[dim_i].plot(test_results[dat_label][dim+' Viz'][valid_index,0],test_results[dat_label][dim+' Viz'][valid_index,1], '-o', label=f"{dat_label}")
            ax[dim_i].set_xlabel('X Position (mm)', fontsize=xy_label_size)
            ax[dim_i].set_ylabel(f"(mm)", fontsize=xy_label_size)
            yticks = ax[dim_i].get_yticks()  # original y-tick values
            # new_labels = []  # rename ticks
            # for val in yticks:
            #     if val > 0:
            #         new_labels.append(f"Mean+{val:.0f}")
            #     else:
            #         new_labels.append(f"Mean-{-val:.0f}")
            # ax[dim_i].set_yticks(yticks)  # ensure same positions
            # ax[dim_i].set_yticklabels(new_labels)  # apply new labels
            ax[dim_i].tick_params(axis='both', which='major', labelsize=xy_tick_size)
            ax[dim_i].set_title(f"Layer "+dim, fontsize=title_size)
            ax[dim_i].legend(fontsize=legend_size)
            ax[dim_i].grid()
        plt.show()

        # get height width error statistics
        for dat_label in test_labels:
            if 'Control Error' not in test_results[dat_label]['Width'].keys():
                continue
            fig,ax = plt.subplots(2,2,figsize=(12,10))
            for dim_i,dim in enumerate(test_dimension):
                for err_i,err_type in enumerate(['Control Error','Prediction Error']):
                    layer_error_all = []
                    layer_error_mean = []
                    layer_error_std = []
                    for layer_n_id, layer_error in enumerate(test_results[dat_label][dim][err_type]):
                        layer_error_mean.append(np.mean(np.abs(layer_error)))
                        layer_error_std.append(np.std(np.abs(layer_error)))
                        layer_error_all.extend(layer_error)
                    ax[dim_i,err_i].errorbar(np.arange(len(layer_error_mean))+correction_start_layer, layer_error_mean, yerr=layer_error_std, fmt='-o', label=f"{dat_label} {err_type}")
                    ax[dim_i,err_i].set_xlabel('Layer Number', fontsize=xy_label_size)
                    ax[dim_i,err_i].set_ylabel('Error (mm)', fontsize=xy_label_size)
                    ax[dim_i,err_i].tick_params(axis='both', which='major', labelsize=xy_tick_size)
                    ax[dim_i,err_i].set_title(f"{dim} {err_type}", fontsize=title_size)
                    # ax[dim_i,err_i].legend(fontsize=legend_size)
                    ax[dim_i,err_i].grid()
                    print(f"{dat_label} {dim} {err_type} Mean Error: {np.mean(np.abs(layer_error_all)):.4f}, 95% Error: {stats.expon(scale=np.std(np.abs(layer_error_all))).interval(0.95)[1]:.4f}")
            plt.show()

    ###### test log-log control model RLS #####
    if test_loglog:
        data_dir = '../../data/wall_weld_test/'
        test_dir = ['weld_fujicontrol_2025_08_14_11_19_59/']
        
        loglog_model_dir = 'loglog_models'
        loglogModel = controlLogLogModel(loglog_model_dir)
        loglogModel_static = controlLogLogModel(loglog_model_dir)

        for logdata_dir_name in test_dir:
            logdata_dir = data_dir + logdata_dir_name
            ## get weld meta file
            with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
                weld_meta = yaml.safe_load(f)
            if 'correction_start_layer' in weld_meta.keys():
                correction_start_layer = weld_meta['correction_start_layer'] 
            else:
                correction_start_layer = 2
            
            ## get shift x
            baselayer1_profile_height = np.loadtxt(logdata_dir+'baselayer1/profile_height.csv',delimiter=',')
            shift_x = get_weld_shift_x(baselayer1_profile_height)
            print(f"Shift X: {shift_x:.2f}")

            ### loop through all layers to get height width statistics
            total_layers_name = glob.glob(logdata_dir+'layer*')
            # get printed layer number
            layer_nums = []
            for layer_name in total_layers_name:
                this_layer = layer_name.split('\\')[-1]
                this_layer = this_layer.split('r')[-1]
                layer_nums.append(int(this_layer))
            layer_nums = np.sort(layer_nums)

            ### visualization
            # Build a color for each batch
            cmap = plt.get_cmap('viridis', len(layer_nums)-correction_start_layer)
            batch_to_color = {layer_nums[i+correction_start_layer]: cmap(i) for i in range(len(layer_nums)-correction_start_layer)}
            print(batch_to_color)
            # Figure + axes
            fig = plt.figure(figsize=(12, 8))
            ax00 = fig.add_subplot(1, 2, 1, projection='3d')  # planes for log h
            ax01 = fig.add_subplot(1, 2, 2, projection='3d')  # planes for log w

            last_profile_height = np.loadtxt(logdata_dir + f'baselayer1/profile_height.csv',delimiter=',')
            last_profile_height[:,0] += shift_x
            prediction_error_dh = []
            prediction_error_width = []
            prediction_error_dh_static = []
            prediction_error_width_static = []
            for layer_n_id, layer_n in enumerate(layer_nums):
                this_layer_dir = logdata_dir + f'layer{layer_n}/'
                profile_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',')
                profile_width = np.loadtxt(this_layer_dir+'profile_width.csv',delimiter=',')
                # shift profiles
                profile_height[:,0] += shift_x
                profile_width[:,0] += shift_x
                
                # get target dh dw vs control/prediction dh dw vs actual dh dw
                if layer_n_id >= correction_start_layer:
                    control_status_log = np.loadtxt(this_layer_dir+'control_status_log.csv',delimiter=',')

                    # rls update
                    log_v, log_feedrate, log_dh, log_dw, dh_pred_error, dw_pred_error = loglogModel.rls_update(profile_height, last_profile_height, profile_width, control_status_log)
                    layer_dh_theta = deepcopy(loglogModel.theta_dh)
                    layer_dw_theta = deepcopy(loglogModel.theta_dw)
                    # get prediction error (before rls)
                    prediction_error_dh.append(dh_pred_error)
                    prediction_error_width.append(dw_pred_error)

                    # static model
                    _,_,_,_, dh_pred_error_static, dw_pred_error_static = loglogModel_static.rls_update(profile_height, last_profile_height, profile_width, control_status_log)
                    prediction_error_dh_static.append(dh_pred_error_static)
                    prediction_error_width_static.append(dw_pred_error_static)
                    # restore the theta
                    loglogModel_static.theta_dh = loglogModel_static.theta_dh_history[-1]
                    loglogModel_static.theta_dw = loglogModel_static.theta_dw_history[-1]

                    if layer_n_id % 1 == 0:
                        # log h scatter
                        ax00.scatter(log_v, log_feedrate, log_dh,
                                    s=6, alpha=0.8, depthshade=False, label=f'layer {layer_n}', color=batch_to_color[layer_n])
                        # log w scatter
                        ax01.scatter(log_v, log_feedrate, log_dw,
                                    s=6, alpha=0.8, depthshade=False, label=f'layer {layer_n}', color=batch_to_color[layer_n])
                        # Create grid for planes
                        Xv, Xo = grid_from_data(np.log([0.5,20]), np.log(np.array([50,250])*inch2mm/60), n=30, pad=0.05)
                        Za = layer_dh_theta[0]*Xv + layer_dh_theta[1]*Xo + layer_dh_theta[2]  # for log h
                        Zw = layer_dw_theta[0]*Xv + layer_dw_theta[1]*Xo + layer_dw_theta[2]  # for log w
                        # Use wireframes or translucent surfaces; wireframes keep clutter down
                        try:
                            surf_h.remove()
                            surf_w.remove()
                        except NameError:
                            pass
                        surf_h = ax00.plot_wireframe(Xv, Xo, Za, rstride=3, cstride=3, color='red', alpha=0.9, linewidth=0.6)
                        surf_w = ax01.plot_wireframe(Xv, Xo, Zw, rstride=3, cstride=3, color='red', alpha=0.9, linewidth=0.6)

                        for ax, zlabel, title in [
                            (ax00, f'$log \Delta h$', f'Regression plane for $log \Delta h$'),
                            (ax01, f'$log w$', f'Regression plane for $log w$')]:
                            ax.set_xlabel(f'$log v$', fontsize=xy_label_size)
                            ax.set_ylabel(f'$log feedrate$', fontsize=xy_label_size)
                            ax.set_zlabel(zlabel, fontsize=xy_label_size)
                            ax.set_title(title, fontsize=title_size)
                            ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
                            ax.view_init(elev=22, azim=-55)  # a nice default view

                        plt.pause(0.5)

                last_profile_height = deepcopy(profile_height)
            
            plt.show()

            dh_error_mean = []
            dh_error_std = []
            dh_error_all = []
            width_error_mean = []
            width_error_std = []
            width_error_all = []
            dh_error_mean_static = []
            dh_error_std_static = []
            dh_error_all_static = []
            width_error_mean_static = []
            width_error_std_static = []
            width_error_all_static = []

            for dh_error, dw_error, dh_error_static, dw_error_static in zip(prediction_error_dh, prediction_error_width, prediction_error_dh_static, prediction_error_width_static):
                dh_error_mean.append(np.mean(np.abs(dh_error)))
                dh_error_std.append(np.std(np.abs(dh_error)))
                dh_error_all.extend(dh_error)
                dh_error_mean_static.append(np.mean(np.abs(dh_error_static)))
                dh_error_std_static.append(np.std(np.abs(dh_error_static)))
                dh_error_all_static.extend(dh_error_static)
                width_error_mean.append(np.mean(np.abs(dw_error)))
                width_error_std.append(np.std(np.abs(dw_error)))
                width_error_all.extend(dw_error)
                width_error_mean_static.append(np.mean(np.abs(dw_error_static)))
                width_error_std_static.append(np.std(np.abs(dw_error_static)))
                width_error_all_static.extend(dw_error_static)
            print("RLS Model:")
            print(f"Height Mean Error: {np.mean(np.abs(dh_error_all)):.4f}, 95% Error: {stats.expon(scale=np.std(np.abs(dh_error_all))).interval(0.95)[1]:.4f}")
            print(f"Width Mean Error: {np.mean(np.abs(width_error_all)):.4f}, 95% Error: {stats.expon(scale=np.std(np.abs(width_error_all))).interval(0.95)[1]:.4f}")
            print("Static Model:")
            print(f"Height Mean Error: {np.mean(np.abs(dh_error_all_static)):.4f}, 95% Error: {stats.expon(scale=np.std(np.abs(dh_error_all_static))).interval(0.95)[1]:.4f}")
            print(f"Width Mean Error: {np.mean(np.abs(width_error_all_static)):.4f}, 95% Error: {stats.expon(scale=np.std(np.abs(width_error_all_static))).interval(0.95)[1]:.4f}")

            fig, ax = plt.subplots(1,2,figsize=(12,6))
            ax[0].errorbar(layer_nums[correction_start_layer:], dh_error_mean_static, yerr=dh_error_std_static, fmt='-o', label=f'Static model')
            ax[0].errorbar(layer_nums[correction_start_layer:], dh_error_mean, yerr=dh_error_std, fmt='-o', label=f'RLS model')
            ax[1].errorbar(layer_nums[correction_start_layer:], width_error_mean_static, yerr=width_error_std_static, fmt='-o', label=f'Static model')
            ax[1].errorbar(layer_nums[correction_start_layer:], width_error_mean, yerr=width_error_std, fmt='-o', label=f'RLS model')
            ax[0].set_xlabel('Layer Number', fontsize=xy_label_size)
            ax[0].set_ylabel('mm', fontsize=xy_label_size)
            ax[1].set_xlabel('Layer Number', fontsize=xy_label_size)
            ax[1].set_ylabel('mm', fontsize=xy_label_size)
            ax[0].tick_params(axis='both', which='major', labelsize=xy_tick_size)
            ax[1].tick_params(axis='both', which='major', labelsize=xy_tick_size)
            ax[0].legend(fontsize=legend_size)
            ax[1].legend(fontsize=legend_size)
            ax[0].set_title(f'$\Delta h$ Prediction Error', fontsize=title_size)
            ax[1].set_title(f'Width Prediction Error', fontsize=title_size)
            plt.show()

    ###### viz geometry #####
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
        # logdata_dir_all = ['weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
        # test_dir = ['weld_fujiscan_2025_07_09_14_52_42/']
        test_dir = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/',\
                        'weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
        
        
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
            show_pcd_list = []
            for layer_n_id, layer_n in enumerate(layer_nums):
                # print(f"Processing layer {layer_n} ({layer_n_id+1}/{len(layer_nums)})")
                # if layer_n_id % 3 != 0:
                #     print(f"Skipping layer {layer_n} due to odd index")
                #     continue
                # if layer_n != 129:
                #     continue
                this_layer_dir = logdata_dir + 'layer' + str(layer_n) + '/'

                profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
                profile_welding_viz.append(profile_welding[:,[1,4]])
                # profile_welding_viz.append(profile_welding[:,[0,4]])

                # profile_welding_cut = profile_welding[(profile_welding[:,1] >= -55) & (profile_welding[:,1] <= 45)]
                # plt.plot(profile_welding_cut[:,0]-profile_welding[0,0], profile_welding_cut[:,5]-np.mean(profile_welding_cut[:,5]), '-o', label='dh')
                # plt.plot(profile_welding_cut[:,0]-profile_welding[0,0], profile_welding_cut[:,6]-np.mean(profile_welding_cut[:,6]), '-o', label='torch height')
                # plt.title(f'Welding Profile at Layer {layer_n}')
                # plt.legend()
                # plt.grid()
                # plt.show()

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

                if layer_n_id >= len(layer_nums)-9:
                    pcd = o3d.io.read_point_cloud(this_layer_dir+'pcd_denoise.pcd')
                    show_pcd_list.append(pcd)
                if layer_n_id == len(layer_nums)-1:
                    # print the height and width of the middle part of the last layer
                    profile_welding = profile_welding[profile_welding[:,1] >= -5]
                    profile_welding = profile_welding[profile_welding[:,1] <= 5]
                    print(f"Feedrate: {float(np.mean(profile_welding[:,3])):.2f} ipm, Welding Speed: {float(np.mean(profile_welding[:,2])):.2f} mm/s")
                    print(f"Average height: {float(np.mean(profile_welding[:,4])):.2f} mm, Average width: {float(np.mean(profile_welding[:,7])):.2f} mm")
                    print("==========================================")
                    visualize_pcd(show_pcd_list)

            # visualize the height
            # for profile_cnt,height_profile in enumerate(height_viz):
            #     plt.plot(height_profile[:,0], height_profile[:,1]+40, '-o', color=color_viz[profile_cnt], label='Height Profile')
            #     # plt.plot(profile_welding_viz[profile_cnt][:,0]-profile_welding_viz[profile_cnt][0,0], profile_welding_viz[profile_cnt][:,1], 'o', label='Welding Profile')
            #     plt.plot(profile_welding_viz[profile_cnt][:,0], profile_welding_viz[profile_cnt][:,1], 'o', label='Welding Profile')
            # plt.xlabel('X Position (mm)')
            # plt.ylabel('Height (mm)')
            # plt.grid()
            # plt.show()

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

    ###### read thermal example #####
    if test_read_thermal:
        with open(this_layer_dir+'thermal_pixel_trace.pickle', 'rb') as f:
            thermal_pixel_trace = pickle.load(f)
        # for x in thermal_pixel_trace.keys():
        #     plt.plot(thermal_pixel_trace[x]['time']-thermal_pixel_trace[x]['time'][0], thermal_pixel_trace[x]['value'])
        #     plt.title(f'Thermal Trace at X={x:.2f} mm')
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Temperature (°C)')
        #     plt.grid()
        #     plt.pause(0.1)

        profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv',delimiter=',',skiprows=1)
        profile_welding_x = profile_welding[:,1]
        profile_welding_t = profile_welding[:,0]
        profile_welding_t_sample = np.arange(np.min(profile_welding_t), np.max(profile_welding_t), 0.1)
        profile_welding_x_sample = np.interp(profile_welding_t_sample, profile_welding_t, profile_welding_x)

        # build a 2D linear interpolation in time and x
        x_time_map = []
        x_time_value = []
        for x in thermal_pixel_trace.keys():
            for t, value in zip(thermal_pixel_trace[x]['time'], thermal_pixel_trace[x]['value']):
                x_time_map.append([x, t])
                x_time_value.append(value)
        x_time_map = np.array(x_time_map)
        x_time_value = np.array(x_time_value)

        print(f"x_time_map shape: {x_time_map.shape}, x_time_value shape: {x_time_value.shape}")
        # thermal_map = LinearNDInterpolator(x_time_map, x_time_value, fill_value=8000)
        # visualize the thermal map
        
        # dynamic local linear interpolation
        thermal_map = None
        t_sample_window = 0.5 # sec
        x_sample_window = 12 # mm
        start_time = time.perf_counter()
        for data_i, (t,x) in enumerate(zip(profile_welding_t_sample, profile_welding_x_sample)):
            if data_i % 10 == 0:
                print(f"Processing data point {data_i}: (t={t}, x={x})")

            x_local = np.arange(x - 10, x + 10.1, 0.1)
            mesh_X, mesh_Y = np.meshgrid(x_local, t)
            # if thermal_map is None:
            this_x_time_map = x_time_map[(x_time_map[:, 0] >= x - x_sample_window) & (x_time_map[:, 0] <= x + x_sample_window)]
            this_x_time_value = x_time_value[(x_time_map[:, 0] >= x - x_sample_window) & (x_time_map[:, 0] <= x + x_sample_window)]
            closest_t = x_time_map[np.argmin(np.abs(x_time_map[:, 1] - t)), 1]
            this_x_time_map = x_time_map[(x_time_map[:, 1] == closest_t)]
            this_x_time_value = x_time_value[(x_time_map[:, 1] == closest_t)]
            this_x_sort_id = np.argsort(this_x_time_map[:, 0])
            this_x = this_x_time_map[this_x_sort_id,0]
            this_values = this_x_time_value[this_x_sort_id]
            Z_interp = np.interp(x_local, this_x, this_values, left=8000, right=8000)
            # if np.where(Z_value == 8000)[0].size > 90:
            #     this_x_time_map = x_time_map[(x_time_map[:, 0] >= x - x_sample_window) & (x_time_map[:, 0] <= x + x_sample_window)]
            #     this_x_time_value = x_time_value[(x_time_map[:, 0] >= x - x_sample_window) & (x_time_map[:, 0] <= x + x_sample_window)]
            #     # this_x_time_map = x_time_map[(x_time_map[:, 1] >= t - t_sample_window) & (x_time_map[:, 1] <= t + t_sample_window)]
            #     # this_x_time_value = x_time_value[(x_time_map[:, 1] >= t - t_sample_window) & (x_time_map[:, 1] <= t + t_sample_window)]
            #     t_one_step_large = np.min(x_time_map[x_time_map[:, 1] > t, 1])
            #     t_one_step_small = np.max(x_time_map[x_time_map[:, 1] < t, 1])
            #     this_x_time_map = x_time_map[(x_time_map[:, 1] >= t_one_step_small) & (x_time_map[:, 1] <= t_one_step_large)]
            #     this_x_time_value = x_time_value[(x_time_map[:, 1] >= t_one_step_small) & (x_time_map[:, 1] <= t_one_step_large)]
            #     thermal_map = LinearNDInterpolator(this_x_time_map, this_x_time_value, fill_value=8000)
            #     Z = thermal_map(mesh_X, mesh_Y)
            #     Z_value = Z.flatten()

            plt.clf()
            plt.plot(x_local, Z_interp, '-o')
            plt.xlabel('X Position (mm)')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Thermal Map at Time={t:.2f} s, X={x:.2f} mm')
            plt.grid()
            plt.pause(0.1)
        print(f"Time taken for dynamic interpolation: {time.perf_counter() - start_time:.2f} s")

        # # Z = thermal_map(X, Y)
        # plt.pcolormesh(X, Y, Z, shading='auto')
        # plt.colorbar(label='Brightness')
        # plt.xlabel('X Position (mm)')
        # plt.ylabel('Time (s)')
        # plt.title('Thermal Map')
        # plt.grid()
        # plt.show()

    ###### reverse thermal pixel trace #####
    if reverse_thermal_pixel_trace:
        logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                       'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                       'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/',\
                       'weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']
        for logdata_dir_name in logdata_dir_all:
            print('Processing:',logdata_dir_name)
            logdata_dir = data_dir+logdata_dir_name
            for weld_parts in ['base','layer']:
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

                for layer_n_id, layer_n in enumerate(layer_nums):
                    # read logged data
                    if weld_parts == 'base':
                        layer_name = 'baselayer'+str(layer_n)
                    else:
                        layer_name = 'layer'+str(layer_n)
                    print('Processing layer:',layer_name)
                    this_layer_dir = logdata_dir+layer_name+'/'
                    with open(this_layer_dir+'thermal_pixel_trace.pickle', 'rb') as f:
                        thermal_pixel_trace = pickle.load(f)
                    thermal_x = []
                    thermal_t = []
                    thermal_value = []
                    thermal_stamp_all = []
                    for lox_x in thermal_pixel_trace.keys():
                        thermal_t.extend(thermal_pixel_trace[lox_x]['time'])
                        thermal_value.extend(thermal_pixel_trace[lox_x]['value'])
                        thermal_x.extend([lox_x]*len(thermal_pixel_trace[lox_x]['time']))
                        thermal_stamp_all.extend(np.setdiff1d(thermal_pixel_trace[lox_x]['time'], thermal_stamp_all))
                    thermal_x = np.array(thermal_x)
                    thermal_t = np.array(thermal_t)
                    thermal_value = np.array(thermal_value)
                    thermal_stamp_all = np.sort(np.array(thermal_stamp_all))
                    thermal_pixel_trace_stamp_key = {}
                    for i, stamp in enumerate(thermal_stamp_all):
                        thermal_pixel_trace_stamp_key[stamp] = {}
                        this_thermal_x = thermal_x[thermal_t == stamp]
                        this_thermal_value = thermal_value[thermal_t == stamp]
                        x_sort_id = np.argsort(this_thermal_x)
                        thermal_pixel_trace_stamp_key[stamp]['x'] = this_thermal_x[x_sort_id]
                        thermal_pixel_trace_stamp_key[stamp]['value'] = this_thermal_value[x_sort_id]
                    with open(this_layer_dir+'thermal_pixel_trace_stamp_key.pickle', 'wb') as f:
                        pickle.dump(thermal_pixel_trace_stamp_key, f)

if __name__ == "__main__":
    
    main()