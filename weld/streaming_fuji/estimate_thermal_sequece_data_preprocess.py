import numpy as np
from matplotlib import pyplot as plt
import sys, datetime, yaml, pathlib, glob, os, time, argparse
sys.path.append('../../mocap/')
from Models import *
from model_train_utils import *

# for plotting
xy_label_size = 14
xy_tick_size = 12
legend_size = 12
title_size = 16
sup_title_size = 18

# load data
geo_data_dir = '../../data/wall_weld_test/'
logdata_dir_all = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
                    'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
                    'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/']

sample_rate = 10
ignore_start_end = 0
start_x = -55 + ignore_start_end
end_x = 55 - ignore_start_end
thermal_window = 40 # 40 mm
thermal_dx_sample = 0.2 # sample every 0.2 mm
thermal_void_value = 7000 # if no values than background

### data processing
data_dirs = []
train_data_batch_len = []
for i, logdata_dir in enumerate(logdata_dir_all):
    total_layers_name = glob.glob(geo_data_dir+logdata_dir+'layer*')
    # get printed layer number
    layer_nums = []
    for layer_name in total_layers_name:
        this_layer = layer_name.split('\\')[-1]
        this_layer = this_layer.split('r')[-1]
        layer_nums.append(int(this_layer))
    layer_nums = np.sort(layer_nums)
    for layer_n_id, layer_n in enumerate(layer_nums):
        layer_name = 'layer'+str(layer_n)
        this_layer_dir = geo_data_dir + logdata_dir + layer_name + '/'
        print("Processing layer "+str(layer_n)+" in "+logdata_dir)
        print('No data for layer '+str(layer_n)+' in '+logdata_dir)
        print("Interpolate data from profile welding.csv")
        # load data
        profile_welding = np.loadtxt(this_layer_dir+'profile_welding.csv', delimiter=',', skiprows=1)
        with open(this_layer_dir+'thermal_pixel_trace_stamp_key.pickle', 'rb') as f:
            thermal_pixel_trace = pickle.load(f)
        # chop x < start_x or x > end_x
        x_location = np.array(profile_welding[:, 1])
        if x_location[-1]>x_location[0]:
            start_index = np.where(x_location > start_x)[0][0] 
            end_index = np.where(x_location < end_x)[0][-1]
        else:
            start_index = np.where(x_location < end_x)[0][0] 
            end_index = np.where(x_location > start_x)[0][-1]
        profile_welding = profile_welding[start_index:end_index+1, :]
        # interpolate the data to the sample rate
        timestamp_welding = profile_welding[:,0]
        timestamps_interp = np.arange(timestamp_welding[0]+1/sample_rate, timestamp_welding[-1], 1/sample_rate)
        x_loc_interp = np.interp(timestamps_interp, timestamp_welding, profile_welding[:,1])
        cmd_v_interp = np.zeros_like(timestamps_interp)
        cmd_fd_interp = np.zeros_like(timestamps_interp)
        dh_interp = np.zeros_like(timestamps_interp)
        dw_interp = np.zeros_like(timestamps_interp)
        stickout_interp = np.zeros_like(timestamps_interp)
        thermal_interp = np.zeros_like(timestamps_interp)
        thermal_x_interp = np.zeros_like(timestamps_interp)
        thermal_y_interp = np.zeros_like(timestamps_interp)
        thermal_neighborhood_interp = []
        thermal_stamps = np.array(list(thermal_pixel_trace.keys()))
        for interp_id, (interp_time, interp_x) in enumerate(zip(timestamps_interp, x_loc_interp)):

            # find closest smaller thermal t
            thermal_t_closest = np.max(thermal_stamps[thermal_stamps <= interp_time])
            this_thermal_neighbor_x = np.arange(interp_x-thermal_window, interp_x+thermal_window, thermal_dx_sample)
            np.append(this_thermal_neighbor_x, interp_x+thermal_window) if this_thermal_neighbor_x[-1] < interp_x+thermal_window else None
            this_thermal_neighbor_value = np.interp(this_thermal_neighbor_x, thermal_pixel_trace[thermal_t_closest]['x'], thermal_pixel_trace[thermal_t_closest]['value'], left=thermal_void_value, right=thermal_void_value)
            # find how many value in this_thermal_neighbor_value are thermal_void_value
            if np.sum(this_thermal_neighbor_value==thermal_void_value)/len(this_thermal_neighbor_value) > 0.51:
                # plt.plot(this_thermal_neighbor_x, this_thermal_neighbor_value, '-o')
                # plt.title(f'More than 50% void values in thermal neighborhood, Time:{interp_time:.2f}, X:{interp_x:.2f}')
                # plt.show()
                # exit()
                print('Warning: More than 51% void values in thermal neighborhood')
            thermal_neighborhood_interp.append(this_thermal_neighbor_value)

            # find the welding data in the time window [interp_time-1/sample_rate, interp_time]
            window_id_start = np.where(timestamp_welding >= interp_time-1/sample_rate)[0][0]
            window_id_end = np.where(timestamp_welding <= interp_time)[0][-1]+1
            if window_id_end<=window_id_start:
                print('Error: window_id_end <= window_id_start, check the data')
                continue
            if window_id_end>len(timestamp_welding) or window_id_start>=len(timestamp_welding):
                print('Error: window_id_end > len(timestamps) or window_id_start >= len(timestamps), check the data')
                continue
            cmd_v_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 2])
            cmd_fd_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 3])
            dh_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 5])
            dw_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 7])
            stickout_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 6])
            thermal_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 9])
            thermal_x_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 10])
            thermal_y_interp[interp_id] = np.mean(profile_welding[window_id_start:window_id_end, 11])

        if np.any(cmd_v_interp==0):
            # interpolate the zero values using linear interpolation
            cmd_v_interp = np.interp(timestamps_interp, timestamps_interp[cmd_v_interp!=0], cmd_v_interp[cmd_v_interp!=0])
            cmd_fd_interp = np.interp(timestamps_interp, timestamps_interp[cmd_fd_interp!=0], cmd_fd_interp[cmd_fd_interp!=0])
            dh_interp = np.interp(timestamps_interp, timestamps_interp[dh_interp!=0], dh_interp[dh_interp!=0])
            dw_interp = np.interp(timestamps_interp, timestamps_interp[dw_interp!=0], dw_interp[dw_interp!=0])
            stickout_interp = np.interp(timestamps_interp, timestamps_interp[stickout_interp!=0], stickout_interp[stickout_interp!=0])
            thermal_interp = np.interp(timestamps_interp, timestamps_interp[thermal_interp!=0], thermal_interp[thermal_interp!=0])
            thermal_x_interp = np.interp(timestamps_interp, timestamps_interp[thermal_x_interp!=0], thermal_x_interp[thermal_x_interp!=0])
            thermal_y_interp = np.interp(timestamps_interp, timestamps_interp[thermal_y_interp!=0], thermal_y_interp[thermal_y_interp!=0])
        
        assert len(timestamps_interp) == len(x_loc_interp) == len(cmd_v_interp) == len(cmd_fd_interp) == len(dh_interp) == len(dw_interp) == len(stickout_interp) == len(thermal_interp) == len(thermal_x_interp) == len(thermal_y_interp) == len(thermal_neighborhood_interp), "Mismatch in interpolated data lengths"

        # save the interpolated data
        interp_data = np.column_stack((timestamps_interp, x_loc_interp, cmd_v_interp, cmd_fd_interp, dh_interp, dw_interp, stickout_interp, thermal_interp, thermal_x_interp, thermal_y_interp))
        np.savetxt(this_layer_dir+'profile_welding_'+str(sample_rate)+'_dhdw.csv', interp_data, delimiter=',', header='timestamp,x_loc,cmd_v,cmd_fd,dh,dw,stickout,thermal,thermal_x,thermal_y')
        thermal_neighborhood_interp = np.array(thermal_neighborhood_interp)
        np.save(this_layer_dir+'profile_welding_'+str(sample_rate)+'_thermal_neighborhood.npy', thermal_neighborhood_interp)
        data_dirs.append(this_layer_dir)
        train_data_batch_len.append(len(interp_data))