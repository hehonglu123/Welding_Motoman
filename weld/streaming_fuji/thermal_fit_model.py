import time, os, copy, sys, yaml, inspect
import glob
from copy import deepcopy
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import CubicSpline, LinearNDInterpolator
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
import open3d as o3d
import cv2 as cv
from motoman_def import *
from robotics_utils import *

# for plotting
xy_label_size = 18
xy_tick_size = 16
legend_size = 16
title_size = 20
sup_title_size = 20

cam_pixel_moving_ratio = 1.77 # 1.77 pixel per mm

# ---------- helpers ----------
def robust_limits(z, q=0.995):
    """Robust symmetric limits around zero using quantiles to ignore outliers."""
    z = np.asarray(z)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return (-1, 1)
    zmax = np.quantile(np.abs(z), q)
    if zmax == 0:
        zmax = np.max(np.abs(z)) if np.max(np.abs(z)) > 0 else 1.0
    return (-zmax, zmax)

def signed_norm(z, center=0.0, q=0.995, symlog=False, linthresh=1e-2):
    """
    Centered norm for signed data. If symlog=True, uses symmetric log to show
    both tiny and huge magnitudes while keeping sign.
    """
    vmin, vmax = robust_limits(z, q=q)
    if symlog:
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax), (vmin, vmax)
    else:
        return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax), (vmin, vmax)

def plot_signed_scatter(ax, XY, Z, title, xlabel, ylabel, cmap="coolwarm",
                        q=0.995, symlog=False, linthresh=1e-2, s=4, alpha=0.9):
    """Scatter for irregular samples with proper signed normalization."""
    if symlog:
        norm, (vmin, vmax) = signed_norm(Z, q=q, symlog=symlog, linthresh=linthresh)
        sc = ax.scatter(XY[:,0], XY[:,1], c=Z, s=s, cmap=cmap, norm=norm, alpha=alpha)
    else:
        sc = ax.scatter(XY[:,0], XY[:,1], c=Z, s=s, cmap=cmap)
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=xy_label_size)
    ax.set_ylabel(ylabel, fontsize=xy_label_size)
    ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
    return sc

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

    visualize_thermal_map = False

    ##### read robot weld exe pose and stamp ####
    weld_relative_exe = np.loadtxt(this_layer_dir+'weld_relative_exe.csv', delimiter=',') # in mm
    weld_relative_v_exe = np.loadtxt(this_layer_dir+'weld_relative_v_exe.csv', delimiter=',') # in mm/s
    weld_relative_speed_exe = np.linalg.norm(weld_relative_v_exe, axis=1) # in mm/s
    rob_js_exe = np.loadtxt(this_layer_dir+'weld_js_exe.csv',delimiter=',')
    weld_cmd = np.loadtxt(this_layer_dir+'weld_cmd.csv',delimiter=',')
    # get js at index 1~6 and 13 14
    rob_js_exe = rob_js_exe[:,[0,1,2,3,4,5,6,13,14]]
    robot_stamps = rob_js_exe[:,0]
    stamps_diff_sorted = np.argsort(np.diff(robot_stamps))[::-1]
    for stamp_diff_id in stamps_diff_sorted:
        # make sure to find the time jump after the welding command
        if robot_stamps[stamp_diff_id] > weld_cmd[-1,0] and robot_stamps[stamp_diff_id] < weld_cmd[-1,0]+3:
            weld_split_id = stamp_diff_id
            break
    # weld_split_id = np.argmax(np.diff(robot_stamps))
    scan_js_exe = deepcopy(rob_js_exe)
    weld_js_exe = rob_js_exe[:weld_split_id+1,:]
    weld_stamps = robot_stamps[:weld_split_id+1]
    assert len(weld_stamps) == len(weld_relative_exe), len(weld_stamps) == len(weld_relative_v_exe)

    ##### read thermal data ####
    with open(this_layer_dir+'thermal_pixel_trace_stamp_key.pickle', 'rb') as f:
        thermal_pixel_trace = pickle.load(f)
    thermal_map = [] # a array with Nx4, each row is x,time,temperature,velocity,wire feed rate (related to power)
    thermal_map_contain_zero = []
    thermal_dx = [] # a array with Nx3, each row is x,time,dT/dx
    thermal_dx2 = [] # a array with Nx3, each row is x,time,d2T/dx2
    thermal_x_sample = np.arange(-81, 98, 0.5) # in mm , relative to weld
    thermal_stamp_all = np.sort(np.array(list(thermal_pixel_trace.keys())))
    min_stamp = np.min(thermal_stamp_all)
    for stamp in thermal_stamp_all:
        # find current weld x position
        weld_id = np.argmin(np.abs(weld_stamps - stamp))
        weld_cmd_id = np.argmin(np.abs(weld_cmd[:,0] - stamp))
        weld_x = weld_relative_exe[weld_id,0]
        this_stamp_thermal_sample_zero = np.interp(thermal_x_sample, thermal_pixel_trace[stamp]['x']-weld_x, thermal_pixel_trace[stamp]['value'], left=0, right=0)
        this_stamp_thermal_sample = np.interp(thermal_x_sample, thermal_pixel_trace[stamp]['x']-weld_x, thermal_pixel_trace[stamp]['value'])
        thermal_map.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp-min_stamp), thermal_x_sample, this_stamp_thermal_sample,\
                                        np.ones_like(thermal_x_sample)*weld_relative_speed_exe[weld_id], np.ones_like(thermal_x_sample)*weld_cmd[weld_cmd_id, 1]),  ).T.tolist() )
        thermal_map_contain_zero.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp-min_stamp), thermal_x_sample, this_stamp_thermal_sample_zero, \
                                        np.ones_like(thermal_x_sample)*weld_relative_speed_exe[weld_id], np.ones_like(thermal_x_sample)*weld_cmd[weld_cmd_id, 1]),  ).T.tolist() )
        # this_stamp_thermal_sample_dx = np.gradient(this_stamp_thermal_sample, thermal_x_sample)
        this_stamp_thermal_sample_dx = savgol_filter(this_stamp_thermal_sample, 11, 3, deriv=1, delta=0.5) # window size 11, polynomial order 3
        thermal_dx.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp-min_stamp), thermal_x_sample, this_stamp_thermal_sample_dx) ).T.tolist() )
        # this_stamp_thermal_sample_dx2 = np.gradient(this_stamp_thermal_sample_dx, thermal_x_sample) # second derivative
        this_stamp_thermal_sample_dx2 = savgol_filter(this_stamp_thermal_sample, 11, 3, deriv=2, delta=0.5) # window size 11, polynomial order 3
        thermal_dx2.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp-min_stamp), thermal_x_sample, this_stamp_thermal_sample_dx2) ).T.tolist() )
    thermal_map = np.array(thermal_map)
    thermal_map_contain_zero = np.array(thermal_map_contain_zero)
    thermal_dx = np.array(thermal_dx)
    thermal_dx2 = np.array(thermal_dx2)

    thermal_dt = np.zeros_like(thermal_map)
    for x in thermal_x_sample:
        this_x_id = np.where(thermal_map[:,1]==x)[0]
        this_x_stamp = thermal_map[this_x_id,0]
        # this_dthermal_dt = np.gradient(thermal_map[this_x_id,2], this_x_stamp)
        this_dthermal_dt = savgol_filter(thermal_map[this_x_id,2], 11, 3, deriv=1, delta=np.mean(np.diff(this_x_stamp))) # window size 11, polynomial order 3
        thermal_dt[this_x_id,2] = this_dthermal_dt
        thermal_dt[this_x_id,0] = this_x_stamp
        thermal_dt[this_x_id,1] = x

    non_zero_index = np.where(thermal_map_contain_zero[:,2]!=0)[0] # get rid of zero value
    thermal_map = thermal_map[non_zero_index]
    thermal_dx = thermal_dx[non_zero_index]
    thermal_dx2 = thermal_dx2[non_zero_index]
    thermal_dt = thermal_dt[non_zero_index]

    if visualize_thermal_map:
        # visualization for verification
        # ---------- data wrappers ----------
        # Expecting arrays shaped (N, 3): [:,0]=time; [:,1]=x; [:,2]=value
        TM  = thermal_map
        DX  = thermal_dx
        DX2 = thermal_dx2
        DT  = thermal_dt
        # Choose normalization style:
        # - For broad dynamic range in derivatives, symlog=True helps a lot.
        use_symlog = True         # try True first; set to False if you prefer linear
        symlog_linthresh = 1e-2   # linear region around zero for SymLogNorm
        # Make figure with nicer layout
        fig = plt.figure(figsize=(12, 8))
        axs = [fig.add_subplot(221), fig.add_subplot(222),
            fig.add_subplot(223), fig.add_subplot(224)]
        # Plot panels (use diverging cmap to highlight +/-)
        sc0 = plot_signed_scatter(
            axs[0], TM[:, :2], TM[:, 2]-8000, "thermal map",
            "time (s)", "y relative to torch (mm)",
            cmap="coolwarm", symlog=False  # temperature often non-negative; keep linear if so
        )
        sc1 = plot_signed_scatter(
            axs[1], DX[:, :2], DX[:, 2], "dU/dy",
            "time (s)", "y relative to torch (mm)",
            cmap="coolwarm", symlog=use_symlog, linthresh=symlog_linthresh
        )
        sc2 = plot_signed_scatter(
            axs[2], DX2[:, :2], DX2[:, 2], "d²U/dy²",
            "time (s)", "y relative to torch (mm)",
            cmap="coolwarm", symlog=use_symlog, linthresh=symlog_linthresh
        )
        sc3 = plot_signed_scatter(
            axs[3], DT[:, :2], DT[:, 2], "dU/dt",
            "time (s)", "y relative to torch (mm)",
            cmap="coolwarm", symlog=use_symlog, linthresh=symlog_linthresh
        )
        # Dedicated colorbars per panel (clearer than sharing when ranges differ)
        for ax, sc in zip(axs, [sc0, sc1, sc2, sc3]):
            cb = plt.colorbar(sc, ax=ax)
            # cb.ax.tick_params(labelsize=xy_tick_size)
        plt.suptitle(f'Layer {layer_n}', fontsize=sup_title_size)
        plt.tight_layout()
        plt.show()

    # with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
    #     ir_exe = pickle.load(f)
    # ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')
    # thermal_centroid_record = []
    # for (ir_id,ir_image_raw, stamp) in zip(range(len(ir_exe)), ir_exe, ir_stamp):
    #     ir_image = deepcopy(ir_image_raw)
    #     img_height, img_width = ir_image.shape

if __name__ == '__main__':
    main()