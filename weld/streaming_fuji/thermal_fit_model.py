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
from qpsolvers import solve_qp

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
    
    visualize_thermal_map = False
    # parameters
    heat_input_mask_window = 1.2 # in mm, the window size of the heat input around the weld
    thermal_x_sample = np.arange(-81, 98, 0.5) # in mm , relative to weld torch center
    ambient_temp = 8000 # in "temperature unit" (not degree C)
    
    ### example data ##
    ## logdata_dir_name = 'weld_fujiscan_2025_06_12_15_33_03/'
    ## layer_n=311

    thermal_map = [] # a array with Nx4, each row is time,x,temperature,velocity,wire feed rate (related to power inputs)
    thermal_dx = [] # a array with Nx3, each row is time,x,dT/dx
    thermal_dx2 = [] # a array with Nx3, each row is time,x,d2T/dx2
    thermal_dt = [] # a array with Nx3, each row is time,x,dT/dt
    
    # log data directory
    logdata_dir_name_all = ['weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_17_33_24/']
    # logdata_dir_name_all = ['weld_fujiscan_2025_06_12_17_33_24/']

    for logdata_dir_name in logdata_dir_name_all:
        print("===============================")
        print("processing data folder: ", logdata_dir_name)
        logdata_dir = data_dir+logdata_dir_name

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
            #### debug only one layer ####
            # if layer_n != 311:
            #     continue
            # if layer_n not in [311,330]:
            #     continue
            if layer_n_id < len(layer_nums)-6:
                continue
            ##############################

            print("==========")
            print("processing layer %d, %d/%d"%(layer_n, layer_n_id+1, len(layer_nums)))

            ##### layer, basic infos ####
            layer_name = 'layer'+str(layer_n)
            this_layer_dir = logdata_dir+layer_name+'/'

            ##### read robot weld exe pose and stamp ####
            weld_relative_exe = np.loadtxt(this_layer_dir+'weld_relative_exe.csv', delimiter=',') # in mm
            weld_relative_v_exe = np.loadtxt(this_layer_dir+'weld_relative_v_exe.csv', delimiter=',') # in mm/s
            weld_relative_speed_exe = deepcopy(weld_relative_v_exe)*np.sign(weld_relative_exe[-1,0]-weld_relative_exe[0,0]) # in mm/s, consider the direction
            print("weld relative speed sign: ", np.sign(weld_relative_exe[-1,0]-weld_relative_exe[0,0]))
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
            layer_thermal_map = [] # a array with Nx4, each row is time,x,temperature,velocity,wire feed rate (related to power inputs)
            layer_thermal_map_contain_zero = []
            layer_thermal_dx = [] # a array with Nx3, each row is time,x,dT/dx
            layer_thermal_dx2 = [] # a array with Nx3, each row is time,x,d2T/dx2
            heat_input_mask = ((thermal_x_sample>=-heat_input_mask_window) & (thermal_x_sample<=heat_input_mask_window)).astype(float)
            thermal_stamp_all = np.sort(np.array(list(thermal_pixel_trace.keys())))
            for stamp in thermal_stamp_all:
                # find current weld x position
                weld_id = np.argmin(np.abs(weld_stamps - stamp))
                weld_cmd_id = np.argmin(np.abs(weld_cmd[:,0] - stamp))
                weld_x = weld_relative_exe[weld_id,0]
                this_stamp_thermal_sample_zero = np.interp(thermal_x_sample, thermal_pixel_trace[stamp]['x']-weld_x, thermal_pixel_trace[stamp]['value']-ambient_temp, left=0, right=0)
                this_stamp_thermal_sample = np.interp(thermal_x_sample, thermal_pixel_trace[stamp]['x']-weld_x, thermal_pixel_trace[stamp]['value']-ambient_temp)
                layer_thermal_map.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp), thermal_x_sample, this_stamp_thermal_sample,\
                                                np.ones_like(thermal_x_sample)*weld_relative_speed_exe[weld_id],\
                                                heat_input_mask*weld_cmd[weld_cmd_id, -1])  ).T.tolist() )
                layer_thermal_map_contain_zero.extend(np.vstack( (np.ones_like(thermal_x_sample)*(stamp), thermal_x_sample, this_stamp_thermal_sample_zero,\
                                                np.ones_like(thermal_x_sample)*weld_relative_speed_exe[weld_id],\
                                                heat_input_mask*weld_cmd[weld_cmd_id, -1])  ).T.tolist() )
                # this_stamp_thermal_sample_dx = np.gradient(this_stamp_thermal_sample, thermal_x_sample)
                this_stamp_thermal_sample_dx = savgol_filter(this_stamp_thermal_sample, 11, 3, deriv=1, delta=0.5) # window size 11, polynomial order 3
                layer_thermal_dx.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp), thermal_x_sample, this_stamp_thermal_sample_dx) ).T.tolist() )
                # this_stamp_thermal_sample_dx2 = np.gradient(this_stamp_thermal_sample_dx, thermal_x_sample) # second derivative
                this_stamp_thermal_sample_dx2 = savgol_filter(this_stamp_thermal_sample, 11, 3, deriv=2, delta=0.5) # window size 11, polynomial order 3
                layer_thermal_dx2.extend( np.vstack( (np.ones_like(thermal_x_sample)*(stamp), thermal_x_sample, this_stamp_thermal_sample_dx2) ).T.tolist() )
            layer_thermal_map = np.array(layer_thermal_map)
            layer_thermal_map_contain_zero = np.array(layer_thermal_map_contain_zero)
            layer_thermal_dx = np.array(layer_thermal_dx)
            layer_thermal_dx2 = np.array(layer_thermal_dx2)

            layer_thermal_dt = np.zeros_like(layer_thermal_map)
            for x in thermal_x_sample:
                this_x_id = np.where(layer_thermal_map[:,1]==x)[0]
                this_x_stamp = layer_thermal_map[this_x_id,0]
                # this_dthermal_dt = np.gradient(thermal_map[this_x_id,2], this_x_stamp)
                this_dthermal_dt = savgol_filter(layer_thermal_map[this_x_id,2], 11, 3, deriv=1, delta=np.mean(np.diff(this_x_stamp))) # window size 11, polynomial order 3
                layer_thermal_dt[this_x_id,2] = this_dthermal_dt
                layer_thermal_dt[this_x_id,0] = this_x_stamp
                layer_thermal_dt[this_x_id,1] = x

            # get rid of zero value in thermal map
            non_zero_index = np.where(layer_thermal_map_contain_zero[:,2]!=0)[0] # get rid of zero value
            layer_thermal_map = layer_thermal_map[non_zero_index]
            layer_thermal_dx = layer_thermal_dx[non_zero_index]
            layer_thermal_dx2 = layer_thermal_dx2[non_zero_index]
            layer_thermal_dt = layer_thermal_dt[non_zero_index]

            # add current layer data to total data
            thermal_map.extend(layer_thermal_map.tolist())
            thermal_dx.extend(layer_thermal_dx.tolist())
            thermal_dx2.extend(layer_thermal_dx2.tolist())
            thermal_dt.extend(layer_thermal_dt.tolist())

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
                    axs[0], TM[:, :2], TM[:, 2]-ambient_temp, "thermal map",
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

    # solve pointwise linear regression
    assert len(thermal_map)==len(thermal_dx) and len(thermal_map)==len(thermal_dt) and len(thermal_map)==len(thermal_dx2)
    print("total thermal data points: ", len(thermal_map))

    # convert to np array
    thermal_map = np.array(thermal_map)
    thermal_dx = np.array(thermal_dx)
    thermal_dx2 = np.array(thermal_dx2)
    thermal_dt = np.array(thermal_dt)

    # check if each index x and stamps are the same
    for data_i in range(len(thermal_map)):
        assert thermal_map[data_i,0]==thermal_dx[data_i,0] and thermal_map[data_i,0]==thermal_dt[data_i,0] and thermal_map[data_i,0]==thermal_dx2[data_i,0]
        assert thermal_map[data_i,1]==thermal_dx[data_i,1] and thermal_map[data_i,1]==thermal_dt[data_i,1] and thermal_map[data_i,1]==thermal_dx2[data_i,1]
    print("data check pass!")

    # sort thermal map according to stamp and x
    thermal_map_sort_stamp_idx = np.argsort(thermal_map[:,0])
    thermal_map = thermal_map[thermal_map_sort_stamp_idx]
    for stamp in thermal_stamp_all:
        stamp_mask = thermal_map[:,0] == stamp
        # sort this stamp data according to x
        stamp_sort_x_idx = np.argsort(thermal_map[stamp_mask,1])
        thermal_map[stamp_mask] = thermal_map[stamp_mask][stamp_sort_x_idx]
    print("data sorting done!")

    # phi_mat = [U_yy, -U, p(t)Pi(y)]
    phi_mat = np.vstack((thermal_dx2[:,2], -thermal_map[:,2], thermal_map[:,4])).T
    # b = [U_t - v(t)U_y]
    b_vector = thermal_dt[:,2] - thermal_map[:,3] * thermal_dx[:,2]

    # Preconditions (good hygiene)
    phi_mat = np.ascontiguousarray(phi_mat, dtype=np.float64)
    b_vector = np.ascontiguousarray(b_vector, dtype=np.float64)

    # normalize columns of phi_mat to avoid ill-conditioned P matrix
    # col_norms = np.linalg.norm(phi_mat, axis=0)
    # col_norms[col_norms == 0] = 1.0
    # phi_mat = phi_mat / col_norms

    print("start solving QP...")
    # solve qp problem
    # min ||phi_mat theta - b||^2
    # where theta = [k, alpha, beta] >=0
    P = (phi_mat.T @ phi_mat)
    P = 0.5 * (P + P.T) # make sure P is symmetric
    # q = - (b_vector.T @ phi_mat).T
    q = -(phi_mat.T @ b_vector)
    q = np.ascontiguousarray(q.reshape(-1), dtype=np.float64)
    G = -np.eye(3, dtype=np.float64)
    h = np.zeros(3, dtype=np.float64)
    print("QP Sanity check:")
    print("Shapes: P", P.shape, "q", q.shape, "G", G.shape, "h", h.shape)
    print("dtypes:", P.dtype, q.dtype, G.dtype, h.dtype)
    print("eig(P) min/max:", np.linalg.eigvalsh(P).min(), np.linalg.eigvalsh(P).max())
    print("||Gx-h|| feasibility test with θ=0:", np.max(G @ np.zeros(3) - h))  # should be <= 0
    print("Condition number of P (2-norm):", np.linalg.cond(P))

    assert np.all(np.isfinite(phi_mat)), "NaNs/Infs in phi_mat"
    assert np.all(np.isfinite(b_vector)), "NaNs/Infs in b_vector"

    # theta = solve_qp(P, q, G, h, solver='quadprog')
    theta = solve_qp(P, q, lb=h, solver='osqp')
    print("QP solved!")
    print(theta)
    print("k: %.4f, alpha: %.4f, beta: %.4f"%(theta[0], theta[1], theta[2]))

    # direction psuedo inverse
    theta_pinv = np.linalg.pinv(phi_mat) @ b_vector
    print("psuedo inverse solved!")
    print(theta_pinv)
    print("k: %.4f, alpha: %.4f, beta: %.4f"%(theta_pinv[0], theta_pinv[1], theta_pinv[2]))

    # fitting error validation #
    # r(y,t) = U_t - (k U_yy + v(t)U_y - alpha U + beta p(t)Pi(y))
    res_error = thermal_dt[:,2] - (theta[0]*thermal_dx2[:,2] + thermal_map[:,3]*thermal_dx[:,2] - theta[1]*thermal_map[:,2] + theta[2]*thermal_map[:,4])
    print("fitting error mean: %.4f, std: %.4f"%(np.mean(np.abs(res_error)), np.std(np.abs(res_error))))
    # visualize the fitting error distribution
    plt.figure(figsize=(8,6))
    plt.hist(np.abs(res_error), bins=500)
    plt.title('Fitting error distribution histogram', fontsize=title_size)
    plt.xlabel('Fitting error (dU/dt)', fontsize=xy_label_size)
    plt.ylabel('Counts', fontsize=xy_label_size)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=xy_tick_size)
    plt.tight_layout()
    plt.show()

    ##### strong-form regression #####
    


    # with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
    #     ir_exe = pickle.load(f)
    # ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')
    # thermal_centroid_record = []
    # for (ir_id,ir_image_raw, stamp) in zip(range(len(ir_exe)), ir_exe, ir_stamp):
    #     ir_image = deepcopy(ir_image_raw)
    #     img_height, img_width = ir_image.shape

if __name__ == '__main__':
    main()