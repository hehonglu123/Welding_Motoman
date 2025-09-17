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

# for plotting
xy_label_size = 18
xy_tick_size = 16
legend_size = 16
title_size = 20
sup_title_size = 20

cam_pixel_moving_ratio = 1.77 # 1.77 pixel per mm

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

    ##### read thermal data ####
    with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
        ir_exe = pickle.load(f)
    ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')

    thermal_centroid_record = []

    for (ir_id,ir_image_raw, stamp) in zip(range(len(ir_exe)), ir_exe, ir_stamp):
        ir_image = deepcopy(ir_image_raw)
        img_height, img_width = ir_image.shape
    