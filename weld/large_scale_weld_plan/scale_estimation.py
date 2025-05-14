import time, os, copy, sys, yaml, pathlib
import traceback
from copy import deepcopy
import numpy as np
import datetime
from motoman_def import *
from lambda_calc import *
import open3d as o3d
from robotics_utils import *
from general_robotics_toolbox import *

def main():
    ############## Robot definition ##############
    config_dir='../../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
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
    robot_thermal=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')
    r_weld_z = robot_weld.fwd(np.zeros(6))
    r_scan_z = robot_scan_motion.fwd(np.zeros(6))
    T_weld_scan = r_weld_z.inv()*r_scan_z
    dist_weld_scan = np.linalg.norm(T_weld_scan.p)
    ##############################################

    ### test object target size
    cylinder_radius = 500 # mm

    ### constraintes parameters
    base_radius = 25 # mm


if __name__ == "__main__":
    main()