import numpy as np
import sys, traceback, time, copy, glob
import matplotlib.pyplot as plt
from matplotlib import cm
from general_robotics_toolbox import *
sys.path.append('../../../toolbox')
from robot_def import *
from multi_robot import *
from error_check import *
from lambda_calc import *

def main():
	
	config_dir='../../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun_old.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
		tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv',\
		base_marker_config_file=config_dir+'D500B_marker_config.yaml',\
		tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	joint_recording=np.loadtxt('slice1_0/joint_recording.csv',delimiter=',')
	timestamp=joint_recording[:,0]
	robot_js=joint_recording[:,2:8]
	positioner_js=joint_recording[:,-2:]

	_,_,_,_,relative_path_exe,relative_path_exe_R=form_relative_path(robot_js,positioner_js,robot,positioner)
	ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'g.-')
	speed=np.divide(np.linalg.norm(np.diff(relative_path_exe,axis=0),axis=1),np.diff(timestamp))



	plt.show()
	plt.plot(speed)
	plt.show()
if __name__ == '__main__':
	main()