import numpy as np
import sys, traceback, time, copy, glob
import matplotlib.pyplot as plt
from matplotlib import cm
from general_robotics_toolbox import *
sys.path.append('../../toolbox')
from robot_def import *
from multi_robot import *
from error_check import *
from lambda_calc import *

def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice2/'
	data_dir='../../data/'+dataset+sliced_alg
	num_layers=5
	
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun_old.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
		tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv',\
		base_marker_config_file=config_dir+'D500B_marker_config.yaml',\
		tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


	positioner_js=[]
	robot_js=[]
	for i in range(num_layers):
		positioner_js.extend(np.loadtxt(data_dir+'curve_sliced_js/D500B_js%i_0.csv'%i,delimiter=','))
		robot_js.extend(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js%i_0.csv'%i,delimiter=','))
	positioner_js=np.array(positioner_js)
	robot_js=np.array(robot_js)
	positioner_pose=positioner.fwd(positioner_js)
	robot_pose=robot.fwd(robot_js)
	relative_path_exe,relative_path_exe_R=form_relative_path_mocap(robot_pose.p_all,robot_pose.R_all,positioner_pose.p_all,positioner_pose.R_all,robot,positioner)
	ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'g.-')

	plt.show()
if __name__ == '__main__':
	main()