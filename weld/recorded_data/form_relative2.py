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
	sliced_alg='auto_slice/'
	data_dir='../data/'+dataset+sliced_alg
	curve_sliced_relative=[]
	num_layers=5
	
	layer_height_num=int(1.0/0.1)

	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
			tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv',\
		base_marker_config_file=config_dir+'D500B_marker_config.yaml',\
		tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	##########nominal relative path#######################
	positioner_js=[]
	robot_js=[]
	num_layer_start=int(0*layer_height_num)
	num_layer_end=int(5*layer_height_num)
	for layer in range(num_layer_start,num_layer_end,layer_height_num):
		positioner_js.extend(np.loadtxt(data_dir+'curve_sliced_js/D500B_js%i_0.csv'%layer,delimiter=','))
		robot_js.extend(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js%i_0.csv'%layer,delimiter=','))
	positioner_js=np.array(positioner_js)
	robot_js=np.array(robot_js)
	positioner_pose=positioner.fwd(positioner_js)
	robot_pose=robot.fwd(robot_js)
	relative_path_exe,relative_path_exe_R=form_relative_path_mocap(robot_pose.p_all,robot_pose.R_all,positioner_pose.p_all,positioner_pose.R_all,robot,positioner)
	ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'g.-')

	################mocap recorded path###########
	curve_exe_data=np.loadtxt('mocap_robot.csv',delimiter=',')
	curve_exe_positioner_data=np.loadtxt('mocap_positioner.csv',delimiter=',')
	curve_exe_positioner_data[:,3]+=380

	pos_T_bottombase_basemarker = positioner.T_base_basemarker*Transform(np.eye(3),-1*(positioner.robot.P[:,0]))
	T_posbase_robbase = robot.T_base_basemarker.inv()*pos_T_bottombase_basemarker
	positioner.base_H=np.array([[ 0.005,   1.,     -0.0021,1652.5628],
					[-0.9664,  0.0053,  0.2569,-910.7085],
					[ 0.2569,  0.0008,  0.9664,-800.2703],
					[0,0,0,1]])

	# positioner.base_H=robot.T_base_basemarker.inv()*positioner.T_base_basemarker
	# positioner.base_H=np.array([[ 0.005,   1.,     -0.0021,1651.7603 ],
	# 							[-0.9664,  0.0053,  0.2569,-813.0717],
	# 							[ 0.2569,  0.0008,  0.9664,-433.0287],
	# 							[0,0,0,1]])


	# relative_path_exe=[]
	# for i in range(len(curve_exe_data)):
	# 	if curve_exe_data[i][0]==curve_exe_positioner_data[i][0]:	#if timestamp aligns
	# 		relative_path_exe.append(curve_exe_data[i][1:4]-(T_posbase_robbase.R@curve_exe_positioner_data[i,1:4]+T_posbase_robbase.p))
	# relative_path_exe=np.array(relative_path_exe)
	
	relative_path_exe, relative_path_exe_R=form_relative_path_mocap(curve_exe_data[:,1:4],w2R(curve_exe_data[:,4:],np.eye(3)),curve_exe_positioner_data[:,1:4],w2R(curve_exe_positioner_data[:,4:],np.eye(3)),robot,positioner)

	

	ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'r.-')


	


	plt.show()
if __name__ == '__main__':
	main()