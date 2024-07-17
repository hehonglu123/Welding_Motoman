import numpy as np
from general_robotics_toolbox import *
import sys
from dual_robot import *
from motoman_def import *
import matplotlib.pyplot as plt
from lambda_calc import *

def main():
	
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='config/'
	# data_dir='../recorded_data/ER316L/streaming/cylinderspiral_T19000/'
	data_dir='../recorded_data/ER316L/VPD10/cylinderspiral_50ipm_v5/'

	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')
	#sort by time
	joint_angle=joint_angle[joint_angle[:,1].argsort()]
	#get rid of index with same timestamp
	joint_angle=joint_angle[np.insert(np.diff(joint_angle[:,1]) != 0, 0, True)]

	curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,relative_path_exe,relative_path_exe_R = form_relative_path(joint_angle[:,-14:-8],joint_angle[:,-2:],robot,positioner)
	lam = calc_lam_cs(relative_path_exe)
	###plot the 3d path
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.plot(relative_path_exe[:,0],relative_path_exe[:,1],relative_path_exe[:,2])
	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_zlabel('Z')
	# plt.show()

	###speed profile
	speed=np.gradient(lam)/np.gradient(joint_angle[:,1])
	print("average speed: ",np.mean(speed))
	plt.plot(joint_angle[:,0],speed)
	plt.title('Speed Profile')
	plt.xlabel('Time (s)')
	plt.ylabel('Speed (mm/s)')
	plt.show()

if __name__ == "__main__":
	main()