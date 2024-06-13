import cv2,copy, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *
from robot_def import *


# Load the IR recording data from the pickle file
# data_dir='../../../recorded_data/wall_bf_100ipm_v10/'
data_dir='../../../recorded_data/wallbf_100ipm_v10_80ipm_v8/'
# data_dir='../../../recorded_data/wallbf_100ipm_v10_120ipm_v12/'
config_dir='../../config/'
joint_angle=np.loadtxt(data_dir+'weld_js_exe.csv',delimiter=',')


robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')



#calculate the velocity
p_all=robot.fwd(joint_angle[:,2:8]).p_all
v_all=np.linalg.norm(np.diff(p_all,axis=0),axis=1)/np.diff(joint_angle[:,0])
plt.plot(joint_angle[1:,0],v_all)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm/s)')
plt.show()