import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from scan_utils import *
from scanPathGen import *
from scanProcess import *
from weldRRSensor import *
from RobotRaconteur.Client import *
from copy import deepcopy
from pathlib import Path
import datetime

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15,\
    base_marker_config_file='../config/MA2010_marker_config.yaml',tool_marker_config_file='../config/weldgun_marker_config.yaml')
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv',\
    base_marker_config_file='../config/MA1440_marker_config.yaml')
robot2_mti=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv',\
    base_marker_config_file='../config/MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv',\
    base_marker_config_file='../config/D500B_marker_config.yaml',tool_marker_config_file='../config/positioner_tcp_marker_config.yaml')

#### change base H to calibrated ones ####
robot_scan_base = robot.T_base_basemarker.inv()*robot2.T_base_basemarker
robot2.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
robot2_mti.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

layer=15
x=0
data_dir='../data/wall/dense_slice/'
recorded_dir=data_dir+'weld_scan_job200_v52023_07_26_12_51_35/'
layer_data_dir=recorded_dir+'layer_'+str(layer)+'/'
out_scan_dir = layer_data_dir+'scans/'

curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
q_out_exe=np.loadtxt(out_scan_dir + 'scan_js_exe.csv',delimiter=',')
robot_stamps=np.loadtxt(out_scan_dir + 'scan_robot_stamps.csv',delimiter=',')
q_init_table=q_out_exe[0][-2:]
with open(out_scan_dir + 'mti_scans.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

h_largest=1.5
curve_x_start = deepcopy(curve_sliced_relative[0][0])
curve_x_end = deepcopy(curve_sliced_relative[-1][0])
Transz0_H=None
z_height_start=h_largest-5
crop_extend=10
crop_min=(curve_x_start-crop_extend,-30,-10)
crop_max=(curve_x_end+crop_extend,30,z_height_start+30)
crop_h_min=(curve_x_start-crop_extend,-20,-10)
crop_h_max=(curve_x_end+crop_extend,20,z_height_start+30)
scan_process = ScanProcess(robot2_mti,positioner)
pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
# visualize_pcd([pcd])
pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                    min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
# visualize_pcd([pcd])
profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
print("Transz0_H:",Transz0_H)

save_output_points=True
if save_output_points:
    o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
    np.save(out_scan_dir+'height_profile.npy',profile_height)
visualize_pcd([pcd])
plt.scatter(profile_height[:,0],profile_height[:,1])
plt.show()