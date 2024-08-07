import numpy as np
import pickle, sys
sys.path.append('../../scan/scan_process/')
from scanProcess import *
import open3d as o3d
from motoman_def import *

def main():
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	##############################################################Load Scan Data####################################################################
	out_scan_dir = 'scans/'
	robot_js=np.loadtxt(out_scan_dir+'scan_js_exe.csv',delimiter=',')
	with open(out_scan_dir + 'mti_scans.pickle','rb') as f:
		mti_recording = pickle.load(f)
	q_init_table=robot_js[0,-2:]
	scan_process = ScanProcess(robot_scan,positioner)
	pcd = scan_process.pcd_register_mti(mti_recording,robot_js[:,8:],robot_js[:,1],static_positioner_q=q_init_table)
	pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)

	#display pcd
	o3d.visualization.draw_geometries([pcd])



if __name__ == '__main__':
	main()