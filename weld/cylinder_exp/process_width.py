import numpy as np
import pickle, sys
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_process/')
from scan_utils import *
from scanProcess import *
from lambda_calc import *
import open3d as o3d
from motoman_def import *
from tqdm import tqdm


def main():
	dataset='cylinder/'
	sliced_alg='dense_slice/'
	data_dir='../../../geometry_data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	waypoint_distance=5 	###waypoint separation
	
	torch_height=44
	scan_stand_off_d = 95
	
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')


	recorded_dir='../../../recorded_data/ER316L/cylinder_scans/'
	joint_recording=np.loadtxt(recorded_dir+'scan_js_exe.csv',delimiter=',')
	with open(recorded_dir+'mti_scans.pickle', 'rb') as f:
		mti_recording = pickle.load(f)
	
	filtered_scan_points=[]
	width=[]
	valid_scan_indices=[]
	
	for scan_i in tqdm(range(len(joint_recording))):
		#filter the points within the torch height
		scan_points=mti_recording[scan_i]
		#display the scan points in 2d matplotlib 
		scan_points = scan_points[:, (scan_points[1] > scan_stand_off_d - 10) & (scan_points[1] < scan_stand_off_d + 10)]

		filtered_scan_points.append(scan_points)
		if scan_points[0].size > 0:
			valid_scan_indices.append(scan_i)
			width.append(np.max(scan_points[0])-np.min(scan_points[0]))

	lam=calc_lam_js(joint_recording[:,-8:-2],robot)
	plt.plot(lam[valid_scan_indices],width)
	plt.xlabel('$\lambda$')
	plt.ylabel('Width (mm)')
	plt.show()
		

	

	# #####display  raw pcd
	# scan_process=ScanProcess(robot_scan, positioner)
	# pcd = scan_process.pcd_register_mti(mti_recording,joint_recording[:,-8:],joint_recording[:,1])
	# crop_min=(-50,-30,torch_height-10)
	# crop_max=(50,30,torch_height+10)
	# pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100,min_bound=crop_min,max_bound=crop_max)
	# o3d.visualization.draw_geometries([pcd])

	#####display filtered pcd
	scan_process=ScanProcess(robot_scan, positioner)
	pcd = scan_process.pcd_register_mti(filtered_scan_points,joint_recording[:,-8:],joint_recording[:,1])
	pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
	o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
	main()