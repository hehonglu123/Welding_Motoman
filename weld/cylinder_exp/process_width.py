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

def find_width(point_cloud):

	flattened_point_cloud=point_cloud[:,:2]
	radius=np.mean(np.linalg.norm(flattened_point_cloud,axis=1))
	#generate points along circle of radius
	num_points=1000
	thetas=np.linspace(0,2*np.pi,num_points)
	width_all=[]
	for theta in tqdm(thetas):
		line_v = np.array([np.cos(theta), np.sin(theta)])  # Ensure line_v is 3D

		# # Compute the cross product for each point with the line vector
		# distances = flattened_point_cloud - flattened_point_cloud@line_v[:,np.newaxis]
		# distances = np.linalg.norm(distances, axis=1)


		# # Compute the direction of each point relative to the line
		# directions = np.sign(np.dot(flattened_point_cloud, line_v))

		# # Find points within the threshold distance
		# close_points = flattened_point_cloud[(distances <= 2) & (directions > 0)]

		# if close_points.size == 0:
		# 	continue

		# # Find the point closest to the origin
		# closest_point = close_points[np.argmin(np.linalg.norm(close_points, axis=1))]@line_v

		# # Find the point farthest from the origin
		# farthest_point = close_points[np.argmax(np.linalg.norm(close_points, axis=1))]@line_v

		p_close=line_v*(radius-5)
		closest_point=flattened_point_cloud[np.argmin(np.linalg.norm(flattened_point_cloud-p_close,axis=1))]@line_v
		p_far=line_v*(radius+5)
		farthest_point=flattened_point_cloud[np.argmin(np.linalg.norm(flattened_point_cloud-p_far,axis=1))]@line_v

		width_all.append(np.linalg.norm(farthest_point - closest_point))

	#remove outliers from the intersection
	width_all=np.array(width_all)
	# width_all=width_all[width_all>2]

	# plt.plot(width_all)
	# plt.show()
	print("Number of points: ", len(width_all))
	return np.mean(width_all), np.std(width_all)


	


def main():
	
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

	width_mean_all=[]
	width_std_all=[]
	v_all=np.arange(5,21)
	for v in v_all:
		recorded_dir='../../../recorded_data/ER316L/VPD10/tubespiral_%iipm_v%i/'%(10*v,v)
		joint_recording=np.loadtxt(recorded_dir+'scan_js_exe.csv',delimiter=',')
		with open(recorded_dir+'mti_scans.pickle', 'rb') as f:
			mti_recording = pickle.load(f)
		
		#####raw pcd process
		torch_height=44
		scan_process=ScanProcess(robot_scan, positioner)
		pcd = scan_process.pcd_register_mti(mti_recording,joint_recording[:,-8:],joint_recording[:,1])
		crop_min=(-50,-30,torch_height-10)
		crop_max=(50,30,torch_height+10)
		pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.0,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=400,min_bound=crop_min,max_bound=crop_max)
		
		#filter the points near z axis (x^2+y^2<10)
		pcd_points=np.asarray(pcd.points)
		pcd_points=pcd_points[np.linalg.norm(pcd_points[:,:2],axis=1)>10]
		pcd.points=o3d.utility.Vector3dVector(pcd_points)

		width_mean,width_std=find_width(pcd_points)
		width_mean_all.append(width_mean)
		width_std_all.append(width_std)
	
	plt.errorbar(v_all, width_mean_all, yerr=width_std_all, fmt='-o', capsize=5)
	plt.xlabel('Speed (mm/s)')
	plt.ylabel('Width (mm)')
	plt.title('Layer Width Scan')
	plt.show()
	
	

if __name__ == '__main__':
	main()