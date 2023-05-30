import time, sys, pickle
import open3d as o3d

sys.path.append('../toolbox/')
from utils import *
from robot_def import *

def raw_data_filtering(single_scan, noise_filter=False):
    ###delete noise centered at x=0
    single_scan=np.delete(single_scan,np.argwhere(abs(single_scan[:,0])==0).flatten(),axis=0)
    ###clip data within range
    indices=np.argwhere(single_scan[:,1]>50).flatten()
    single_scan=single_scan[indices]

    if noise_filter:
        ###filter out outlier noise
        outlier_indices=identify_outliers2(single_scan[:,1],rolling_window=20,threshold=1e-2)
        return np.delete(single_scan,outlier_indices,axis=0)
    else:
        return single_scan


robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

joint_recording=np.loadtxt('recording3/joint_recording.csv',delimiter=',')
with open('recording3/mti_recording.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

pc=[]
for i in range(0,len(mti_recording),5):
    line_scan=raw_data_filtering(mti_recording[i].T,noise_filter=False) ###filter out bad points
    # line_scan[:,0]=-line_scan[:,0]
    line_scan=np.hstack((np.zeros((len(line_scan),1)),line_scan))

    pose=robot2.fwd(joint_recording[i])
    pc.extend(transform_curve(line_scan,H_from_RT(pose.R,pose.p)))

pointcloud=o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pointcloud])
