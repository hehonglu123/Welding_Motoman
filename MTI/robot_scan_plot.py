import time, sys, pickle
import open3d as o3d

sys.path.append('../toolbox/')
from utils import *
from robot_def import *

robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

joint_recording=np.loadtxt('recording1/joint_recording.csv',delimiter=',')
with open('recording1/mti_recording.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

pc=[]
for i in range(len(mti_recording)):
    line_scan=np.vstack((np.zeros(len(mti_recording[i][0])),mti_recording[i])).T
    pose=robot2.fwd(joint_recording[i])
    pc.extend(transform_curve(line_scan,H_from_RT(pose.R,pose.p)))

pointcloud=o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pointcloud])
