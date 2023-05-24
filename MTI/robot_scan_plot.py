import time, sys
import open3d as o3d
from utils import *

sys.path.append('../toolbox/')
from robot_def import *

robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

joint_recording=np.loadtxt('joint_recording.csv',delimiter=',')
mti_recording=np.loadtxt('mti_recording.csv',delimiter=',')
pc=[]
for i in range(len(mti_recording)):
    line_scan=np.hstack((np.zeros((len(mti_recording[i]),1)),mti_recording[i]))
    pose=robot2.fwd(joint_recording[i])
    pc.extend(transform_curve(line_scan,pose))

pointcloud=o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pointcloud])
