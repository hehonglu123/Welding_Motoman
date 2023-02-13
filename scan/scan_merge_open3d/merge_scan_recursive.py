import sys

from sklearn import cluster
sys.path.append('../../toolbox/')
from robot_def import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from copy import deepcopy
import colorsys
import math

def colormap(all_h):

    all_h=(1-all_h)*270
    all_color=[]
    for h in all_h:
        all_color.append(colorsys.hsv_to_rgb(h,0.7,0.9))
    return np.array(all_color)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0.8, 0])
    o3d.visualization.draw([inlier_cloud, outlier_cloud])

def visualize_pcd(show_pcd_list):
    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3,origin=[0,0,0])
    show_pcd_list.append(points_frame)
    o3d.visualization.draw_geometries(show_pcd_list,width=960,height=540)
    # o3d.visualization.draw(show_pcd_list,width=960,height=540)

data_dir='test2/'
config_dir='../../config/'

scan_resolution=5 #scan every 5 mm
scan_per_pose=3 # take 3 scan every pose

robot=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
    pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')

cart_p=[]
joints_p=np.loadtxt(data_dir+'scan_js.csv',delimiter=",", dtype=np.float64)
for i in range(len(joints_p)):
    joints_p[i][5] = -joints_p[i][5]
    joints_p[i][3] = -joints_p[i][3]
    joints_p[i][1] = 90 - joints_p[i][1]
    joints_p[i][2] = joints_p[i][2] + joints_p[i][1]

joints_p=np.radians(joints_p)
for q in joints_p:
    cart_p.append(robot.fwd(q))
print("Joint Space")
print(np.degrees(joints_p))
print("Cart Space")
print(cart_p)

curve=[]
curve_R=[]
curve_js=[]
total_step=0
for i in range(len(cart_p)-1):
    travel_vec=cart_p[i+1].p-cart_p[i].p	
    travel_dis=np.linalg.norm(travel_vec)
    travel_vec=travel_vec/travel_dis*scan_resolution
    print("Travel Vector:",travel_vec)
    print("Travel Distance:",travel_dis)

    xp=np.append(np.arange(cart_p[i].p[0],cart_p[i+1].p[0],travel_vec[0]),cart_p[i+1].p[0])
    yp=np.append(np.arange(cart_p[i].p[1],cart_p[i+1].p[1],travel_vec[1]),cart_p[i+1].p[1])
    zp=np.append(np.arange(cart_p[i].p[2],cart_p[i+1].p[2],travel_vec[2]),cart_p[i+1].p[2])
    print(len(xp))
    print(len(yp))
    print(len(zp))

    for travel_i in range(len(xp)):
        this_p=Transform(cart_p[i].R,[xp[travel_i],yp[travel_i],zp[travel_i]])
        curve.append(this_p.p)
        curve_R.append(this_p.R)
        if len(curve_js)!=0:
            curve_js.append(robot.inv(this_p.p,this_p.R,curve_js[-1])[0])
        else:
            curve_js.append(robot.inv(this_p.p,this_p.R,joints_p[0])[0])
    total_step+=len(xp)
curve=np.array(curve)
curve_js=np.array(curve_js)
print(curve)
# print(np.degrees(curve_js))
print("Total step:",total_step)

T_base_frame1 = Transform(curve_R[0],curve[0])

## move bricks to origin
T_origin_R=rot([0,0,1],np.radians(87.5))
T_origin=Transform(T_origin_R,np.dot(T_origin_R,-curve[0])+np.array([-3,-19.1,243.9]))
print(T_origin)

# total_scan=len(curve)
total_scan=2

## process param
voxel_size=0.1
colors = plt.get_cmap("tab20")(range(total_scan))
# pcd_combined.colors = o3d.utility.Vector3dVector(colors[:, :3])

pcd_combined = o3d.geometry.PointCloud()
for i in range(total_scan):
    pcd = o3d.geometry.PointCloud()
    for scan_i in range(scan_per_pose):
        points = np.loadtxt(data_dir + 'points_'+str(i)+'_'+str(scan_i)+'.csv',delimiter=",", dtype=np.float64)
        points = np.transpose(np.matmul(curve_R[i],np.transpose(points)))+curve[i]
        ## get the points closed to origin
        points = np.transpose(np.matmul(T_origin.R,np.transpose(points)))+T_origin.p

        ###### preprocessing
        ## to pcd
        scan_pcd = o3d.geometry.PointCloud()
        scan_pcd.points=o3d.utility.Vector3dVector(points)
        # visualize_pcd([pcd])
        # exit()
        ## voxel down sample
        scan_pcd = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
        ## paints
        scan_pcd.paint_uniform_color(colors[i,:3])
        ## add to combined pcd
        pcd = pcd+scan_pcd
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    pcd_combined += pcd
visualize_pcd([pcd_combined])
exit()