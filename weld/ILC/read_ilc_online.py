from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from utils import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import open3d as o3d

data_dir='data/weld_scan_2023_11_01_17_44_58/'
seg_dist=1.6
dh=2.5
yk_d=[dh]

total_iteration=9

show_pcd=[0,8]
for iter_read in range(total_iteration):
    if iter_read in show_pcd:
        pcd_all=o3d.geometry.PointCloud()
        for layer in range(2):
            pcd=o3d.io.read_point_cloud(data_dir+'iteration_'+str(iter_read)+'/half_0/layer_'+str(layer)+'/processed_pcd.pcd')
            if layer==0:
                pcd=pcd.paint_uniform_color([0.1,0.1,0.1])
            elif layer==1:
                pcd=pcd.paint_uniform_color([1,0,0])
            pcd_all+=pcd
        visualize_pcd([pcd_all])

plt.axhline(y = yk_d[0], color = 'r', linestyle = '-',label='desired h') 
for iter_read in range(total_iteration):

    yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
    # try:
    #     yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
    # except:
    #     yk_prime=deepcopy(yk)
    plt.plot(np.arange(len(yk))*seg_dist,yk,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
    # plt.scatter(np.arange(len(yk_prime)),yk_prime)
plt.xlabel("L (mm)",fontsize=14)
plt.ylabel("Height (mm)",fontsize=14)
plt.legend()
plt.title("Output height of iterations",fontsize=14)
plt.show()

plt.axhline(y = yk_d[0], color = 'r', linestyle = '-',label='desired h') 
for iter_read in range(total_iteration):

    uk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/half_0/input_uk.csv',delimiter=',')
    # try:
    #     yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
    # except:
    #     yk_prime=deepcopy(yk)
    plt.plot(np.arange(len(uk))*seg_dist,uk,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
    # plt.scatter(np.arange(len(yk_prime)),yk_prime)
plt.xlabel("L (mm)",fontsize=14)
plt.ylabel("Height (mm)",fontsize=14)
plt.legend()
plt.title("Input u of iterations",fontsize=14)
plt.show()

plt.axhline(y = 0, color = 'r', linestyle = '-',label='zero')
for iter_read in range(total_iteration):

    yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
    yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
    ek_prime=yk_prime-yk
    gradient_direction=np.flip(ek_prime)
    plt.plot(np.arange(len(gradient_direction))*seg_dist,gradient_direction,marker='o',markersize=4,linewidth=2,label='iter '+str(iter_read))
    # plt.scatter(np.arange(len(yk_prime)),yk_prime)
plt.xlabel("L (mm)",fontsize=14)
plt.ylabel("Height (mm)",fontsize=14)
plt.legend()
plt.title("gradient direction of iterations",fontsize=14)
plt.show()

norm_iter=[]
for iter_read in range(total_iteration):
    yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
    norm_iter.append(np.linalg.norm(yk-dh))
plt.plot(norm_iter,'-o',markersize=6,linewidth=2)
plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Error Norm (mm)",fontsize=14)
plt.legend()
plt.title("Error norm of iterations",fontsize=14)
plt.show()

# norm of each part
norm_iter=[]
for iter_read in range(total_iteration):
    yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
    yk=np.reshape(yk,(9,5))
    norm_iter.append(np.squeeze(np.linalg.norm(yk-dh,axis=1)))
norm_iter=np.array(norm_iter)
draw_part=[0,1,4,7,8]
for part_n in range(9):
    if part_n in draw_part:  
        plt.plot(norm_iter[:,part_n],'-o',markersize=6,linewidth=2,label='part '+str(part_n))
plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Error Norm of parts (mm)",fontsize=14)
plt.legend()
plt.title("Error norm of parts of iterations",fontsize=14)
plt.show()