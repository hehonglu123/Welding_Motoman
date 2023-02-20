from cProfile import label
import sys
import matplotlib

sys.path.append('../../toolbox/')
from robot_def import *
from utils import *
from general_robotics_toolbox import *
import open3d as o3d

import numpy as np
from scipy.interpolate import griddata,interp1d
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import time
from copy import deepcopy
import math
import pickle

table_colors = list(mcolors.TABLEAU_COLORS.values())

data_dir='../../data/wall_weld_test/test3_2/'
config_dir='../../config/'

all_welds_width = pickle.load(open(data_dir+'all_welds_width_recursive.pickle','rb'))
all_welds_height = pickle.load(open(data_dir+'all_welds_height_recursive.pickle','rb'))

for weld_i in range(len(all_welds_width)):
    welds_width=all_welds_width[weld_i]
    welds_height=all_welds_height[weld_i]

    all_z = np.array(list(welds_width.keys()))
    all_y = []
    all_points = None
    all_width = []
    all_height = []
    all_height_interp=[]
    all_height_draw={}
    for z in all_z:
        this_y=np.array(list(welds_width[z].keys()))
        all_y.append(this_y)

        for y in this_y:
            if y not in all_height_draw.keys():
                all_height_draw[y]=deepcopy(welds_height[z][y])
            else:
                if welds_height[z][y]>all_height_draw[y]:
                    all_height_draw[y]=deepcopy(welds_height[z][y])

        ### sample points (z,y)
        if all_points is None:
            all_points=np.vstack((np.ones(this_y.size)*z,this_y)).T
        else:
            this_points = np.vstack((np.ones(this_y.size)*z,this_y)).T
            all_points = np.vstack((all_points,this_points))
        ### sample data width and height
        all_width.extend(list(welds_width[z].values()))
        all_height.extend(list(welds_height[z].values()))

        ### interp height
        # all_height_interp.append(interp1d(this_y, np.array(list(welds_height[z].values())),kind='cubic'))
    all_width=np.array(all_width)
    all_height=np.array(all_height)

    #### plot the height in y-axis
    height_y_draw = np.array(list(all_height_draw.keys()))
    height_draw = np.array(list(all_height_draw.values()))
    height_y_sort_i = np.argsort(height_y_draw)
    height_y_draw = height_y_draw[height_y_sort_i]
    height_draw = height_draw[height_y_sort_i]
    plt.plot(height_y_draw,height_draw)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('y-axis (mm)',fontsize=16)
    plt.ylabel('heights (mm)',fontsize=16)
    plt.title("Heights (mm), Weld "+str(weld_i),fontsize=20)
    plt.show()

    ### plot height histogram
    # z_min=np.min(all_points[:,0])
    # z_max=np.max(all_points[:,0])
    # z_resolution=0.05
    # y_min=np.min(all_points[:,1])
    # y_max=np.max(all_points[:,1])
    # y_resolution=0.1
    # grid_x, grid_y = np.mgrid[z_min:z_max:z_resolution, y_min:y_max:y_resolution]
    # grid_height = griddata(all_points, all_height, (grid_x, grid_y), method='cubic')
    # plt.imshow(grid_height[::-1])
    # plt.xticks(np.arange(len(grid_y[0]))[::int(len(grid_y[0])/10)],np.round(grid_y[0][::int(len(grid_y[0])/10)],1),fontsize=14)
    # plot_y = grid_x[:,0][::-1]
    # plt.yticks(np.arange(len(plot_y))[::int(len(plot_y)/5)],np.round(plot_y[::int(len(plot_y)/5)],1),fontsize=14)
    # cbar=plt.colorbar(location='bottom')
    # cbar.ax.tick_params(labelsize=14)
    # plt.xlabel('y_axis',fontsize=16)
    # plt.ylabel('z_axis',fontsize=16)
    # plt.title("Height Map, Weld "+str(weld_i),fontsize=20)
    # plt.show()

    ### plot width histogram
    plot_width_index = np.argwhere(np.array(all_width)!=0)[:,0]
    all_points_with_width = all_points[plot_width_index]
    all_width_plot = all_width[plot_width_index]

    y_min=np.min(all_points_with_width[:,1])
    y_max=np.max(all_points_with_width[:,1])
    y_resolution=0.1
    y_total = (y_max-y_min)/y_resolution
    z_min=np.min(all_points_with_width[:,0])
    z_max=np.max(all_points_with_width[:,0])
    # z_resolution=(z_max-z_min)/y_total
    z_resolution=0.05
    grid_x, grid_y = np.mgrid[z_min:z_max:z_resolution, y_min:y_max:y_resolution]
    grid_width = griddata(all_points_with_width, all_width_plot, (grid_x, grid_y), method='cubic')
    grid_width = np.clip(grid_width,0,np.max(all_width_plot))
    plt.imshow(grid_width[::-1])
    plt.xticks(np.arange(len(grid_y[0]))[::int(len(grid_y[0])/10)],np.round(grid_y[0][::int(len(grid_y[0])/10)],1),fontsize=14)
    plot_y = grid_x[:,0][::-1]
    plt.yticks(np.arange(len(plot_y))[::int(len(plot_y)/5)],np.round(plot_y[::int(len(plot_y)/5)],1),fontsize=14)
    cbar=plt.colorbar(location='bottom')
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('y-axis (mm)',fontsize=16)
    plt.ylabel('z-axis (mm)',fontsize=16)
    plt.title("Width Map (mm), Weld "+str(weld_i),fontsize=20)
    plt.show()