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

data_dir='../../data/wall_weld_test/test4/150_ipm_1st_layer/'
config_dir='../../config/'

all_welds_width = pickle.load(open(data_dir+'all_welds_width.pickle','rb'))
all_welds_height = pickle.load(open(data_dir+'all_welds_height.pickle','rb'))

for weld_i in range(len(all_welds_width)):
    welds_width=all_welds_width[weld_i]
    welds_height=all_welds_height[weld_i]

    all_z = np.array(list(welds_width.keys()))
    all_x = []
    all_points = None
    all_width = []
    all_height = []
    all_height_interp=[]
    all_height_draw={}
    for z in all_z:
        this_x=np.array(list(welds_width[z].keys()))
        all_x.append(this_x)

        for x in this_x:
            if x not in all_height_draw.keys():
                all_height_draw[x]=deepcopy(welds_height[z][x])
            else:
                if welds_height[z][x]>all_height_draw[x]:
                    all_height_draw[x]=deepcopy(welds_height[z][x])

        ### sample points (z,x)
        if all_points is None:
            all_points=np.vstack((np.ones(this_x.size)*z,this_x)).T
        else:
            this_points = np.vstack((np.ones(this_x.size)*z,this_x)).T
            all_points = np.vstack((all_points,this_points))
        ### sample data width and height
        all_width.extend(list(welds_width[z].values()))
        all_height.extend(list(welds_height[z].values()))

        ### interp height
        # all_height_interp.append(interp1d(this_x, np.array(list(welds_height[z].values())),kind='cubic'))
    all_width=np.array(all_width)
    all_height=np.array(all_height)

    #### plot the height in x-axis
    height_x_draw = np.array(list(all_height_draw.keys()))
    height_draw = np.array(list(all_height_draw.values()))
    height_x_sort_i = np.argsort(height_x_draw)
    height_x_draw = height_x_draw[height_x_sort_i]
    height_draw = height_draw[height_x_sort_i]
    plt.plot(height_x_draw,height_draw)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('x-axis (mm)',fontsize=16)
    plt.ylabel('heights (mm)',fontsize=16)
    plt.title("Heights (mm), Weld "+str(weld_i),fontsize=20)
    plt.show()

    ### plot width histogram
    plot_width_index = np.argwhere(np.array(all_width)!=0)[:,0]
    all_points_with_width = all_points[plot_width_index]
    all_width_plot = all_width[plot_width_index]

    x_min=np.min(all_points_with_width[:,1])
    x_max=np.max(all_points_with_width[:,1])
    x_resolution=0.1
    x_total = (x_max-x_min)/x_resolution
    z_min=np.min(all_points_with_width[:,0])
    z_max=np.max(all_points_with_width[:,0])
    # z_resolution=(z_max-z_min)/x_total
    z_resolution=0.05
    grid_x, grid_y = np.mgrid[z_min:z_max:z_resolution, x_min:x_max:x_resolution]
    grid_width = griddata(all_points_with_width, all_width_plot, (grid_x, grid_y), method='cubic')
    grid_width = np.clip(grid_width,0,np.max(all_width_plot))

    while True:
        h_input = input("Which height (mm) do you want? (press q to leave) ")
        if h_input == 'q':
            break
        h_input = float(h_input)
        wid = np.argsort(np.abs(grid_x[:,0]-h_input))[0]
        data = deepcopy(grid_width[wid])
        data[data < 0.5] = np.nan
        print("Mean Width",np.nanmean(data), "at",h_input)
        print("Max Width",np.nanmax(data), "at",h_input)
        print("Min Width",np.nanmin(data), "at",h_input)

    plt.imshow(grid_width[::-1])
    plt.xticks(np.arange(len(grid_y[0]))[::int(len(grid_y[0])/10)],np.round(grid_y[0][::int(len(grid_y[0])/10)],1),fontsize=14)
    plot_y = grid_x[:,0][::-1]
    plt.yticks(np.arange(len(plot_y))[::int(len(plot_y)/5)],np.round(plot_y[::int(len(plot_y)/5)],1),fontsize=14)
    cbar=plt.colorbar(location='bottom')
    cbar.ax.tick_params(labelsize=14)
    plt.xlabel('x-axis (mm)',fontsize=16)
    plt.ylabel('z-axis (mm)',fontsize=16)
    plt.title("Width Map (mm), Weld "+str(weld_i),fontsize=20)
    plt.show()