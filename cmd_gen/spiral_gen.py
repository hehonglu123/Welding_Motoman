import numpy as np
from pandas import *
import sys, traceback, glob, fnmatch, os
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox/')
from robot_def import *


data_dir='../data/spiral_cylinder/'
solution_dir='baseline/'
robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

curve_js=np.loadtxt(data_dir+solution_dir+'curve_sliced_js/0.csv',delimiter=',')

primitives_choices=['movej']
cmd_dir=data_dir+solution_dir+'1C/'
points=[[robot.fwd(curve_js[0]).p]]
breakpoints=[0]
q_bp=[[np.array(curve_js[0])]]


num_layers=50

indices=np.linspace(0,len(curve_js)-1,num_layers,endpoint=False).astype(int)

for i in range(len(indices)-2):

	primitives_choices.append('movec_fit')

	points.append([robot.fwd(curve_js[indices[i+1]]).p,robot.fwd(curve_js[int((indices[i+1]+indices[i+2])/2)]).p])
	q_bp.append([np.array(curve_js[indices[i+1]]),np.array(curve_js[int((indices[i+1]+indices[i+2])/2)])])


	breakpoints.append(indices[i])

df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
df.to_csv(cmd_dir+'command0.csv',header=True,index=False)
