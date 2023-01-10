import numpy as np
from pandas import *
import sys, traceback, glob, fnmatch, os
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox/')
from robot_def import *


data_dir='../data/cylinder/'
solution_dir='baseline/'
robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')


primitives_choices=['movej']
cmd_dir=data_dir+solution_dir+'spiral/'
points=[]
breakpoints=[0]
q_bp=[]

num_layers=len(fnmatch.filter(os.listdir(data_dir+solution_dir+'curve_sliced_js/'), '*.csv'))

for i in range(num_layers):
	curve_js=np.loadtxt(data_dir+solution_dir+'curve_sliced_js/'+str(i)+'.csv',delimiter=',')
	
	if i==0:
		q_bp.append([np.array(curve_js[0])])
		points.append([robot.fwd(curve_js[0]).p])


	primitives_choices.append('movec_fit')

	mid_idx=int((len(curve_js))/2)
	points.append([robot.fwd(curve_js[mid_idx]).p,robot.fwd(curve_js[-1]).p])
	q_bp.append([np.array(curve_js[mid_idx]),np.array(curve_js[-1])])


	breakpoints.append(breakpoints[-1]+len(curve_js))

df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
df.to_csv(cmd_dir+'command0.csv',header=True,index=False)
