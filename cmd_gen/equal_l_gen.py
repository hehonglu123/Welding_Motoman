import numpy as np
from pandas import *
import sys, traceback, glob
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox/')
from robot_def import *


data_dir='../data/wall/'
solution_dir='baseline/'
robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

num_l=1

for file in glob.glob(data_dir+solution_dir+'curve_sliced_js/*.csv'):
	curve_js=np.loadtxt(file,delimiter=',')

	cmd_dir=data_dir+solution_dir+str(num_l)+'L/'

	breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
	points=[]
	points.append([robot.fwd(curve_js[0]).p])
	q_bp=[]
	q_bp.append([np.array(curve_js[0])])
	for i in range(1,num_l+1):
		points.append([robot.fwd(curve_js[breakpoints[i]-1]).p])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

	primitives_choices=['movej']+['movel_fit']*num_l

	breakpoints[1:]=breakpoints[1:]-1

	filename=file.split('\\')[-1]

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command'+filename,header=True,index=False)
