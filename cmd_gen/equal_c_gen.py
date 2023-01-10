import numpy as np
from pandas import *
import sys, traceback, glob
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox/')
from robot_def import *


data_dir='../data/cylinder/'
solution_dir='baseline/'
robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

num_l=1

reverse=True

count=0
for file in glob.glob(data_dir+solution_dir+'curve_sliced_js/*.csv'):
	curve_js=np.loadtxt(file,delimiter=',')

	
	if reverse:
		if count%2==1:
			curve_js=np.flip(curve_js,axis=0)

		breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
		points=[]
		points.append([robot.fwd(curve_js[0]).p])
		q_bp=[]
		q_bp.append([np.array(curve_js[0])])
		for i in range(1,num_l+1):
			mid_idx=int((breakpoints[i]-breakpoints[i-1])/2)
			points.append([robot.fwd(curve_js[mid_idx]).p,robot.fwd(curve_js[breakpoints[i]-1]).p])
			q_bp.append([np.array(curve_js[mid_idx]),np.array(curve_js[breakpoints[i]-1])])

		primitives_choices=['movej']+['movec_fit']*num_l

		breakpoints[1:]=breakpoints[1:]-1

		filename=file.split('\\')[-1]


		cmd_dir=data_dir+solution_dir+str(num_l)+'C_reverse/'

	else:
		breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
		points=[]
		points.append([robot.fwd(curve_js[0]).p])
		q_bp=[]
		q_bp.append([np.array(curve_js[0])])
		for i in range(1,num_l+1):
			mid_idx=int((breakpoints[i]-breakpoints[i-1])/2)
			points.append([robot.fwd(curve_js[mid_idx]).p,robot.fwd(curve_js[breakpoints[i]-1]).p])
			q_bp.append([np.array(curve_js[mid_idx]),np.array(curve_js[breakpoints[i]-1])])

		primitives_choices=['movej']+['movec_fit']*num_l

		breakpoints[1:]=breakpoints[1:]-1

		filename=file.split('\\')[-1]


		cmd_dir=data_dir+solution_dir+str(num_l)+'C/'

	count+=1

	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command'+filename,header=True,index=False)
