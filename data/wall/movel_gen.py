import numpy as np
from pandas import *
import sys, traceback
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robots_def import *
from utils import *
from lambda_calc import *


robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

curve_js = read_csv(data_dir+solution_dir+'Curve_js.csv',header=None).values
curve = read_csv(data_dir+solution_dir+"Curve_in_base_frame.csv",header=None).values

curve_fit=np.zeros((len(curve_js),3))
curve_fit_R=np.zeros((len(curve_js),3,3))
for num_l in num_ls:
	cmd_dir=data_dir+solution_dir+str(num_l)+'L/'

	breakpoints=np.linspace(0,len(curve_js),num_l+1).astype(int)
	points=[]
	points.append([np.array(curve[0,:3])])
	q_bp=[]
	q_bp.append([np.array(curve_js[0])])
	for i in range(1,num_l+1):
		points.append([np.array(curve[breakpoints[i]-1,:3])])
		q_bp.append([np.array(curve_js[breakpoints[i]-1])])

		if i==1:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1],:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1])
			R_init=robot.fwd(curve_js[breakpoints[i-1]]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1])

		else:
			curve_fit[breakpoints[i-1]:breakpoints[i]]=np.linspace(curve[breakpoints[i-1]-1,:3],curve[breakpoints[i]-1,:3],num=breakpoints[i]-breakpoints[i-1]+1)[1:]
			R_init=robot.fwd(curve_js[breakpoints[i-1]-1]).R
			R_end=robot.fwd(curve_js[breakpoints[i]-1]).R
			curve_fit_R[breakpoints[i-1]:breakpoints[i]]=orientation_interp(R_init,R_end,breakpoints[i]-breakpoints[i-1]+1)[1:]
		
	primitives_choices=['moveabsj']+['movel_fit']*num_l

	breakpoints[1:]=breakpoints[1:]-1
	df=DataFrame({'breakpoints':breakpoints,'primitives':primitives_choices,'p_bp':points,'q_bp':q_bp})
	df.to_csv(cmd_dir+'command.csv',header=True,index=False)

	# curve_fit_js=car2js(robot,curve_js[0],curve_fit,curve_fit_R)
	# DataFrame(curve_fit_js).to_csv(cmd_dir+'curve_fit_js.csv',header=False,index=False)
	# DataFrame(curve_fit).to_csv(cmd_dir+'/curve_fit.csv',header=False,index=False)


