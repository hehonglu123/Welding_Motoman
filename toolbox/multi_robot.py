import numpy as np
import copy
from general_robotics_toolbox import *
# from utils import *
# from lambda_calc import *
from pandas import read_csv

def form_relative_path_mocap(curve_exe1,curve_exe1_R,curve_exe2,curve_exe2_R,robot1,robot2):
	relative_path_exe=[]
	relative_path_exe_R=[]
	for i in range(len(curve_exe1)):

		curve_exe2_world_now=robot2.base_H[:3,:3]@curve_exe2[i]+robot2.base_H[:3,-1]
		curve_exe2_R_world_now=robot2.base_H[:3,:3]@curve_exe2_R[i]

		relative_path_exe.append(curve_exe2_R_world_now.T@(curve_exe1[i]-curve_exe2_world_now))
		relative_path_exe_R.append(curve_exe2_R_world_now.T@curve_exe1_R[i])

	return np.array(relative_path_exe),np.array(relative_path_exe_R)

def form_relative_path(curve_js1,curve_js2,robot1,robot2):
	relative_path_exe=[]
	relative_path_exe_R=[]
	curve_exe1=[]
	curve_exe2=[]
	curve_exe_R1=[]
	curve_exe_R2=[]
	for i in range(len(curve_js1)):
		pose1_now=robot1.fwd(curve_js1[i])
		pose2_now=robot2.fwd(curve_js2[i])

		curve_exe1.append(pose1_now.p)
		curve_exe2.append(pose2_now.p)
		curve_exe_R1.append(pose1_now.R)
		curve_exe_R2.append(pose2_now.R)

		pose2_world_now=robot2.fwd(curve_js2[i],world=True)

		relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
		relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)
	return np.array(curve_exe1),np.array(curve_exe2), np.array(curve_exe_R1),np.array(curve_exe_R2),np.array(relative_path_exe),np.array(relative_path_exe_R)

def calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints):
	speed_ratio=[]      ###speed of robot1 TCP / robot2 TCP
	dt=[]
	for i in range(1,len(breakpoints)):
		speed_ratio.append((lam1[breakpoints[i]-1]-lam1[breakpoints[i-1]-1])/(lam2[breakpoints[i]-1]-lam2[breakpoints[i-1]-1]))
		dt.append((lam_relative[breakpoints[i]]-lam_relative[breakpoints[i-1]])/vd_relative)
	###specify speed here for robot2
	s2_all=[]
	s1_all=[]
	for i in range(len(breakpoints)-1):
		s1_all.append((lam1[breakpoints[i+1]]-lam1[breakpoints[i]])/dt[i])
		s2=s1_all[i]/speed_ratio[i]
		s2_all.append(s2)

	return s1_all,s2_all

def initialize_data(dataset,data_dir,solution_dir,robot1,robot2):
	relative_path = read_csv(data_dir+"Curve_dense.csv", header=None).values

	###calculate desired TCP speed for both arm
	curve_js1=read_csv(solution_dir+'arm1.csv', header=None).values
	curve_js2=read_csv(solution_dir+'arm2.csv', header=None).values

	

	lam_relative_path=calc_lam_cs(relative_path)


	lam1=calc_lam_js(curve_js1,robot1)
	lam2=calc_lam_js(curve_js2,robot2)


	return relative_path,lam_relative_path,lam1,lam2,curve_js1,curve_js2


def cmd_speed_profile(breakpoints,s1_all,s2_all):
	###return commanded speed profile, with same lenght as lambda
	s1_cmd=[]
	s2_cmd=[]

	for i in range(len(breakpoints)):
		s1_cmd.extend([s1_all[i]]*(breakpoints[i]-breakpoints[i-1]))
		s2_cmd.extend([s2_all[i]]*(breakpoints[i]-breakpoints[i-1]))
	return np.array(s1_cmd).flatten(),np.array(s2_cmd).flatten()

