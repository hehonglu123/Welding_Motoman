import numpy as np
import copy
from general_robotics_toolbox import *
from utils import moving_average
###calculate distance between point to line
def get_distance(p1,p2,p3):
	return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

###calculate maximum error of 1 point and the curve in cartesian space, distance only
def calc_error_backup(p,curve):
	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
	order=np.argsort(dist)
	error=get_distance(curve[order[0]],curve[order[1]],p)
	return error
def calc_error(p,curve):
	dist=np.linalg.norm(curve-np.tile(p,(len(curve),1)),axis=1)
	order=np.argsort(dist)
	return dist[order[0]], order[0]
	
###calculate maximum error between fit curve and original curve in cartesian space, distance only
def calc_max_error(fit,curve):
	max_error=0
	idx=0
	max_error_idx=0
	for p in fit:
		error,idx2=calc_error(p,curve)
		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		idx+=1

	return max_error, max_error_idx

def calc_max_error_w_normal(fit,curve,fit_normal,curve_normal,extension=False, eval_mode=False):
	if extension:
		start_idx=np.argmin(np.linalg.norm(curve[0]-fit,axis=1))
		end_idx=np.argmin(np.linalg.norm(curve[-1]-fit,axis=1))
		fit=fit[start_idx:end_idx+1]
		fit_normal=fit_normal[start_idx:end_idx+1]

	max_error=0
	max_error_angle=0
	idx=0
	max_error_idx=0
	error_log = [0] * len(fit)
	normal_error_log = [0] * len(fit_normal)
	for i in range(len(fit)):
		error,idx2=calc_error(fit[i],curve)
		normal_angle=get_angle(fit_normal[i],curve_normal[idx2])

		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		if normal_angle>max_error_angle:
			max_error_angle=copy.deepcopy(normal_angle)
		idx+=1

		if eval_mode:
			error_log[i] = error
			normal_error_log[i] = normal_angle
	if eval_mode:
		return max_error,max_error_angle, max_error_idx, error_log, normal_error_log

	return max_error,max_error_angle, max_error_idx

def calc_max_error_js(robot,fit_js,curve_js):
	fit=[]
	for i in range(len(fit_js)):
		fit.append(robot.fwd(fit_js[i]).p)
	curve=[]
	for i in range(len(curve_js)):
		curve.append(robot.fwd(curve_js[i]).p)
		

	max_error=0
	idx=0
	max_error_idx=0
	for p in fit:
		error=calc_error(p,curve)
		if error>max_error:
			max_error_idx=idx
			max_error=copy.deepcopy(error)
		idx+=1

	return max_error, max_error_idx

def calc_avg_error(fit,curve):
	error=calc_all_error(fit,curve)
	return sum(error)/len(error)

def calc_all_error(fit,curve):
	error=[]
	for p in fit:
		error_temp,idx=calc_error(p,curve)
		error.append(error_temp)
	return error

def calc_all_error_w_normal(fit,curve,fit_normal,curve_normal,extension=False):
	if extension:
		start_idx=np.argmin(np.linalg.norm(curve[0]-fit,axis=1))
		end_idx=np.argmin(np.linalg.norm(curve[-1]-fit,axis=1))
		fit=fit[start_idx:end_idx+1]
		fit_normal=fit_normal[start_idx:end_idx+1]
	error=[]
	angle_error=[]
	for i in range(len(fit)):
		error_temp,idx=calc_error(fit[i],curve)
		normal_angle=get_angle(fit_normal[i],curve_normal[idx])
		error.append(error_temp)
		angle_error.append(normal_angle)
	return np.array(error), np.array(angle_error)

def calc_all_error_ex_blending(fit,curve,zone,lam,breakpoints_lam):
	for breakpoint_lam in breakpoints_lam:
		temp=np.abs(lam-breakpoint_lam)
		idx=np.where(temp<zone)
		lam=np.delete(lam,idx)
		fit=np.delete(fit,idx,axis=0)
	error=[]

	for p in fit:
		error_temp,idx=calc_error(p,curve)
		error.append(error_temp)
	return error,lam


def complete_points_check(fit,curve,R_fit,R_curve):
	error=[]
	
	rotation_error=[]
	for i in range(len(fit)):
		error_temp=np.linalg.norm(curve-fit[i],axis=1)
		idx=np.argmin(error_temp)
		error.append(error_temp[idx])

		R=np.dot(R_fit[i],R_curve[i].T)
		k,theta=R2rot(R)
		rotation_error.append(theta)

	error=np.array(error)
	max_cartesian_error=np.max(error)
	avg_cartesian_error=np.average(error)
	max_cartesian_error_index=np.argmax(error)

	return max_cartesian_error,max_cartesian_error_index,avg_cartesian_error,np.max(np.array(rotation_error))

def complete_points_check2(fit_backproj,curve_backproj,fit,curve):	###error metric on 9/17 by prof Julius
	error_backproj=[]
	
	error=[]
	for i in range(len(fit)):
		error_temp=np.linalg.norm(curve-fit[i],axis=1)
		idx=np.argmin(error_temp)
		error.append(error_temp[idx])

		error_temp=np.linalg.norm(curve_backproj-fit_backproj[i],axis=1)
		idx=np.argmin(error_temp)
		error_backproj.append(error_temp[idx])

	error=np.array(error)
	error_backproj=np.array(error_backproj)
	error_total=error+error_backproj

	max_cartesian_error=np.max(error)
	avg_cartesian_error=np.average(error)
	max_cartesian_error_backproj=np.max(error_backproj)
	avg_cartesian_error_backproj=np.average(error_backproj)
	max_total_error=np.max(error_total)
	avg_total_error=np.average(error_total)
	max_error_index=np.argmax(error_total)

	return max_cartesian_error,avg_cartesian_error,max_cartesian_error_backproj,avg_cartesian_error_backproj,max_total_error,avg_total_error,max_error_index

def logged_data_analysis(robot,timestamp,curve_exe_js):

	act_speed=[]
	lam=[0]
	curve_exe=[]
	curve_exe_R=[]
	for i in range(len(curve_exe_js)):
		robot_pose=robot.fwd(curve_exe_js[i])
		curve_exe.append(robot_pose.p)
		curve_exe_R.append(robot_pose.R)
		if i>0:
			lam.append(lam[-1]+np.linalg.norm(curve_exe[i]-curve_exe[i-1]))
		try:
			act_speed.append(np.linalg.norm(curve_exe[-1]-curve_exe[-2])/(timestamp[i]-timestamp[i-1]))
		except IndexError:
			pass

	# act_speed=moving_average(act_speed,padding=True)

	return lam, np.array(curve_exe), np.array(curve_exe_R), act_speed


def logged_data_analysis_multimove(robot1,robot2,timestamp,curve_exe_js_dual):
	curve_exe_js1=curve_exe_js_dual[:,:6]
	curve_exe_js2=curve_exe_js_dual[:,6:]


	act_speed=[]
	lam=[0]
	relative_path_exe=[]
	relative_path_exe_R=[]
	curve_exe1=[]
	curve_exe2=[]
	curve_exe_R1=[]
	curve_exe_R2=[]
	for i in range(len(curve_exe_js1)):
		pose1_now=robot1.fwd(curve_exe_js1[i])
		pose2_now=robot2.fwd(curve_exe_js2[i])

		curve_exe1.append(pose1_now.p)
		curve_exe2.append(pose2_now.p)
		curve_exe_R1.append(pose1_now.R)
		curve_exe_R2.append(pose2_now.R)

		pose2_world_now=robot2.fwd(curve_exe_js2[i],world=True)

		relative_path_exe.append(np.dot(pose2_world_now.R.T,pose1_now.p-pose2_world_now.p))
		relative_path_exe_R.append(pose2_world_now.R.T@pose1_now.R)
		if i>0:
			lam.append(lam[-1]+np.linalg.norm(relative_path_exe[i]-relative_path_exe[i-1]))
		try:
			if np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2])==0:
				act_speed.append(0)
			else:
				if timestamp[i-1]!=timestamp[i]:
					act_speed.append(np.linalg.norm(relative_path_exe[-1]-relative_path_exe[-2])/(timestamp[i]-timestamp[i-1]))
				else:
					act_speed.append(act_speed[-1])
				
		except IndexError:
			pass

	###speed filter
	act_speed=moving_average(act_speed,padding=True)
	

	return np.array(lam), np.array(curve_exe1),np.array(curve_exe2), np.array(curve_exe_R1),np.array(curve_exe_R2),curve_exe_js1,curve_exe_js2, act_speed, timestamp, np.array(relative_path_exe), np.array(relative_path_exe_R)