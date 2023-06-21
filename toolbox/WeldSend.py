import numpy as np
from general_robotics_toolbox import *
import sys, glob, fnmatch
from robot_def import *
from pandas import read_csv
from dx200_motion_program_exec_client import *

# from lambda_calc import *

class WeldSend(object):
	def __init__(self,client) -> None:
		self.client=client
		self.ROBOT_CHOICE_MAP={'MA2010_A0':'RB1','MA1440_A0':'RB2','D500B':'ST1'}
		self.ROBOT_TOOL_MAP={'MA2010_A0':12,'MA1440_A0':2}

	def jog_single(self,robot,q,v=1):
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot.robot_name])
		mp.MoveJ(np.degrees(q),v,None)
		self.client.execute_motion_program(mp)
	
	def jog_dual(self,robot1,robot2,q1,q2,v=1):
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=robot2.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		mp.MoveJ(np.degrees(q1),v,None,target2=['MOVJ',np.degrees(q2),1,0])
		self.client.execute_motion_program(mp)

	def weld_segment_single(self,robot,q_all,v_all,cond_all,arc=False):
		###single arm weld segment 
		#q_all: list of joint angles (N x 6)
		#v_all: list of segment speed (N-1 x 1)
		#cond_all: list of job number (N-1 x 1) or (1,), 0 refers to off
		q_all=np.degrees(q_all)
		arcof=True
		
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot.robot_name])
		mp.MoveJ(q_all[0],1,None)
		if arc:
			mp.setArc(True,cond_num=cond_all[0])
			arcof=False
		for i in range(1,len(q_all)):
			mp.MoveL(q_all[i],v_all[i-1],None)
			if len(cond_all)>1 and arc:
				if arcof:
					if cond_all[i-1]!=0:
						mp.setArc(True, cond_all[i-1])
						arcof=False
				else:
					if cond_all[i-1]==0:
						mp.setArc(False)
					else:
						mp.changeArc(cond_all[i-1])
		if arc:
			mp.setArc(False)
		self.client.execute_motion_program(mp)

	def weld_segment_dual(self,robot1,robot2,q1_all,q2_all,v1_all,v2_all,cond_all,arc=False):
		print(len(q1_all),len(q2_all),len(v1_all),len(v2_all))
		###robot+positioner weld segment 
		#q1_all: list of robot joint angles (N x 6)
		#q2_all: list of positioenr joint angles (N x 2)
		#v1_all: list of 1segment speed (N x 1)
		#v2_all: list of 2segment speed (N x 1)
		#cond_all: list of job number (N x 1) or (1,), 0 refers to off

		q1_all=np.degrees(q1_all)
		q2_all=np.degrees(q2_all)
		arcof=True

		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=robot2.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		mp.MoveJ(q1_all[0], v1_all[0],None,target2=['MOVJ',q2_all[0],1,0])
		if arc:
			if len(cond_all)==1: 
				mp.setArc(True,cond_num=cond_all[0])
				arcof=False
			else:
				if cond_all[1]!=0:
					mp.setArc(True,cond_num=cond_all[1])
					arcof=False


		for i in range(1,len(q1_all)):
			if len(cond_all)>1 and arc and i>1:
				if arcof:
					if cond_all[i]!=0:
						mp.setArc(True, cond_all[i])
						arcof=False
				else:
					if cond_all[i]==0:
						mp.setArc(False)
						arcof=True
					else:
						mp.changeArc(cond_all[i])

			mp.MoveL(q1_all[i],v1_all[i-1],None,target2=['MOVJ',q2_all[i],v2_all[i-1],0])
			
		if arc and not arcof:
			mp.setArc(False)
		return self.client.execute_motion_program(mp)


	def wire_cut(self,robot,speed=5,q_safe=np.radians([-23.88,37.9,40.66,7.42,-72,-20])):
		###cut wire, length given in robot standoff d
		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		p=[463.1378, 1347, -293]
		R=np.array([[ -0.2081, -0.9781, 0],
					[ -0.9781, 0.2081,  0],
					[ 0, 	 	0, 		-1]])

		q_cut1=robot.inv(p+np.array([0,0,50]),R,np.zeros(6))[0]
		q_cut2=robot.inv(p,R,np.zeros(6))[0]
		mp.MoveJ(np.degrees(q_safe),speed/2)
		mp.MoveJ(np.zeros(6),speed)
		mp.MoveJ(np.degrees(q_cut1),speed)
		mp.MoveL(np.degrees(q_cut2),50)
		mp.setDO(4095,1)
		mp.setWaitTime(1)
		mp.setDO(4095,0)
		mp.setDOPulse(11,2)
		mp.MoveL(np.degrees(q_cut1),50)
		mp.MoveJ(np.zeros(6),speed)
		mp.MoveJ(np.degrees(q_safe),speed)

		self.client.execute_motion_program(mp)

	def touchsense(self,robot,p1,p2,R,q_safe=np.radians([-23.88,37.9,40.66,7.42,-72,-20])):
		###p1: list of poitns as starting points
		###p2: list of points as touchsense direction
		###R: tool direction
		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		mp.MoveJ(np.degrees(q_safe),2)
		self.client.execute_motion_program(mp)
		q_all=[]
		for i in range(len(p1)):
			mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
			q1=robot.inv(p1[i],R,np.zeros(6))[0]
			mp.MoveJ(np.degrees(q1),4)
			q2=robot.inv(p2[i],R,np.zeros(6))[0]
			mp.touchsense(np.degrees(q2), 10 ,20)
			_,joint_recording,_,_=self.client.execute_motion_program(mp)
			q_all.append(joint_recording[-1][:6])

		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		mp.MoveJ(np.degrees(q_safe),2)
		self.client.execute_motion_program(mp)

		return np.array(q_all)


	def logged_data_analysis_mocap(self,robot,curve_exe_dict,curve_exe_R_dict,timestamp_dict):
		curve_exe = np.array(curve_exe_dict[robot.robot_name])
		curve_exe_R = np.array(curve_exe_R_dict[robot.robot_name])
		timestamp = np.array(timestamp_dict[robot.robot_name])
		len_min=min(len(timestamp),len(curve_exe),len(curve_exe_R))
		curve_exe=curve_exe[:len_min]
		timestamp=timestamp[:len_min]
		curve_exe_R=curve_exe_R[:len_min]

		curve_exe_w=smooth_w(R2w(curve_exe_R,np.eye(3)))
		###filter noise
		timestamp, curve_exe_pw=lfilter(timestamp, np.hstack((curve_exe,curve_exe_w)))


		return  curve_exe_pw[:,:3], curve_exe_pw[:,3:], timestamp


	# def extract_data_from_cmd(self,filename):
	# 	data = read_csv(filename)
	# 	breakpoints=np.array(data['breakpoints'].tolist())
	# 	primitives=data['primitives'].tolist()
	# 	points=data['p_bp'].tolist()
	# 	qs=data['q_bp'].tolist()

	# 	p_bp=[]
	# 	q_bp=[]
	# 	for i in range(len(breakpoints)):
	# 		if 'movel' in primitives[i]:
	# 			point=extract_points(primitives[i],points[i])
	# 			p_bp.append([point])
	# 			q=extract_points(primitives[i],qs[i])
	# 			q_bp.append([q])


	# 		elif 'movec' in primitives[i]:
	# 			point1,point2=extract_points(primitives[i],points[i])
	# 			p_bp.append([point1,point2])
	# 			q1,q2=extract_points(primitives[i],qs[i])
	# 			q_bp.append([q1,q2])

	# 		else:
	# 			point=extract_points(primitives[i],points[i])
	# 			p_bp.append([point])
	# 			q=extract_points(primitives[i],qs[i])
	# 			q_bp.append([q])

	# 	return breakpoints,primitives, p_bp,q_bp



	# def form_motion_cmd(self,client,primitives,q_bp,p_bp,speed,zone,arc=False):
	# 	for i in range(len(primitives)):
	# 		if 'movel' in primitives[i]:
	# 			###TODO: fix pose
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone[i])
	# 				else:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed,zone[i])
	# 				else:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed,zone)

	# 		elif 'movec' in primitives[i]:		###moveC needs testing
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone[i])
	# 				else:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone[i])
	# 				else:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone)

				

	# 		elif 'movej' in primitives[i]:
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone[i])
	# 				else:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed,zone[i])
	# 				else:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed,zone)
	# 			if arc==True and i==0:
	# 				client.SetArc(True)
	# 	if arc:
	# 		client.SetArc(False)

	# 	return client

	# def form_motion_cmd_multimove(self,client,primitives_robot,p_bp_robot,q_bp_robot,primitives_positioner,p_bp_positioner,q_bp_positioner,speed,zone,arc=False):
	# 	for i in range(len(primitives)):
	# 		if 'movel' in primitives[i]:
	# 			###TODO: fix pose
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone[i])
	# 				else:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed,zone[i])
	# 				else:
	# 					client.MoveL(np.degrees(q_bp[i][0]),speed,zone)

	# 		elif 'movec' in primitives[i]:		###moveC needs testing
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone[i])
	# 				else:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone[i])
	# 				else:
	# 					client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone)

				

	# 		elif 'movej' in primitives[i]:
	# 			if type(speed) is list:
	# 				if type(zone) is list:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone[i])
	# 				else:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone)
	# 			else:
	# 				if type(zone) is list:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed,zone[i])
	# 				else:
	# 					client.MoveJ(np.degrees(q_bp[i][0]),speed,zone)
	# 			if arc==True and i==0:
	# 				client.SetArc(True)
	# 	if arc:
	# 		client.SetArc(False)

	# 	return client

