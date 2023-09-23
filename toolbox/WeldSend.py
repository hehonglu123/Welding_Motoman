import numpy as np
from general_robotics_toolbox import *
import sys, glob, fnmatch
from robot_def import *
from pandas import read_csv,DataFrame
from dx200_motion_program_exec_client import *

class WeldSend(object):
	def __init__(self,client) -> None:
		self.client=client
		self.ROBOT_CHOICE_MAP={'MA2010_A0':'RB1','MA1440_A0':'RB2','D500B':'ST1'}
		self.ROBOT_TOOL_MAP={'MA2010_A0':12,'MA1440_A0':2,'D500B':0}

	def jog_single(self,robot,q,v=1):
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot.robot_name])
		
		if len(np.array(q).shape)==1:
			mp.MoveJ(np.degrees(q),v,0)
		else:
			for q_wp in q:
				mp.MoveJ(np.degrees(q_wp),v,0)

		self.client.execute_motion_program(mp)
	
	def jog_dual(self,robot1,robot2,q1,q2,v=1):
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=robot2.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		
		if len(np.array(q1).shape)==1 and len(np.array(q2).shape)==1:
			mp.MoveJ(np.degrees(q1),v,0,target2=['MOVJ',np.degrees(q2),10])
		elif len(np.array(q1).shape)==2 and len(np.array(q2).shape)==1:
			for q1_wp in q1:
				mp.MoveJ(np.degrees(q1_wp),v,0,target2=['MOVJ',np.degrees(q2),10])
		elif len(np.array(q1).shape)==1 and len(np.array(q2).shape)==2:
			for q2_wp in q2:
				mp.MoveJ(np.degrees(q1),v,0,target2=['MOVJ',np.degrees(q2_wp),10])
		else:
			wp_length = min(len(q1),len(q2))
			for wp_i in range(wp_length):
				mp.MoveJ(np.degrees(q1[wp_i]),v,0,target2=['MOVJ',np.degrees(q2[wp_i]),10])

		self.client.execute_motion_program(mp)
	
	def jog_tri(self,robot1,positioner,robot2,q1,q_positioner,q2,v=1):
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[positioner.robot_name],ROBOT_CHOICE3=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=positioner.pulse2deg,pulse2deg_3=robot2.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		mp.MoveJ(np.degrees(q1),v,None,target2=['MOVJ',np.degrees(q_positioner),None],target3=['MOVJ',np.degrees(q2),None])
		self.client.execute_motion_program(mp)


	def weld_segment_single(self,primitives,robot,q_all,v_all,cond_all,arc=False,wait=0):
		###single arm weld segment 
		#q_all: list of joint angles (N x 6)
		#v_all: list of segment speed (N x 1)
		#cond_all: list of job number (N x 1) or (1,), 0 refers to off
		#wait: wait time after motion, before arcof
		q_all=np.degrees(q_all)
		arcof=True
		
		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot.robot_name])
		mp.primitive_call(primitives[0],q_all[0],v_all[0])
		if arc:
			if len(cond_all)==1: 
				mp.setArc(True,cond_num=cond_all[0])
				arcof=False
			else:
				if cond_all[1]!=0:
					mp.setArc(True,cond_num=cond_all[1])
					arcof=False


		for i in range(1,len(q_all)):
			if len(cond_all)>1 and arc and i>1:
				if arcof:
					if cond_all[i]!=0:
						mp.setArc(True, cond_all[i])
						arcof=False
				else:
					if cond_all[i]==0:
						mp.setArc(False)
						arcof=True
					elif cond_all[i]!=cond_all[i-1]:
						mp.changeArc(cond_all[i])

			mp.primitive_call(primitives[i],q_all[i],v_all[i])
		
		if wait:
			mp.setWaitTime(wait)
			
		if arc and not arcof:
			mp.setArc(False)
		return self.client.execute_motion_program(mp)

	def weld_segment_dual(self,primitives,robot1,robot2,q1_all,q2_all,v1_all,v2_all,cond_all,arc=False):
		###robot+positioner weld segment, MOVEJ + MOVEL x (N-1)
		#q1_all: list of robot joint angles (N x 6)
		#q2_all: list of positioenr joint angles (N x 2)
		#v1_all: list of 1segment speed (N x 1)
		#v2_all: list of 2segment speed (N x 1)
		#cond_all: list of job number (N x 1) or (1,), 0 refers to off

		q1_all=np.degrees(q1_all)
		q2_all=np.degrees(q2_all)
		arcof=True

		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=robot2.pulse2deg, tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		mp.primitive_call_dual(primitives[0],q1_all[0],v1_all[0],target2=['MOVJ',q2_all[0],v2_all[0]])
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
					elif cond_all[i]!=cond_all[i-1]:
						mp.changeArc(cond_all[i])

			mp.primitive_call_dual(primitives[i],q1_all[i],v1_all[i],target2=['MOVJ',q2_all[i],v2_all[i]])
			
		if arc and not arcof:
			mp.setArc(False)
		return self.client.execute_motion_program(mp)

	def weld_segment_tri(self,primitives,robot1,positioner,robot2,q1_all,positioner_all,q2_all,v1_all,v2_all,cond_all,arc=False):
		###robot+positioner weld segment, MOVEJ + MOVEL x (N-1)
		#q1_all: list of robot joint angles (N x 6)
		#positioner_all: list of positioner joint angles (N x 2)
		#q2_all: list of positioenr joint angles (N x 6)
		#v1_all: list of robot1segment speed (N x 1)
		#v2_all: list of robot2segment speed (N x 1)
		#cond_all: list of job number (N x 1) or (1,), 0 refers to off

		q1_all=np.degrees(q1_all)
		positioner_all=np.degrees(positioner_all)
		q2_all=np.degrees(q2_all)
		arcof=True

		mp=MotionProgram(ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot1.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[positioner.robot_name],
		   ROBOT_CHOICE3=self.ROBOT_CHOICE_MAP[robot2.robot_name],pulse2deg=robot1.pulse2deg,pulse2deg_2=positioner.pulse2deg,pulse2deg_3=robot2.pulse2deg, 
		   tool_num=self.ROBOT_TOOL_MAP[robot1.robot_name])
		
		mp.primitive_call_tri(primitives[0],q1_all[0],v1_all[0],target2=['MOVJ',positioner_all[0],None],target3=['MOVJ',q2_all[0],v2_all[0]])
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
					elif cond_all[i]!=cond_all[i-1]:
						mp.changeArc(cond_all[i])

			mp.primitive_call_tri(primitives[i],q1_all[i],v1_all[i],target2=['MOVJ',positioner_all 	[i],None],target3=['MOVL',q2_all[i],v2_all[i]])
			
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

	def save_weld_cmd(self,filename,breakpoints,primitives,q_bp,weld_v):

		q_bp_new=[]
		weld_v_new=[]
		for i in range(len(primitives)):
			if len(q_bp[i])==2:
				q_bp_new.append([np.array(q_bp[i][0]),np.array(q_bp[i][1])])
				weld_v_new.append([weld_v[i],weld_v[i]])
			else:
				q_bp_new.append([np.array(q_bp[i][0])])
				weld_v_new.append([weld_v[i]])
		df=DataFrame({'breakpoints':breakpoints,'primitives':primitives, 'q_bp':q_bp_new, 'weld_v':weld_v_new})
		df.to_csv(filename,header=True,index=False)
	
	def load_weld_cmd(self,filename):
		
		data = read_csv(filename)
		breakpoints=np.array(data['breakpoints'].tolist()).astype(int)
		primitives=data['primitives'].tolist()
		qs=data['q_bp'].tolist()
		weld_v_str=np.array(data['weld_v'].tolist())
		q_bp=[]
		for q in qs:
			endpoint=q[8:-3].split(',')
			qarr = np.array(list(map(float, endpoint)))
			q_bp.append([np.array(qarr)])
		weld_v=[]
		for v in weld_v_str:
			weld_v.append(float(v[1:-1]))

		return breakpoints,primitives,q_bp,weld_v