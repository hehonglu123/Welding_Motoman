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
		self.ROBOT_CHOICE_MAP={'MA_2010_A0':'RB1','MA_1440_A0':'RB2','D500B':'ST1'}

	def weld_segment(self,robot,q1,q2,speed,cond_num,arc=False):
		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		mp.MoveJ(q1,1,0)
		mp.setArc(arc,cond_num=200)
		mp.MoveL(q2,speed,0)
		mp.setArc(False)
		self.client.execute_motion_program(mp)

	def wire_cut(self,robot,speed=5):
		###cut wire, length given in robot standoff d
		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		p=[463.1378, 1347, -293]
		R=np.array([[ -0.2081, -0.9781, 0],
					[ -0.9781, 0.2081,  0],
					[ 0, 	 	0, 		-1]])

		q_cut1=robot.inv(p+np.array([0,0,50]),R,np.zeros(6))[0]
		q_cut2=robot.inv(p,R,np.zeros(6))[0]
		mp.MoveJ(np.array([-23.88,37.9,40.66,7.42,-72,-20]),speed/2)
		mp.MoveJ(np.zeros(6),speed)
		mp.MoveJ(np.degrees(q_cut1),speed)
		mp.MoveL(np.degrees(q_cut2),50)
		mp.setDO(4095,1)
		mp.setWaitTime(1)
		mp.setDO(4095,0)
		mp.setDOPulse(11,2)
		mp.MoveL(np.degrees(q_cut1),50)
		mp.MoveJ(np.zeros(6),speed)
		mp.MoveJ(np.array([-23.88,37.9,40.66,7.42,-72,-20]),speed)

		self.client.execute_motion_program(mp)

	def touchsense(self,robot,p1,p2,R):
		###p1: list of poitns as starting points
		###p2: list of points as touchsense direction
		###R: tool direction
		q_all=[]
		for i in range(len(p1)):
			mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
			q1=robot.inv(p1[i],R,np.zeros(6))[0]
			mp.MoveJ(np.degrees(q1),4)
			q2=robot.inv(p2[i],R,np.zeros(6))[0]
			mp.touchsense(np.degrees(q2), 30 ,20)
			_,joint_recording,_,_=self.client.execute_motion_program(mp)
			q_all.append(joint_recording[-1][:6])

		return np.array(q_all)

	def extract_data_from_cmd(self,filename):
		data = read_csv(filename)
		breakpoints=np.array(data['breakpoints'].tolist())
		primitives=data['primitives'].tolist()
		points=data['p_bp'].tolist()
		qs=data['q_bp'].tolist()

		p_bp=[]
		q_bp=[]
		for i in range(len(breakpoints)):
			if 'movel' in primitives[i]:
				point=extract_points(primitives[i],points[i])
				p_bp.append([point])
				q=extract_points(primitives[i],qs[i])
				q_bp.append([q])


			elif 'movec' in primitives[i]:
				point1,point2=extract_points(primitives[i],points[i])
				p_bp.append([point1,point2])
				q1,q2=extract_points(primitives[i],qs[i])
				q_bp.append([q1,q2])

			else:
				point=extract_points(primitives[i],points[i])
				p_bp.append([point])
				q=extract_points(primitives[i],qs[i])
				q_bp.append([q])

		return breakpoints,primitives, p_bp,q_bp

	def exec_motion_from_dir(self,robot,directory,arc=False):
		client = MotionProgramExecClient(IP=self.IP,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg)
		
		# num_command=len(fnmatch.filter(os.listdir(directory), '*.csv'))
		num_command=50
		for i in range(num_command):
			breakpoints,primitives, p_bp,q_bp=self.extract_data_from_cmd(directory+'command'+str(i)+'.csv')
			if i<3:
				client=self.form_motion_cmd(client,primitives,q_bp,p_bp,[1,5],0.1,arc)
			else:
				client=self.form_motion_cmd(client,primitives,q_bp,p_bp,[1,50],0.5,arc)

		# num_layers=20
		# breakpoints,primitives, p_bp,q_bp=self.extract_data_from_cmd(directory+'command0.csv')
		# client=self.form_motion_cmd(client,primitives[:num_layers],q_bp[:num_layers],p_bp[:num_layers],[1,5,10,15]+[50]*20,1,arc)

		client.ProgEnd()
		client.execute_motion_program("AAA.JBI")

	def form_motion_cmd(self,client,primitives,q_bp,p_bp,speed,zone,arc=False):
		for i in range(len(primitives)):
			if 'movel' in primitives[i]:
				###TODO: fix pose
				if type(speed) is list:
					if type(zone) is list:
						client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveL(np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveL(np.degrees(q_bp[i][0]),speed,zone)

			elif 'movec' in primitives[i]:		###moveC needs testing
				if type(speed) is list:
					if type(zone) is list:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone[i])
					else:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone[i])
					else:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone)

				

			elif 'movej' in primitives[i]:
				if type(speed) is list:
					if type(zone) is list:
						client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveJ(np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveJ(np.degrees(q_bp[i][0]),speed,zone)
				if arc==True and i==0:
					client.SetArc(True)
		if arc:
			client.SetArc(False)

		return client

	def form_motion_cmd_multimove(self,client,primitives_robot,p_bp_robot,q_bp_robot,primitives_positioner,p_bp_positioner,q_bp_positioner,speed,zone,arc=False):
		for i in range(len(primitives)):
			if 'movel' in primitives[i]:
				###TODO: fix pose
				if type(speed) is list:
					if type(zone) is list:
						client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveL(np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveL(np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveL(np.degrees(q_bp[i][0]),speed,zone)

			elif 'movec' in primitives[i]:		###moveC needs testing
				if type(speed) is list:
					if type(zone) is list:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone[i])
					else:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone[i])
					else:
						client.MoveC(np.degrees(q_bp[i-1][-1]),np.degrees(q_bp[i][0]),np.degrees(q_bp[i][1]),speed,zone)

				

			elif 'movej' in primitives[i]:
				if type(speed) is list:
					if type(zone) is list:
						client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveJ(np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveJ(np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveJ(np.degrees(q_bp[i][0]),speed,zone)
				if arc==True and i==0:
					client.SetArc(True)
		if arc:
			client.SetArc(False)

		return client

	def exec_motions(self,robot,primitives,p_bp,q_bp,speed,zone):
		client = MotionProgramExecClient(IP=self.IP,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg)

		client=self.form_motion_cmd(client,primitives,q_bp,p_bp,speed,zone)


		client.ProgEnd()
		client.execute_motion_program("AAA.JBI")
		

		return

	def exec_motions_multimove(self,robot,positioner,primitives_robot,p_bp_robot,q_bp_robot,primitives_positioner,p_bp_positioner,q_bp_positioner,speed,zone):
		client = MotionProgramExecClient(IP=self.IP,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],ROBOT_CHOICE2=self.ROBOT_CHOICE_MAP[positioner.robot_name],pulse2deg=robot.pulse2deg,pulse2deg2=positioner.pulse2deg)

		client=self.form_motion_cmd_multimove(client,primitives,primitives_robot,p_bp_robot,q_bp_robot,primitives_positioner,p_bp_positioner,q_bp_positioner,speed,zone)


		client.ProgEnd()
		client.execute_motion_program("AAA.JBI")
		

		return