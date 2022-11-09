import numpy as np
from general_robotics_toolbox import *
import sys
from robots_def import *
from error_check import *
from toolbox_circular_fit import *
from lambda_calc import *

class MotionSend(object):
	def __init__(self,ip='192.168.55.1') -> None:
		self.ip=ip
		self.ROBOT_CHOICE_MAP={'MA_2010_A0':'RB1','MA_1440_A0':'RB2','D500B':'S1'}

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

	def exe_from_file(self,robot,filename,speed,zone):
		breakpoints,primitives, p_bp,q_bp=self.extract_data_from_cmd(filename)
		return self.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,speed,zone)

	def exec_motions(self,robot,primitives,breakpoints,p_bp,q_bp,speed,zone):
		self.client = MotionProgramExecClient(ip=self.ip,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg)

		self.client.robodk_rob.ACTIVE_TOOL=1
	    self.client.robodk_rob.ProgStart(r"""AAA""")
	    self.client.robodk_rob.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

		for i in range(len(primitives)):
			if 'movel' in primitives[i]:
				###TODO: fix pose
				self.client.robodk_rob.MoveL(Pose(p_bp[i][0]+),q_bp[i][0],speed,zone)
				

			elif 'movec' in primitives[i]:
				self.client.robodk_rob.MoveC(Pose(p_bp[i][0]+),q_bp[i][0],speed,zone)
				self.client.robodk_rob.MoveC(Pose(p_bp[i][1]+),q_bp[i][1],speed,zone)

			elif 'movej' in primitives[i]:
				self.client.robodk_rob.MoveJ(Pose(p_bp[i][0]+),q_bp[i][0],speed,zone)

		self.client.robodk_rob.ProgFinish(r"""AAA""")
	    self.client.robodk_rob.ProgSave(".","AAA",False)

	    self.client.execute_motion_program("AAA.JBI")
	    
		return