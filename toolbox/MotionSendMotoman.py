import numpy as np
from general_robotics_toolbox import *
import sys, glob, fnmatch
from robot_def import *
from pandas import read_csv
from dx200_motion_program_exec_client import *

# from error_check import *
# from toolbox_circular_fit import *
# from lambda_calc import *

class MotionSend(object):
	def __init__(self,IP='192.168.1.31') -> None:
		self.IP=IP
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

	def exec_motion_from_dir(self,robot,directory,arc=False):
		client = MotionProgramExecClient(IP=self.IP,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg)
		client.ACTIVE_TOOL=1
		client.ProgStart(r"""AAA""")
		client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

		# num_command=len(fnmatch.filter(os.listdir(directory), '*.csv'))
		num_command=10
		for i in range(num_command):
			breakpoints,primitives, p_bp,q_bp=self.extract_data_from_cmd(directory+'command'+str(i)+'.csv')
			client=self.form_motion_cmd(client,primitives,q_bp,p_bp,[1,20],0,arc)

		client.ProgFinish(r"""AAA""")
		client.ProgSave(".","AAA",False)

		client.execute_motion_program("AAA.JBI")

	def form_motion_cmd(self,client,primitives,q_bp,p_bp,speed,zone,arc=False):
		for i in range(len(primitives)):
			if 'movel' in primitives[i]:
				###TODO: fix pose
				if type(speed) is list:
					if type(zone) is list:
						client.MoveL(None,np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveL(None,np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveL(None,np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveL(None,np.degrees(q_bp[i][0]),speed,zone)

			elif 'movec' in primitives[i]:		###moveC needs testing
				client.MoveC(None,np.degrees(q_bp[i][0]),speed,zone)
				client.MoveC(None,np.degrees(q_bp[i][1]),speed,zone)

			elif 'movej' in primitives[i]:
				if type(speed) is list:
					if type(zone) is list:
						client.MoveJ(None,np.degrees(q_bp[i][0]),speed[i],zone[i])
					else:
						client.MoveJ(None,np.degrees(q_bp[i][0]),speed[i],zone)
				else:
					if type(zone) is list:
						client.MoveJ(None,np.degrees(q_bp[i][0]),speed,zone[i])
					else:
						client.MoveJ(None,np.degrees(q_bp[i][0]),speed,zone)
				if arc==True and i==0:
					client.SetArc(True)
		if arc:
			client.SetArc(False)

		return client

	def exec_motions(self,robot,primitives,breakpoints,p_bp,q_bp,speed,zone):
		client = MotionProgramExecClient(IP=self.IP,ROBOT_CHOICE=self.ROBOT_CHOICE_MAP[robot.robot_name],pulse2deg=robot.pulse2deg)
		client.ACTIVE_TOOL=1
		client.ProgStart(r"""AAA""")
		client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
		client=self.form_motion_cmd(client,primitives,q_bp,p_bp,speed,zone)


		client.ProgFinish(r"""AAA""")
		client.ProgSave(".","AAA",False)

		client.execute_motion_program("AAA.JBI")
		

		return