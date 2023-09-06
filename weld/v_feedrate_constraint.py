import numpy as np
import traceback, time, sys
from RobotRaconteur.Client import *
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
import matplotlib.pyplot as plt
from WeldSend import *


##########KINEMATICS 
robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])
x_all_exp=[1645,1660,1675,1690]
v_all_exp=np.linspace(5,15,len(x_all_exp))
feedrate=70
base_layer_height=3
layer_height=1.5

client=MotionProgramExecClient()
ws=WeldSend(client)


for m in range(len(x_all_exp)):
	p_start=np.array([x_all_exp[m],-860,-260])
	p_end=np.array([x_all_exp[m],-760,-260])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
	
	
	q_all=[]
	v_all=[]
	primitives=[]
	cond_all=[]
	
	for i in range(2):
		if i%2==0:
			p1=p_start+np.array([0,0,i*base_layer_height])
			p2=p_end+np.array([0,0,i*base_layer_height])
		else:
			p1=p_end+np.array([0,0,i*base_layer_height])
			p2=p_start+np.array([0,0,i*base_layer_height])

	
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]

		q_all.extend([q_init,q_end])
		v_all.extend([1,5])
		primitives.extend(['movej','movel'])
		cond_all.extend([0,int(250/10+200)])

	for i in range(2,10):
		if i%2==0:
			p1=p_start+np.array([0,0,2*base_layer_height+(i-2)*layer_height])
			p2=p_end+np.array([0,0,2*base_layer_height+(i-2)*layer_height])
		else:
			p1=p_end+np.array([0,0,2*base_layer_height+(i-2)*layer_height])
			p2=p_start+np.array([0,0,2*base_layer_height+(i-2)*layer_height])

	
		q_init=robot.inv(p1,R,q_seed)[0]
		q_end=robot.inv(p2,R,q_seed)[0]

		q_all.extend([q_end])
		v_all.extend([v_all_exp[m]])
		primitives.extend(['movel'])
		cond_all.extend([int(feedrate/10+200)])

	ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.)
