import numpy as np
from pandas import *
import sys, traceback, glob
from general_robotics_toolbox import *
import matplotlib.pyplot as plt

sys.path.append('../toolbox')
from robots_def import *
from lambda_calc import *
from utils import *

class cmd_gen(object):
	###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot,positioner,curve_sliced_js,positioner_js):
		# curve_sliced_js: list of sliced layers, in robot joint space
		# positioner_js: list of sliced layers, in positioner joint space
		# robot: welder robot
		# positioner: 2DOF rotational positioner
		self.robot=robot
		self.positioner=positioner
		self.curve_sliced_js=curve_sliced_js
		self.positioner_js=positioner_js

	def equal_j_gen(self,num_segmets_per_layer=50):
		###generate equally spaced MOVEJ segments
		primitives=[['movej']*num_segmets_per_layer+1]*len(curve_sliced_js)
		breakpoints=[]
		p_bp_robot=[]
		q_bp_robot=[]
		p_bp_positioner=[]
		q_bp_positioner=[]

		for i in range(len(self.curve_sliced_js))
			p_layer_robot=[[self.robot.fwd(self.curve_sliced_js[i][0]).p]]
			q_layer_robot=[[self.curve_sliced_js[i][0]]]
			p_layer_positioner=[[self.positioner.fwd(self.positioner_js[i][0]).p]]
			q_layer_positioner=[[self.positioner_js[i][0]]]
			breakpoints.append(np.linspace(0,len(self.curve_sliced_js[i]),num=num_segmets_per_layer+1).astype(int))
			for j in range(num_segmets_per_layer):
				p_layer_robot.append([self.robot.fwd(self.curve_sliced_js[i][breakpoints[j]+1]).p])
				q_layer_robot.append([self.curve_sliced_js[i][breakpoints[j]+1]])

				p_layer_positioner.append([self.positioner.fwd(self.positioner_js[i][breakpoints[j]+1]).p])
				q_layer_positioner.append([self.positioner_js[i][breakpoints[j]+1]])

			p_bp_robot.append(p_layer_robot)
			q_bp_robot.append(q_layer_robot)

			p_bp_positioner.append(p_layer_positioner)
			q_bp_positioner.append(q_layer_positioner)

		return primitives,primitives,breakpoints,p_bp_robot, q_bp_robot, p_bp_positioner, q_bp_positioner

	def equal_l_gen(self):
		return

def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=50
	cmd_dir=data_dir+'cmd/'+str(num_layers)+'J/'

	robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
		pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')
	positioner=robot_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file=)

	curve_sliced_js=[]
	positioner_js=[]
	for i in range(num_layers):
		curve_sliced.append(np.readtxt(data_dir+'curve_sliced_js/'+str(i)+'.csv',delimiter=','))
		positioner_js.append(np.readtxt(data_dir+'positioner_js/'+str(i)+'.csv',delimiter=','))

	cmd_gen_obj=cmd_gen(robot,positioner,curve_sliced_js,positioner_js)
	primitives_robot,primitives_positioner,breakpoints,p_bp_robot, q_bp_robot, p_bp_positioner, q_bp_positioners=cmd_gen_obj.equal_j_gen(50)
	for i in range(num_layers):
		df=DataFrame({'breakpoints':breakpoints[i],'primitives':primitives_robot[i],'p_bp':p_bp_robot[i],'q_bp':q_bp_robot[i]})
		df.to_csv(cmd_dir+'robot_command'+str(i)+'.csv',header=True,index=False)
		df=DataFrame({'breakpoints':breakpoints[i],'primitives':primitives_positioner[i],'p_bp':p_bp_positioner[i],'q_bp':q_bp_positioner[i]})
		df.to_csv(cmd_dir+'positioner_command'+str(i)+'.csv',header=True,index=False)


if __name__ == '__main__':
	main()