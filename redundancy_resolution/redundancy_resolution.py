import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robots_def import *
from lambda_calc import *
from utils import *

class redundancy_resolution(object):
	###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot,positioner,curve_sliced):
		# curve_sliced: list of sliced layers, in curve frame
		# robot: welder robot
		# positioner: 2DOF rotational positioner
		self.robot=robot
		self.positioner=positioner
		self.curve_sliced=curve_sliced

	

	def baseline(self,R_torch,curve_sliced_relative,q_init=np.zeros(6)):
		####baseline redundancy resolution, with fixed orientation
		positioner_js=self.positioner_resolution()		#solve for positioner first
	
		curve_sliced_js=np.zeros((len(curve_sliced_relative),len(curve_sliced_relative[0]),6))
		for i in range(len(curve_sliced_relative)):			#solve for robot invkin
			for j in range(len(curve_sliced_relative[i])): 
				###get positioner TCP world pose
				positioner_pose=self.positioner.fwd(positioner_js[i][j],world=True)
				p=positioner_pose.R@curve_sliced_relative[i,j,:3]+positioner_pose
				###solve for invkin
				if i==0 and j==0:
					q=self.robot.inv(p,R_torch,last_joints=q_init)
				else:
					q=self.robot.inv(p,R_torch,last_joints=q_prev)
					q_prev=q

		return positioner_js,curve_sliced_js



	def positioner_resolution(self):
		###resolve 2DOF positioner joint angle 
		positioner_js=np.zeros((len(curve_sliced_relative),len(curve_sliced_relative[0]),2))
		for i in range(len(curve_sliced_relative)):
			for j in range(len(curve_sliced_relative[i])):
				positioner_js[i][j]=self.positioner.inv(curve_sliced_relative[i,j,:3],curve_sliced_relative[i,j,3:])
		return positioner_js


def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=50
	curve_sliced_relative=[]
	for i in range(num_layers):
		curve_sliced_relative.append(np.readtxt(data_dir+'curve_sliced_relative/'+str(i)+'.csv',delimiter=','))

	robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
		pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file=)

	R_torch=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,np.array(curve_sliced_relative))
	positioner_js,curve_sliced_js=rr.baseline(R_torch,q_seed)

if __name__ == '__main__':
	main()