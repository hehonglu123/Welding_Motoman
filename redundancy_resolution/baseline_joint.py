import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *


def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice2/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=5
	curve_sliced_relative=[]
	curve_sliced=[]
	for i in range(num_layers):
		curve_sliced_relative.append(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'.csv',delimiter=','))
		curve_sliced.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'.csv',delimiter=','))

	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')


	R_torch=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,curve_sliced)
	H=np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')

	positioner_js,curve_sliced_js=rr.baseline_joint(R_torch,curve_sliced_relative)

	for i in range(num_layers):
		np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'.csv',positioner_js[i],delimiter=',')
		np.savetxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'.csv',curve_sliced_js[i],delimiter=',')



	# positioner_js,curve_sliced_js=rr.baseline(R_torch,q_seed)

if __name__ == '__main__':
	main()