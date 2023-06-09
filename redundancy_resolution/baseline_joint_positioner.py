import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *


def main():
	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')


	dataset='cup/'
	sliced_alg='circular_slice_shifted/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=507
	num_baselayers=0
	q_positioner_seed=[1,3]
	# dataset='blade0.1/'
	# sliced_alg='auto_slice/'
	# data_dir='../data/'+dataset+sliced_alg
	# num_layers=757
	# num_baselayers=2
	# q_positioner_seed=[0,-2]

	curve_sliced_relative_base=[]
	curve_sliced_relative=[]
	curve_sliced=[]
	for i in range(num_baselayers):
		num_sections=len(glob.glob(data_dir+'curve_sliced_relative/baselayer'+str(i)+'_*.csv'))
		curve_sliced_relative_base_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_relative_base_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_relative_base.append(curve_sliced_relative_base_ith_layer)

	for i in range(num_layers):
		num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(i)+'_*.csv'))
		curve_sliced_relative_ith_layer=[]
		curve_sliced_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_relative_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
			curve_sliced_ith_layer.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_relative.append(curve_sliced_relative_ith_layer)
		curve_sliced.append(curve_sliced_ith_layer)


	rr=redundancy_resolution(robot,positioner,curve_sliced)
	H=np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')

	positioner_js=rr.positioner_resolution(curve_sliced_relative,q_seed=q_positioner_seed)		#solve for positioner first
	
	###singularity js smoothing
	positioner_js=rr.introducing_tolerance2(positioner_js)
	positioner_js=rr.conditional_rolling_average(positioner_js)
	positioner_js[0][0][:,1]=positioner_js[1][0][0,1]


	for i in range(num_layers):
		for x in range(len(positioner_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',positioner_js[i][x],delimiter=',')

if __name__ == '__main__':
	main()