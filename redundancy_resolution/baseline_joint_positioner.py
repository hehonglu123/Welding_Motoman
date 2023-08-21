import numpy as np
import sys, traceback, time, copy, glob, yaml
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *


def main():
	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

	dataset='diamond/'
	sliced_alg='cont_sections/'
	data_dir='../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)


	curve_sliced_relative_base=[]
	curve_sliced_relative=[]
	curve_sliced=[]
	for i in range(slicing_meta['num_baselayers']):
		num_sections=len(glob.glob(data_dir+'curve_sliced_relative/baselayer'+str(i)+'_*.csv'))
		curve_sliced_relative_base_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_relative_base_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_relative_base.append(curve_sliced_relative_base_ith_layer)

	for i in range(slicing_meta['num_layers']):
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

	positioner_js=rr.positioner_resolution(curve_sliced_relative,q_seed=slicing_meta['q_positioner_seed'],smooth_filter=slicing_meta['smooth_filter'])		#solve for positioner first
	# positioner_js=rr.positioner_resolution_qp(curve_sliced_relative,q_seed=slicing_meta['q_positioner_seed'])		#solve for positioner first
	
	###singularity js smoothing
	positioner_js=rr.introducing_tolerance2(positioner_js)
	positioner_js=rr.conditional_rolling_average(positioner_js)
	positioner_js[0][0][:,1]=positioner_js[1][0][0,1]


	for i in range(slicing_meta['num_layers']):
		for x in range(len(positioner_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',positioner_js[i][x],delimiter=',')

if __name__ == '__main__':
	main()