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
	
	dataset='funnel/'
	sliced_alg='circular_slice/'
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

	


	R_torch=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,curve_sliced)

	positioner_js,curve_sliced_js,positioner_js_base,curve_sliced_js_base=rr.baseline_joint(R_torch,curve_sliced_relative,curve_sliced_relative_base,q_seed,slicing_meta['q_positioner_seed'],smooth_filter=slicing_meta['smooth_filter'])

	for i in range(slicing_meta['num_layers']):
		for x in range(len(positioner_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',positioner_js[i][x],delimiter=',')
			np.savetxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',curve_sliced_js[i][x],delimiter=',')

	for i in range(slicing_meta['num_baselayers']):
		for x in range(len(positioner_js_base[i])):
			np.savetxt(data_dir+'curve_sliced_js/D500B_base_js'+str(i)+'_'+str(x)+'.csv',positioner_js_base[i][x],delimiter=',')
			np.savetxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(i)+'_'+str(x)+'.csv',curve_sliced_js_base[i][x],delimiter=',')


if __name__ == '__main__':
	main()