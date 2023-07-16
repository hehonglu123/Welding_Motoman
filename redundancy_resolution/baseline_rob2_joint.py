import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *
from utils import *


def main():
	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
		pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')
	
	dataset='blade0.1/'
	sliced_alg='auto_slice/'
	data_dir='../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)
		

	###################################################################
	curve_sliced=[]
	curve_sliced_js=[]
	for i in range(slicing_meta['num_layers']):
		num_sections=len(glob.glob(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_*.csv'))
		curve_sliced_js_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_js_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_js.append(curve_sliced_js_ith_layer)
	# for i in range(slicing_meta['num_layers']):
	# 	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(i)+'_*.csv'))
	# 	curve_sliced_relative_ith_layer=[]
	# 	curve_sliced_ith_layer=[]
	# 	for x in range(num_sections):
	# 		curve_sliced_relative_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
	# 		curve_sliced_ith_layer.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
	# 	curve_sliced_relative.append(curve_sliced_relative_ith_layer)
	# 	curve_sliced.append(curve_sliced_ith_layer)



	rr=redundancy_resolution(robot,positioner,None)

	rob2_curve_js=rr.rob2_flir_resolution(curve_sliced_js,robot2,measure_distance=500)

	for i in range(slicing_meta['num_layers']):
		for x in range(len(rob2_curve_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',rob2_curve_js[i][x],delimiter=',')

	# for i in range(slicing_meta['num_baselayers']):
	# 	for x in range(len(positioner_js_base[i])):
	# 		np.savetxt(data_dir+'curve_sliced_js/D500B_base_js'+str(i)+'_'+str(x)+'.csv',positioner_js_base[i][x],delimiter=',')
	# 		np.savetxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(i)+'_'+str(x)+'.csv',curve_sliced_js_base[i][x],delimiter=',')


if __name__ == '__main__':
	main()