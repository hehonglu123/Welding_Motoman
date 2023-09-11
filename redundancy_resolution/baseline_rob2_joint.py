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
		pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose_mocap.csv')
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')
	
	dataset='funnel/'
	sliced_alg='circular_slice/'
	data_dir='../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)
		

	###################################################################
	curve_sliced=[]
	curve_sliced_js=[]
	curve_sliced_js_base=[]
	for i in range(slicing_meta['num_layers']):
		num_sections=len(glob.glob(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_*.csv'))
		curve_sliced_js_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_js_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_js.append(curve_sliced_js_ith_layer)

	for i in range(slicing_meta['num_baselayers']):
		num_sections=len(glob.glob(data_dir+'curve_sliced_js/MA2010_base_js'+str(i)+'_*.csv'))
		curve_sliced_js_base_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_js_base_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		curve_sliced_js_base.append(curve_sliced_js_base_ith_layer)



	rr=redundancy_resolution(robot,positioner,None)
	distance=400
	rob2_curve_js_base=rr.rob2_flir_resolution(curve_sliced_js_base,robot2,measure_distance=distance)
	for i in range(slicing_meta['num_baselayers']):
		for x in range(len(rob2_curve_js_base[i])):
			np.savetxt(data_dir+'curve_sliced_js/MA1440_base_js'+str(i)+'_'+str(x)+'.csv',rob2_curve_js_base[i][x],delimiter=',')

	rob2_curve_js=rr.rob2_flir_resolution(curve_sliced_js,robot2,measure_distance=distance)
	for i in range(slicing_meta['num_layers']):
		for x in range(len(rob2_curve_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',rob2_curve_js[i][x],delimiter=',')


if __name__ == '__main__':
	main()