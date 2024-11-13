import numpy as np
import sys, traceback, time, copy, glob, yaml, pathlib
from general_robotics_toolbox import *
from redundancy_resolution import *
from motoman_def import *


def main():
	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

	# dataset='face/'
	# sliced_alg='auto_slice/'
	dataset='face_mesh_tanja_straight/'
	sliced_alg='slicing_result_10/'

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
	
	### if exceed joint limit, interpolate to the nearest joint limit
	positioner_js = rr.positioner_joint_limit_interpolation(positioner_js)

	###singularity js smoothing
	positioner_js=rr.introducing_tolerance2(positioner_js)
	positioner_js=rr.conditional_rolling_average(positioner_js)
	positioner_js[0][0][:,1]=positioner_js[1][0][0,1]

	# plot js every N layers 
	# N = 1
	# import matplotlib.animation as animation
	# fig, ax = plt.subplots()
	# ax.plot([0, 600], [np.degrees(positioner.upper_limit[0])]*2, 'r--')
	# ax.plot([0, 600], [np.degrees(positioner.lower_limit[0])]*2, 'r--')
	# def update(num, positioner_js):
	# 	# ax.clear()
	# 	line.set_data(np.arange(0,len(positioner_js[num][0][:,0])), np.degrees(positioner_js[num][0][:,0]))
	# 	ax.set_title('Layer ' + str(num))
	# 	return line,
	# line, = ax.plot([], [])
	# ani = animation.FuncAnimation(fig, update, frames=range(0, len(positioner_js), N), fargs=(positioner_js,), blit=True)
	# plt.show()

	pathlib.Path(data_dir+'curve_sliced_js').mkdir(parents=True, exist_ok=True)

	for i in range(slicing_meta['num_layers']):
		for x in range(len(positioner_js[i])):
			np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',positioner_js[i][x],delimiter=',')
	
	

if __name__ == '__main__':
	main()