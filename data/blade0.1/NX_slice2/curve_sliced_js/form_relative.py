import numpy as np
import sys, traceback, time, copy, glob
import matplotlib.pyplot as plt
from matplotlib import cm
from general_robotics_toolbox import *
sys.path.append('../../../../toolbox')
from robot_def import *
from multi_robot import *

def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice2/'
	data_dir='../../../../data/'+dataset+sliced_alg
	curve_sliced_relative_all=[]
	curve_sliced_relative_form_all=[]

	robot=robot_obj('MA2010_A0',def_path='../../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../../config/weldgun.csv',\
		pulse2deg_file_path='../../../../config/MA2010_A0_pulse2deg.csv',d=20)
	positioner=positioner_obj('D500B',def_path='../../../../config/D500B_robot_default_config.yml',tool_file_path='../../../../config/positioner_tcp.csv',\
		pulse2deg_file_path='../../../../config/D500B_pulse2deg.csv',base_transformation_file='../../../../config/D500B_pose.csv')
	
	num_layer_start=70
	num_layer_end=75
	for layer in range(num_layer_start,num_layer_end):
		num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))
		for x in range(num_sections):
			if layer % 2==1:
				curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
				positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
				curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
			else:
				curve_sliced_js=np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=','),axis=0)
				positioner_js=np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=','),axis=0)
				curve_sliced_relative=np.flip(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=','),axis=0)

			curve_sliced_relative_all.append(curve_sliced_relative)
			_,_,_,_,relative_path_exe,_= form_relative_path(curve_sliced_js,positioner_js,robot,positioner)
			curve_sliced_relative_form_all.append(relative_path_exe)

	curve_sliced_relative_all=np.concatenate(curve_sliced_relative_all)
	curve_sliced_relative_form_all=np.concatenate(curve_sliced_relative_form_all)

			

	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	ax.plot3D(curve_sliced_relative_all[::vis_step,0],curve_sliced_relative_all[::vis_step,1],curve_sliced_relative_all[::vis_step,2],'r.-')
	# ax.quiver(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],relative_path_exe_R[::vis_step,0,-1],relative_path_exe_R[::vis_step,1,-1],relative_path_exe_R[::vis_step,2,-1],length=0.3, normalize=True)

	# ax.plot3D(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],'g.-')
	# ax.quiver(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],length=0.3, normalize=True)


	# ax.set_xlim3d(-80, 80)
	# ax.set_ylim3d(0, 80)
	# ax.set_zlim3d(-80, 80)
	plt.show()
if __name__ == '__main__':
	main()