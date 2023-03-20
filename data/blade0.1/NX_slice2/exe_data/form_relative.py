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
	curve_sliced_relative=[]
	num_layers=30

	robot=robot_obj('MA2010_A0',def_path='../../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../../config/weldgun.csv',\
		pulse2deg_file_path='../../../../config/MA2010_A0_pulse2deg.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../../../../config/D500B_robot_default_config.yml',tool_file_path='../../../../config/positioner_tcp.csv',\
		pulse2deg_file_path='../../../../config/D500B_pulse2deg.csv',base_transformation_file='../../../../config/D500B_pose.csv')
	joint_recording=np.loadtxt('joint_recording.csv',delimiter=',')

	timestamp=joint_recording[:,0]
	robot_js=joint_recording[:,1:7]
	positioner_js=joint_recording[:,-2:]

	curve_sliced_relative=[]
	for i in range(num_layers):
		###get number of disconnected sections
		num_sections=len(glob.glob(data_dir+'curve_sliced/slice'+str(i)+'_*.csv'))

		slicei_complete=[]
		for x in range(num_sections):
			if i % 2 ==1:
				slicei_complete.append(np.loadtxt('../curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=','))
			else:
				slicei_complete.append(np.flip(np.loadtxt('../curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=','),axis=0))
		
		curve_sliced_relative.append(np.concatenate(slicei_complete,axis=0))

	curve_sliced_relative=np.concatenate(curve_sliced_relative,axis=0)


	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	
	curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,relative_path_exe,relative_path_exe_R=form_relative_path(robot_js,positioner_js,robot,positioner)

	ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'r.-')
	# ax.quiver(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],relative_path_exe_R[::vis_step,0,-1],relative_path_exe_R[::vis_step,1,-1],relative_path_exe_R[::vis_step,2,-1],length=0.3, normalize=True)

	ax.plot3D(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],'g.-')
	# ax.quiver(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],length=0.3, normalize=True)


	# ax.set_xlim3d(-80, 80)
	# ax.set_ylim3d(0, 80)
	# ax.set_zlim3d(-80, 80)
	plt.show()
if __name__ == '__main__':
	main()