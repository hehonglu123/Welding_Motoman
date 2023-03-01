import numpy as np
import sys, traceback, time, copy
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
	num_layers=2
	curve_sliced_relative=[]



	robot=robot_obj('MA2010_A0',def_path='../../../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../../../config/weldgun.csv',\
		pulse2deg_file_path='../../../../config/MA2010_A0_pulse2deg.csv',d=20)
	positioner=positioner_obj('D500B',def_path='../../../../config/D500B_robot_default_config.yml',tool_file_path='../../../../config/positioner_tcp.csv',\
		pulse2deg_file_path='../../../../config/D500B_pulse2deg.csv',base_transformation_file='../../../../config/D500B_pose.csv')
	
	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	for i in range(num_layers):
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'.csv',delimiter=',')
		curve_exe1,curve_exe2,curve_exe_R1,curve_exe_R2,relative_path_exe,relative_path_exe_R=form_relative_path(curve_sliced_js,positioner_js,robot,positioner)

		ax.plot3D(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],'r.-')
		ax.quiver(relative_path_exe[::vis_step,0],relative_path_exe[::vis_step,1],relative_path_exe[::vis_step,2],relative_path_exe_R[::vis_step,0,-1],relative_path_exe_R[::vis_step,1,-1],relative_path_exe_R[::vis_step,2,-1],length=0.3, normalize=True)


		# curve_exe2_world=[]
		# curve_exe_R2_world=[]
		# for j in range(len(curve_sliced_js)):
		# 	transform=positioner.fwd(positioner_js[j],world=True)
		# 	curve_exe2_world.append(transform.p)
		# 	curve_exe_R2_world.append(transform.R)
		# curve_exe2_world=np.array(curve_exe2_world)
		# curve_exe_R2_world=np.array(curve_exe_R2_world)
		
		# ax.plot3D(curve_exe1[::vis_step,0],curve_exe1[::vis_step,1],curve_exe1[::vis_step,2],'r.-')
		# ax.plot3D(curve_exe2_world[::vis_step,0],curve_exe2_world[::vis_step,1],curve_exe2_world[::vis_step,2],'g.-')

	# ax.set_xlim3d(-80, 80)
	# ax.set_ylim3d(0, 80)
	# ax.set_zlim3d(-80, 80)
	plt.show()
if __name__ == '__main__':
	main()