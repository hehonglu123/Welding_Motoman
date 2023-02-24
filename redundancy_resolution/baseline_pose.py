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
	base_thickness=3
	num_baselayers=2
	curve_sliced=[]
	for i in range(num_layers):
		curve_sliced.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'.csv',delimiter=','))

	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=20)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')

	R_torch=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,curve_sliced)
	H=rr.baseline_pose()
	H[2,-1]+=num_baselayers*base_thickness

	np.savetxt(data_dir+'curve_pose.csv',H,delimiter=',')

	###convert curve slices to positioner TCP frame
	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	


	curve_sliced_relative=copy.deepcopy(rr.curve_sliced)
	for x in range(len(rr.curve_sliced)):
		for i in range(len(rr.curve_sliced[x])):
			curve_sliced_relative[x][i,:3]=np.dot(H,np.hstack((rr.curve_sliced[x][i,:3],[1])).T)[:-1]
			#convert curve direction to base frame
			curve_sliced_relative[x][i,3:]=np.dot(H[:3,:3],rr.curve_sliced[x][i,3:]).T

		if x==0:
			ax.plot3D(curve_sliced_relative[x][::vis_step,0],curve_sliced_relative[x][::vis_step,1],curve_sliced_relative[x][::vis_step,2],'r.-')
		elif x==1:
			ax.plot3D(curve_sliced_relative[x][::vis_step,0],curve_sliced_relative[x][::vis_step,1],curve_sliced_relative[x][::vis_step,2],'g.-')
		else:
			ax.plot3D(curve_sliced_relative[x][::vis_step,0],curve_sliced_relative[x][::vis_step,1],curve_sliced_relative[x][::vis_step,2],'b.-')

		ax.quiver(curve_sliced_relative[x][::vis_step,0],curve_sliced_relative[x][::vis_step,1],curve_sliced_relative[x][::vis_step,2],curve_sliced_relative[x][::vis_step,3],curve_sliced_relative[x][::vis_step,4],curve_sliced_relative[x][::vis_step,5],length=0.3, normalize=True)
	
		np.savetxt(data_dir+'curve_sliced_relative/slice'+str(x)+'.csv',curve_sliced_relative[x],delimiter=',')
	
	###base layer appendant
	curve_sliced_relative_base=[]
	for x in range(num_baselayers):
		curve_sliced_relative_base.append(copy.deepcopy(curve_sliced_relative[0]))
		curve_sliced_relative_base[-1][:,2]-=(num_baselayers-x)*base_thickness
		ax.plot3D(curve_sliced_relative_base[x][::vis_step,0],curve_sliced_relative_base[x][::vis_step,1],curve_sliced_relative_base[x][::vis_step,2],'r.-')
		ax.quiver(curve_sliced_relative_base[x][::vis_step,0],curve_sliced_relative_base[x][::vis_step,1],curve_sliced_relative_base[x][::vis_step,2],curve_sliced_relative_base[x][::vis_step,3],curve_sliced_relative_base[x][::vis_step,4],curve_sliced_relative_base[x][::vis_step,5],length=0.3, normalize=True)
	
		np.savetxt(data_dir+'curve_sliced_relative/baselayer'+str(x)+'.csv',curve_sliced_relative_base[x],delimiter=',')


	plt.title('0.1 blade first X layers')
	plt.show()



if __name__ == '__main__':
	main()