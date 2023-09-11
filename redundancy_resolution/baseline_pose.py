import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *


def main():
	dataset='funnel/'
	sliced_alg='circular_slice/'
	data_dir='../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	base_thickness=3
	curve_sliced=[]
	for i in range(slicing_meta['num_layers']):
		###get number of disconnected sections
		num_sections=len(glob.glob(data_dir+'curve_sliced/slice'+str(i)+'_*.csv'))
		curve_sliced_ith_layer=[]
		for x in range(num_sections):
			curve_sliced_ith_layer.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))

		curve_sliced.append(curve_sliced_ith_layer)

	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

	R_torch=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,curve_sliced)
	# H=rr.baseline_pose(vec=np.array([-0.95,0.31224989992]))

	try:
		H=np.array(slicing_meta['H'])
	except KeyError:
		try:
			H=rr.baseline_pose(vec=slicing_meta['placing_vector'])
		except KeyError:
			H=rr.baseline_pose()

	
	H[2,-1]+=slicing_meta['num_baselayers']*slicing_meta['baselayer_thickness']

	np.savetxt(data_dir+'curve_pose.csv',H,delimiter=',')

	###convert curve slices to positioner TCP frame
	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	


	curve_sliced_relative=copy.deepcopy(rr.curve_sliced)
	for i in range(len(rr.curve_sliced)):
		for x in range(len(rr.curve_sliced[i])):
			for j in range(len(rr.curve_sliced[i][x])):
				
				curve_sliced_relative[i][x][j,:3]=np.dot(H,np.hstack((rr.curve_sliced[i][x][j,:3],[1])).T)[:-1]
				#convert curve direction to base frame
				curve_sliced_relative[i][x][j,3:]=np.dot(H[:3,:3],rr.curve_sliced[i][x][j,3:]).T


			if i==0:
				ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'r.-')
			elif i==1:
				ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'g.-')
			else:
				ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'b.-')

			ax.quiver(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],curve_sliced_relative[i][x][::vis_step,3],curve_sliced_relative[i][x][::vis_step,4],curve_sliced_relative[i][x][::vis_step,5],length=0.3, normalize=True)
		
			np.savetxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',curve_sliced_relative[i][x],delimiter=',')
	
	###base layer appendant
	curve_sliced_relative_base=[]
	for i in range(slicing_meta['num_baselayers']):
		curve_sliced_relative_base.append(copy.deepcopy(curve_sliced_relative[0]))
		for x in range(len(curve_sliced_relative[0])):

			curve_sliced_relative_base[-1][x][:,2]-=(slicing_meta['num_baselayers']-i)*base_thickness
			ax.plot3D(curve_sliced_relative_base[i][x][::vis_step,0],curve_sliced_relative_base[i][x][::vis_step,1],curve_sliced_relative_base[i][x][::vis_step,2],'r.-')
			ax.quiver(curve_sliced_relative_base[i][x][::vis_step,0],curve_sliced_relative_base[i][x][::vis_step,1],curve_sliced_relative_base[i][x][::vis_step,2],curve_sliced_relative_base[i][x][::vis_step,3],curve_sliced_relative_base[i][x][::vis_step,4],curve_sliced_relative_base[i][x][::vis_step,5],length=0.3, normalize=True)
	
			np.savetxt(data_dir+'curve_sliced_relative/baselayer'+str(i)+'_'+str(x)+'.csv',curve_sliced_relative_base[i][x],delimiter=',')


	plt.title(dataset[:-1]+' first '+str(slicing_meta['num_layers'])+' slices')
	plt.show()



if __name__ == '__main__':
	main()