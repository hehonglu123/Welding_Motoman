import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
from redundancy_resolution import *
from motoman_def import *


def main():
	dataset='bell/'
	sliced_alg='dense_slice/'
	data_dir='../../geometry_data/'+dataset+sliced_alg
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

	
	H[2,-1]+=slicing_meta['num_baselayers']*slicing_meta['baselayer_thickness']+slicing_meta['num_supportlayers']*slicing_meta['supportlayer_thickness']

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


			# if i==0:
			# 	ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'r.-')
			# elif i==1:
			# 	ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'g.-')
			# else:
			ax.plot3D(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],'b.-')

			ax.quiver(curve_sliced_relative[i][x][::vis_step,0],curve_sliced_relative[i][x][::vis_step,1],curve_sliced_relative[i][x][::vis_step,2],curve_sliced_relative[i][x][::vis_step,3],curve_sliced_relative[i][x][::vis_step,4],curve_sliced_relative[i][x][::vis_step,5],length=0.3, normalize=True)
		
			np.savetxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',curve_sliced_relative[i][x],delimiter=',')
	

	###support layer appendant
	curve_sliced_relative_support=[]

	support_layer_temp=copy.deepcopy(curve_sliced_relative[0])
	#extend the layer forward and backward by 10mm
	extension_length=10
	point_distance = np.mean(np.linalg.norm(np.diff(support_layer_temp[0][:,:3],axis=0),axis=1))

	num_points_extension=int(extension_length/point_distance)
	init_ext=np.zeros((num_points_extension,6))
	init_vec=support_layer_temp[0][0,:3]-support_layer_temp[0][1,:3]
	init_point=support_layer_temp[0][0,:3]+extension_length*init_vec/np.linalg.norm(init_vec)
	init_ext[:,:3]=np.linspace(init_point,support_layer_temp[0][0,:3],num_points_extension,endpoint=False).reshape((-1,3))
	init_ext[:,3:]=support_layer_temp[0][0,3:]

	end_vec=support_layer_temp[0][-1,:3]-support_layer_temp[0][-2,:3]
	end_point=support_layer_temp[0][-1,:3]+extension_length*end_vec/np.linalg.norm(end_vec)
	end_ext=np.zeros((num_points_extension,6))
	end_ext[:,:3]=np.flip(np.linspace(end_point,support_layer_temp[0][-1,:3],num_points_extension,endpoint=False).reshape((-1,3)),axis=0)
	end_ext[:,3:]=support_layer_temp[0][-1,3:]

	support_layer_temp[0]=np.vstack((init_ext,support_layer_temp[0],end_ext))

	for i in range(slicing_meta['num_supportlayers']):
		curve_sliced_relative_support.append(copy.deepcopy(support_layer_temp))
		for x in range(len(curve_sliced_relative[0])):

			curve_sliced_relative_support[-1][x][:,2]-=(slicing_meta['num_supportlayers']-i)*slicing_meta['supportlayer_thickness']
			ax.plot3D(curve_sliced_relative_support[i][x][::vis_step,0],curve_sliced_relative_support[i][x][::vis_step,1],curve_sliced_relative_support[i][x][::vis_step,2],'g.-')
			ax.quiver(curve_sliced_relative_support[i][x][::vis_step,0],curve_sliced_relative_support[i][x][::vis_step,1],curve_sliced_relative_support[i][x][::vis_step,2],curve_sliced_relative_support[i][x][::vis_step,3],curve_sliced_relative_support[i][x][::vis_step,4],curve_sliced_relative_support[i][x][::vis_step,5],length=0.3, normalize=True)
	
			np.savetxt(data_dir+'curve_sliced_relative/support_slice'+str(i)+'_'+str(x)+'.csv',curve_sliced_relative_support[i][x],delimiter=',')



	###base layer appendant
	curve_sliced_relative_base=[]
	for i in range(slicing_meta['num_baselayers']):
		curve_sliced_relative_base.append(copy.deepcopy(support_layer_temp))
		for x in range(len(support_layer_temp)):

			curve_sliced_relative_base[-1][x][:,2]-=((slicing_meta['num_baselayers']-i)*base_thickness+slicing_meta['num_supportlayers']*slicing_meta['supportlayer_thickness'])
			ax.plot3D(curve_sliced_relative_base[i][x][::vis_step,0],curve_sliced_relative_base[i][x][::vis_step,1],curve_sliced_relative_base[i][x][::vis_step,2],'r.-')
			ax.quiver(curve_sliced_relative_base[i][x][::vis_step,0],curve_sliced_relative_base[i][x][::vis_step,1],curve_sliced_relative_base[i][x][::vis_step,2],curve_sliced_relative_base[i][x][::vis_step,3],curve_sliced_relative_base[i][x][::vis_step,4],curve_sliced_relative_base[i][x][::vis_step,5],length=0.3, normalize=True)
			np.savetxt(data_dir+'curve_sliced_relative/base_slice'+str(i)+'_'+str(x)+'.csv',curve_sliced_relative_base[i][x],delimiter=',')

	
	set_axes_equal(ax)
	plt.title(dataset[:-1]+' first '+str(slicing_meta['num_layers'])+' slices')
	plt.show()



if __name__ == '__main__':
	main()