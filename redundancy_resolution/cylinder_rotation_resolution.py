import numpy as np
from tqdm import tqdm
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
from redundancy_resolution import *
sys.path.append('../toolbox')
from robot_def import *


def main():
	dataset='cylinder/'
	sliced_alg='dense_slice/'
	data_dir='../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
		pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_extended_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')
	R_torch=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])
	
	curve_sliced=[]
	for i in range(slicing_meta['num_layers']):
		curve_sliced.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',').reshape((-1,6)))
	
	curve_sliced_relative=copy.deepcopy(curve_sliced)
	positioner_js=[]
	for i in range(len(curve_sliced)):
		x=0
		positioner_js.append(np.linspace(np.array([-15.*np.pi/180.,np.pi/2]),np.array([-15.*np.pi/180.,np.pi/2+2*np.pi]),num=len(curve_sliced[i]),endpoint=False))
	

	curve_sliced_js=[]
	for i in tqdm(range(len(curve_sliced_relative))):			#solve for robot invkin

		curve_sliced_js_ith_layer=[]
		for j in range(len(curve_sliced_relative[i])): 
			###get positioner TCP world pose
			positioner_pose=positioner.fwd(positioner_js[i][j],world=True)
			p=positioner_pose.R@curve_sliced_relative[i][j,:3]+positioner_pose.p
			###solve for invkin
			if i==0 and j==0:
				q=robot.inv(p,R_torch,last_joints=np.zeros(6))[0]
				q_prev=q
			else:
				q=robot.inv(p,R_torch,last_joints=q_prev)[0]
				q_prev=q

			curve_sliced_js_ith_layer.append(np.array(q))
		curve_sliced_js.append(curve_sliced_js_ith_layer)

	for i in tqdm(range(slicing_meta['num_layers'])):

		np.savetxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',positioner_js[i],delimiter=',')
		np.savetxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',curve_sliced_js[i],delimiter=',')


if __name__ == '__main__':
	main()