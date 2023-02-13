import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
# from lambda_calc import *
# from utils import *

class redundancy_resolution(object):
	###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot,positioner,curve_sliced):
		# curve_sliced: list of sliced layers, in curve frame
		# robot: welder robot
		# positioner: 2DOF rotational positioner
		self.robot=robot
		self.positioner=positioner
		self.curve_sliced=curve_sliced

	

	def baseline_joint(self,R_torch,curve_sliced_relative,q_init=np.zeros(6)):
		####baseline redundancy resolution, with fixed orientation
		positioner_js=self.positioner_resolution()		#solve for positioner first
	
		curve_sliced_js=np.zeros((len(curve_sliced_relative),len(curve_sliced_relative[0]),6))
		for i in range(len(curve_sliced_relative)):			#solve for robot invkin
			for j in range(len(curve_sliced_relative[i])): 
				###get positioner TCP world pose
				positioner_pose=self.positioner.fwd(positioner_js[i][j],world=True)
				p=positioner_pose.R@curve_sliced_relative[i,j,:3]+positioner_pose
				###solve for invkin
				if i==0 and j==0:
					q=self.robot.inv(p,R_torch,last_joints=q_init)
				else:
					q=self.robot.inv(p,R_torch,last_joints=q_prev)
					q_prev=q

		return positioner_js,curve_sliced_js

	def baseline_pose(self):
		###where to place the welded part on positioner
		###assume first layer normal z always vertical
		###baseline, only look at first layer
		COM=np.average(self.curve_sliced[0][:,:3],axis=0)

		###determine Vx by eig(cov)
		curve_cov=np.cov(self.curve_sliced[0][:,:3].T)
		eigenValues, eigenVectors = np.linalg.eig(curve_cov)
		idx = eigenValues.argsort()[::-1]   
		eigenValues = eigenValues[idx]
		eigenVectors = eigenVectors[:,idx]
		Vx=eigenVectors[0]

		###put normal along G direction
		N=-np.sum(self.curve_sliced[0][:,3:],axis=0)
		N=N/np.linalg.norm(N)

		Vx=VectorPlaneProjection(Vx,N)

		###form transformation
		R=np.vstack((Vx,np.cross(N,Vx),N))
		T=-R@COM

		return H_from_RT(R,T)


	def positioner_resolution(self):
		###resolve 2DOF positioner joint angle 
		positioner_js=np.zeros((len(curve_sliced_relative),len(curve_sliced_relative[0]),2))
		for i in range(len(curve_sliced_relative)):
			for j in range(len(curve_sliced_relative[i])):
				positioner_js[i][j]=self.positioner.inv(curve_sliced_relative[i,j,:3],curve_sliced_relative[i,j,3:])
		return positioner_js


def main():
	dataset='blade0.1/'
	sliced_alg='NX_slice/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=3
	curve_sliced=[]
	for i in range(num_layers):
		curve_sliced.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'.csv',delimiter=','))

	robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
		pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')

	R_torch=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

	rr=redundancy_resolution(robot,positioner,np.array(curve_sliced))
	H=rr.baseline_pose()

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
	
	print(curve_sliced_relative[0][:,3:])
	plt.title('0.1 blade first 3 layers')
	plt.show()

	# positioner_js,curve_sliced_js=rr.baseline(R_torch,q_seed)

if __name__ == '__main__':
	main()