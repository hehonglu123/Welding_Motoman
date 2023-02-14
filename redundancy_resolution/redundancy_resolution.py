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
		positioner_js=self.positioner_resolution(curve_sliced_relative)		#solve for positioner first
	
		curve_sliced_js=np.zeros((len(curve_sliced_relative),len(curve_sliced_relative[0]),6))
		for i in range(len(curve_sliced_relative)):			#solve for robot invkin
			for j in range(len(curve_sliced_relative[i])): 
				###get positioner TCP world pose
				positioner_pose=self.positioner.fwd(positioner_js[i][j],world=True)
				p=positioner_pose.R@curve_sliced_relative[i][j,:3]+positioner_pose.p
				print(positioner_js[i][j])
				print(positioner_pose.R@curve_sliced_relative[i][j,3:])
				print(self.positioner.fwd(positioner_js[i][j]))
				print(positioner_pose.p)
				###solve for invkin
				if i==0 and j==0:
					print('starting p: ',p)
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


	def positioner_resolution(self,curve_sliced_relative):
		###resolve 2DOF positioner joint angle 
		positioner_js=[]
		for i in range(len(curve_sliced_relative)):
			positioner_js_ith=[]
			for j in range(len(curve_sliced_relative[i])):
				###curve normal as torch orientation, opposite of positioner
				positioner_js_ith.append(self.positioner.inv(-curve_sliced_relative[i][j,3:]))

			positioner_js.append(positioner_js_ith)
		return positioner_js


def main():
	return

if __name__ == '__main__':
	main()