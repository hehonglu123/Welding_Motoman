import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from path_calc import *
# from utils import *

def linear_fit(data,p_constraint=[]):
	###no constraint
	if len(p_constraint)==0:
		A=np.vstack((np.ones(len(data)),np.arange(0,len(data)))).T
		b=data
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		start_point=res[0]
		slope=res[1].reshape(1,-1)

		data_fit=np.dot(np.arange(0,len(data)).reshape(-1,1),slope)+start_point
	###with constraint point
	else:
		start_point=p_constraint

		A=np.arange(1,len(data)+1).reshape(-1,1)
		b=data-start_point
		res=np.linalg.lstsq(A,b,rcond=None)[0]
		slope=res.reshape(1,-1)

		data_fit=np.dot(np.arange(1,len(data)+1).reshape(-1,1),slope)+start_point

	return data_fit

class redundancy_resolution(object):
	###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot,positioner,curve_sliced):
		# curve_sliced: list of sliced layers, in curve frame
		# robot: welder robot
		# positioner: 2DOF rotational positioner
		self.robot=robot
		self.positioner=positioner
		self.curve_sliced=curve_sliced
	
	def conditional_rolling_average(self,positioner_js):
		###conditional rolling average
		tolerance=np.radians(3)
		steps=25
		positioner_js_new=copy.deepcopy(positioner_js)
		for i in range(len(positioner_js)):
			for j in range(len(positioner_js[i])):
				if get_angle(self.positioner.fwd(positioner_js[i][j],world=True).R[:,-1],[0,0,1])<tolerance:
					if j-steps<0:
						start_point=0
						end_point=j+steps
					elif j+steps>len(positioner_js[i]):
						end_point=len(positioner_js[i])
						start_point=j-steps
					else:
						start_point=j-steps
						end_point=j+steps

					positioner_js_new[i][j]=np.average(positioner_js[i][start_point:end_point],axis=0)
			###reverse
			positioner_js_new2=copy.deepcopy(positioner_js_new)
			for j in reversed(range(len(positioner_js[i]))):
				if get_angle(self.positioner.fwd(positioner_js[i][j],world=True).R[:,-1],[0,0,1])<tolerance:
					if j-steps<0:
						start_point=0
						end_point=j+steps
					elif j+steps>len(positioner_js[i]):
						end_point=len(positioner_js[i])
						start_point=j-steps
					else:
						start_point=j-steps
						end_point=j+steps

					positioner_js_new2[i][j]=np.average(positioner_js[i][start_point:end_point],axis=0)

		return positioner_js_new2

	def introducing_tolerance(self,positioner_js):
		### introduce tolerance to positioner inverse kinematics by linear fit
		tolerance=np.radians(3)
		
		for i in range(len(positioner_js)):
			start_idx=0
			end_idx=0
			for j in range(len(positioner_js[i])):
				if get_angle(self.positioner.fwd(positioner_js[i][j],world=True).R[:,-1],[0,0,1])<tolerance:
					start_idx=j
					for k in range(j+1,len(positioner_js[i])):
						if get_angle(self.positioner.fwd(positioner_js[i][k],world=True).R[:,-1],[0,0,1])>tolerance:
							end_idx=k
							break
					break

			if start_idx==0 and (end_idx>start_idx):
				positioner_js_temp=linear_fit(np.flip(positioner_js[i][start_idx:end_idx],axis=0),p_constraint=positioner_js[i][end_idx])
				positioner_js[i][start_idx:end_idx]=np.flip(positioner_js_temp,axis=0)
		

		return positioner_js

	def introducing_tolerance2(self,positioner_js):
		### introduce tolerance to positioner inverse kinematics by setting a constant 
		tolerance=np.radians(2)
		
		for i in range(len(positioner_js)):
			start_idx=0
			end_idx=0
			for j in range(len(positioner_js[i])):
				if get_angle(self.positioner.fwd(positioner_js[i][j],world=True).R[:,-1],[0,0,1])<tolerance:
					start_idx=j
					for k in range(j+1,len(positioner_js[i])):
						if get_angle(self.positioner.fwd(positioner_js[i][k],world=True).R[:,-1],[0,0,1])>tolerance:
							end_idx=k
							break
					break

			if start_idx==0 and (end_idx>start_idx):
				positioner_js[i][start_idx:end_idx]=positioner_js[i][end_idx]

		return positioner_js



	def baseline_joint(self,R_torch,curve_sliced_relative,curve_sliced_relative_base,q_init=np.zeros(6)):
		####baseline redundancy resolution, with fixed orientation
		positioner_js=self.positioner_resolution(curve_sliced_relative)		#solve for positioner first
		###TO FIX: override first layer positioner q2
		positioner_js[0][:,1]=positioner_js[1][0,1]
		
		###singularity js smoothing
		# positioner_js=self.conditional_rolling_average(positioner_js)
		positioner_js=self.introducing_tolerance2(positioner_js)
		positioner_js=self.conditional_rolling_average(positioner_js)
		positioner_js[0][:,1]=positioner_js[1][0,1]

		###append base layers positioner
		positioner_js_base=[copy.deepcopy(positioner_js[0])]*len(curve_sliced_relative_base)

		curve_sliced_js=[]
		for i in range(len(curve_sliced_relative)):			#solve for robot invkin
			curve_sliced_js_ith=[]
			for j in range(len(curve_sliced_relative[i])): 
				###get positioner TCP world pose
				positioner_pose=self.positioner.fwd(positioner_js[i][j],world=True)
				p=positioner_pose.R@curve_sliced_relative[i][j,:3]+positioner_pose.p
				###solve for invkin
				if i==0 and j==0:
					q=self.robot.inv(p,R_torch,last_joints=q_init)[0]
					q_prev=q
				else:
					q=self.robot.inv(p,R_torch,last_joints=q_prev)[0]
					q_prev=q

				curve_sliced_js_ith.append(q)
			curve_sliced_js.append(np.array(curve_sliced_js_ith))
		
		curve_sliced_js_base=[]
		for i in range(len(curve_sliced_relative_base)):			#solve for robot invkin
			curve_sliced_js_base_ith=[]
			for j in range(len(curve_sliced_relative_base[i])): 
				###get positioner TCP world pose
				positioner_pose=self.positioner.fwd(positioner_js_base[i][j],world=True)
				p=positioner_pose.R@curve_sliced_relative_base[i][j,:3]+positioner_pose.p
				###solve for invkin
				q=self.robot.inv(p,R_torch,last_joints=q_init)[0]

				curve_sliced_js_base_ith.append(q)
			curve_sliced_js_base.append(np.array(curve_sliced_js_base_ith))

		return positioner_js,curve_sliced_js,positioner_js_base,curve_sliced_js_base

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
			q_prev=[0,0]
			for j in reversed(range(len(curve_sliced_relative[i]))):	###reverse, solve from the end because start may be upward
				###curve normal as torch orientation, opposite of positioner
				positioner_js_ith.append(self.positioner.inv(-curve_sliced_relative[i][j,3:],q_prev))
				q_prev=positioner_js_ith[-1]

			positioner_js_ith.reverse()
			positioner_js_ith=np.array(positioner_js_ith)

			###filter noise
			positioner_js_ith[:,0]=moving_average(positioner_js_ith[:,0],padding=True)
			positioner_js_ith[:,1]=moving_average(positioner_js_ith[:,1],padding=True)

			positioner_js.append(np.array(positioner_js_ith))

		
		return positioner_js


	def positioner_qp(self,q_now,ez_relative):
		R_now=self.positioner.fwd(q_now,world=True).R
		n_now=R_now[:,-1]
		error_angle=get_angle(n_now,[0,0,1])
		
		J=self.positioner.jacobian(q_now)        #calculate current Jacobian
		JR=J[:3,:]
		JR_mod=-np.dot(hat(R_now),JR)

		H=np.dot(np.transpose(JR_mod),JR_mod)
		H=(H+np.transpose(H))/2

		ezdotd=(curve_normal[i]-pose_now.R[:,-1])

		f=-np.transpose(JR_mod)@ezdotd
		qdot=solve_qp(H,f)

		###line search
		alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_all[-1],qdot,curve_sliced_relative[i],))
		if alpha<0.01:
			break

			
	def positioner_qp_smooth(self,positioner_js, curve_sliced_relative):
		### positioner trajectory smoothing with QP
		positioner_js_out=[]
		tolerance=np.radians(3)

		for i in range(len(curve_sliced_relative)):
			positioner_js_ith_out=[positioner_js[i][0]]
			q_all=[positioner_js[i][0]]
			Kw=1
			for i in range(len(curve)):
				# print(i)
				try:
					now=time.time()
					error_angle=999

					while error_angle>tolerance:

						R_now=self.positioner.fwd(q_all[-1],world=True).R
						n_now=R_now[:,-1]
						error_angle=get_angle(n_now,[0,0,1])
						
						J=self.positioner.jacobian(q_all[-1])        #calculate current Jacobian
						JR=J[:3,:]
						JR_mod=-np.dot(hat(R_now),JR)

						H=np.dot(np.transpose(JR_mod),JR_mod)
						H=(H+np.transpose(H))/2

						vd=curve[i]-pose_now.p
						ezdotd=(curve_normal[i]-pose_now.R[:,-1])

						f=-np.transpose(JR_mod)@ezdotd
						qdot=solve_qp(H,f)

						###line search
						alpha=fminbound(self.error_calc,0,0.999999999999999999999,args=(q_all[-1],qdot,curve_sliced_relative[i],))
						if alpha<0.01:
							break
						q_all.append(q_all[-1]+alpha*qdot)
						# print(q_all[-1])
				except:
					q_out.append(q_all[-1])
					traceback.print_exc()
					raise AssertionError
					break

				positioner_js_ith_out.append(q_all[-1])

			positioner_js_out.append(np.array(positioner_js_ith_out))
		

		return positioner_js_out


def main():
	return

if __name__ == '__main__':
	main()