from general_robotics_toolbox import * 
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox

import numpy as np
import yaml, copy, time
import pickle
from utils import *

def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
ex=np.array([[1.],[0.],[0.]])
ey=np.array([[0.],[1.],[0.]])
ez=np.array([[0.],[0.],[1.]])


class robot_obj(object):
	###robot object class
	def __init__(self,robot_name,def_path,tool_file_path='',base_transformation_file='',d=0,acc_dict_path='',pulse2deg_file_path=''):
		#def_path: robot 			definition yaml file, name must include robot vendor
		#tool_file_path: 			tool transformation to robot flange csv file
		#base_transformation_file: 	base transformation to world frame csv file
		#d: 						tool z extension
		#acc_dict_path: 			accleration profile

		self.robot_name=robot_name
		with open(def_path, 'r') as f:
			self.robot = rr_rox.load_robot_info_yaml_to_robot(f)

		self.def_path=def_path
		#define robot without tool
		self.robot_def_nT=Robot(self.robot.H,self.robot.P,self.robot.joint_type)

		if len(tool_file_path)>0:
			tool_H=np.loadtxt(tool_file_path,delimiter=',')
			self.robot.R_tool=tool_H[:3,:3]
			self.robot.p_tool=tool_H[:3,-1]+np.dot(tool_H[:3,:3],np.array([0,0,d]))
			self.p_tool=self.robot.p_tool
			self.R_tool=self.robot.R_tool		

		if len(base_transformation_file)>0:
			self.base_H=np.loadtxt(base_transformation_file,delimiter=',')
		else:
			self.base_H=np.eye(4)

		if len(pulse2deg_file_path)>0:
			self.pulse2deg=np.loadtxt(pulse2deg_file_path,delimiter=',')


		###set attributes
		self.upper_limit=self.robot.joint_upper_limit 
		self.lower_limit=self.robot.joint_lower_limit 
		self.joint_vel_limit=self.robot.joint_vel_limit 
		self.joint_acc_limit=self.robot.joint_acc_limit 

		###acceleration table
		if len(acc_dict_path)>0:
			acc_dict= pickle.load(open(acc_dict_path,'rb'))
			q2_config=[]
			q3_config=[]
			q1_acc_n=[]
			q1_acc_p=[]
			q2_acc_n=[]
			q2_acc_p=[]
			q3_acc_n=[]
			q3_acc_p=[]
			for key, value in acc_dict.items():
				q2_config.append(key[0])
				q3_config.append(key[1])
				q1_acc_n.append(value[0%len(value)])
				q1_acc_p.append(value[1%len(value)])
				q2_acc_n.append(value[2%len(value)])
				q2_acc_p.append(value[3%len(value)])
				q3_acc_n.append(value[4%len(value)])
				q3_acc_p.append(value[5%len(value)])
			self.q2q3_config=np.array([q2_config,q3_config]).T
			self.q1q2q3_acc=np.array([q1_acc_n,q1_acc_p,q2_acc_n,q2_acc_p,q3_acc_n,q3_acc_p]).T

	def get_acc(self,q_all,direction=[]):
		###get acceleration limit from q config, assume last 3 joints acc fixed direction is 3 length vector, 0 is -, 1 is +
		#if a single point
		if q_all.ndim==1:
			###find closest q2q3 config, along with constant last 3 joints acc
			idx=np.argmin(np.linalg.norm(self.q2q3_config-q_all[1:3],axis=1))
			acc_lim=[]
			if len(direction)==0:
				raise AssertionError('direciton not provided')
				return
			for d in direction:
				acc_lim.append(self.q1q2q3_acc[idx][2*len(acc_lim)+d])

			return np.append(acc_lim,self.joint_acc_limit[-3:])
		#if a list of points
		else:
			dq=np.gradient(q_all,axis=0)[:,:3]
			direction=(np.sign(dq)+1)/2
			direction=direction.astype(int)
			acc_limit_all=[]
			for i in range(len(q_all)):
				idx=np.argmin(np.linalg.norm(self.q2q3_config-q_all[i][1:3],axis=1))
				acc_lim=[]
				for d in direction[i]:
					acc_lim.append(self.q1q2q3_acc[idx][2*len(acc_lim)+d])

				acc_limit_all.append(np.append(acc_lim,self.joint_acc_limit[-3:]))

		return np.array(acc_limit_all)

	def fwd(self,q_all,world=False,qlim_override=False):
		###robot forworld kinematics
		#q_all:			robot joint angles or list of robot joint angles
		#world:			bool, if want to get coordinate in world frame or robot base frame

		if q_all.ndim==1:
			q=q_all
			pose_temp=fwdkin(self.robot,q)

			if world:
				pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
				pose_temp.R=self.base_H[:3,:3]@pose_temp.R
			return pose_temp
		else:
			pose_p_all=[]
			pose_R_all=[]
			for q in q_all:
				pose_temp=fwdkin(self.robot,q)
				if world:
					pose_temp.p=self.base_H[:3,:3]@pose_temp.p+self.base_H[:3,-1]
					pose_temp.R=self.base_H[:3,:3]@pose_temp.R

				pose_p_all.append(pose_temp.p)
				pose_R_all.append(pose_temp.R)

			return Transform_all(pose_p_all,pose_R_all)
	
	def jacobian(self,q):
		return robotjacobian(self.robot_def,q)

	def inv(self,p,R=np.eye(3),last_joints=None):
		pose=Transform(R,p)
		q_all=robot6_sphericalwrist_invkin(self.robot,pose,last_joints)
		
		return q_all


class Transform_all(object):
	def __init__(self, p_all, R_all):
		self.R_all=np.array(R_all)
		self.p_all=np.array(p_all)



def HomogTrans(q,h,p,jt):

	if jt==0:
		H=np.vstack((np.hstack((rot(h,q), p.reshape((3,1)))),np.array([0, 0, 0, 1,])))
	else:
		H=np.vstack((np.hstack((np.eye(3), p + np.dot(q, h))),np.array([0, 0, 0, 1,])))
	return H
def Hvec(h,jtype):

	if jtype>0:
		H=np.vstack((np.zeros((3,1)),h))
	else:
		H=np.vstack((h.reshape((3,1)),np.zeros((3,1))))
	return H
def phi(R,p):

	Phi=np.vstack((np.hstack((R,np.zeros((3,3)))),np.hstack((-np.dot(R,hat(p)),R))))
	return Phi


def jdot(q,qdot):
	zv=np.zeros((3,1))
	H=np.eye(4)
	J=[]
	Jdot=[]
	n=6
	Jmat=[]
	Jdotmat=[]
	for i in range(n+1):
		if i<n:
			hi=self.robot_def.H[:,i]
			qi=q[i]
			qdi=qdot[i]
			ji=self.robot_def.joint_type[i]

		else:
			qi=0
			qdi=0
			di=0
			ji=0

		Pi=self.robot_def.P[:,i]
		Hi=HomogTrans(qi,hi,Pi,ji)
		Hn=np.dot(H,Hi)
		H=Hn

		PHI=phi(Hi[:3,:3].T,Hi[:3,-1])
		Hveci=Hvec(hi,ji)
		###Partial Jacobian progagation
		if(len(J)>0):
			Jn=np.hstack((np.dot(PHI,J), Hveci))
			temp=np.vstack((np.hstack((hat(hi), np.zeros((3,3)))),np.hstack((np.zeros((3,3)),hat(hi)))))
			Jdotn=-np.dot(qdi,np.dot(temp,Jn)) + np.dot(PHI,np.hstack((Jdot, np.zeros(Hveci.shape))))
		else:
			Jn=Hveci
			Jdotn=np.zeros(Jn.shape)

		Jmat.append(Jn) 
		Jdotmat.append(Jdotn)
		J=Jn
		Jdot=Jdotn

	Jmat[-1]=Jmat[-1][:,:n]
	Jdotmat[-1]=Jdotmat[-1][:,:n]
	return Jdotmat[-1]


def main1():
	###robot object class
	robot_name='MA_1440_A0'
	robot=robot_obj(robot_name,def_path='../config/'+robot_name+'_robot_default_config.yml',tool_file_path='../config/scanner_tcp.csv')
	
	pulse2deg_1440=np.array([1.435355447016790322e+03,1.300329111270902331e+03,1.422225409601069941e+03,9.699560942607320158e+02,9.802408285708806943e+02,4.547552630640436178e+02])
	pulse2deg_2010=np.array([1.341416193724337745e+03,1.907685083229250267e+03,1.592916090846681982e+03,1.022871664227330484e+03,9.802549195016306385e+02,4.547554799861444508e+02])
	pulse2deg=pulse2deg_1440
	q_pulse=np.array([-26967,20050,-65667,-58160,-89688,-36278])
	q=np.radians(q_pulse/pulse2deg)
	print(q)
	# q=np.radians([-70.11,41.39,44.30,27.01,28.79,0])
	pose=robot.fwd(q)
	print(pose)
	print(np.degrees(rotationMatrixToEulerAngles(pose.R)))

	print(robot.inv(pose.p,pose.R,q))

def main2():
	robot=robot_obj('MA_1440_A0',def_path='../config/MA_1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp.csv',\
	pulse2deg_file_path='../config/MA_1440_A0_pulse2deg.csv')

	pose=robot.fwd(np.ones(6))
	print(pose)
	print(robot.inv(pose.p,pose.R))
if __name__ == '__main__':
	main1()