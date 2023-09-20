from general_robotics_toolbox import * 
from general_robotics_toolbox import tesseract as rox_tesseract
from general_robotics_toolbox import robotraconteur as rr_rox

import numpy as np
import yaml, copy, time
import pickle
from utils import *


ex=np.array([[1.],[0.],[0.]])
ey=np.array([[0.],[1.],[0.]])
ez=np.array([[0.],[0.],[1.]])


class robot_obj(object):
	###robot object class
	def __init__(self,robot_name,def_path,tool_file_path='',base_transformation_file='',d=0,acc_dict_path='',pulse2deg_file_path='',
				base_marker_config_file='',tool_marker_config_file=''):
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
			self.pulse2deg=np.abs(np.loadtxt(pulse2deg_file_path,delimiter=',')) #negate joint 2, 4, 6


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
		
		### load mocap marker config
		self.base_marker_config_file=base_marker_config_file
		self.T_base_basemarker = None # T^base_basemaker
		self.T_base_mocap = None # T^base_mocap
		self.T_tool_flange = None
		self.calib_zero_config=np.zeros(self.robot.H.shape[1])
		if len(base_marker_config_file)>0:
			with open(base_marker_config_file,'r') as file:
				marker_data = yaml.safe_load(file)
				self.base_markers_id = marker_data['base_markers']
				self.base_rigid_id = self.base_markers_id[0].split('_')[1]
				self.calib_markers_id = marker_data['calibration_markers']
				if 'calib_base_basemarker_pose' in marker_data.keys():
					p = [marker_data['calib_base_basemarker_pose']['position']['x'],
						marker_data['calib_base_basemarker_pose']['position']['y'],
						marker_data['calib_base_basemarker_pose']['position']['z']]
					q = [marker_data['calib_base_basemarker_pose']['orientation']['w'],
						marker_data['calib_base_basemarker_pose']['orientation']['x'],
						marker_data['calib_base_basemarker_pose']['orientation']['y'],
						marker_data['calib_base_basemarker_pose']['orientation']['z']]
					self.T_base_basemarker = Transform(q2R(q),p)
				if 'calib_base_mocap_pose' in marker_data.keys():
					p = [marker_data['calib_base_mocap_pose']['position']['x'],
						marker_data['calib_base_mocap_pose']['position']['y'],
						marker_data['calib_base_mocap_pose']['position']['z']]
					q = [marker_data['calib_base_mocap_pose']['orientation']['w'],
						marker_data['calib_base_mocap_pose']['orientation']['x'],
						marker_data['calib_base_mocap_pose']['orientation']['y'],
						marker_data['calib_base_mocap_pose']['orientation']['z']]
					self.T_base_mocap = Transform(q2R(q),p)
				if 'calib_tool_flange_pose' in marker_data.keys():
					p = [marker_data['calib_tool_flange_pose']['position']['x'],
						marker_data['calib_tool_flange_pose']['position']['y'],
						marker_data['calib_tool_flange_pose']['position']['z']]
					q = [marker_data['calib_tool_flange_pose']['orientation']['w'],
						marker_data['calib_tool_flange_pose']['orientation']['x'],
						marker_data['calib_tool_flange_pose']['orientation']['y'],
						marker_data['calib_tool_flange_pose']['orientation']['z']]
					self.T_tool_flange = Transform(q2R(q),p)
				if 'P' in marker_data.keys():
					self.calib_P = np.zeros(self.robot.P.shape)
					for i in range(len(marker_data['P'])):
						self.calib_P[0,i] = marker_data['P'][i]['x']
						self.calib_P[1,i] = marker_data['P'][i]['y']
						self.calib_P[2,i] = marker_data['P'][i]['z']
				if 'H' in marker_data.keys():
					self.calib_H = np.zeros(self.robot.H.shape)
					for i in range(len(marker_data['H'])):
						self.calib_H[0,i] = marker_data['H'][i]['x']
						self.calib_H[1,i] = marker_data['H'][i]['y']
						self.calib_H[2,i] = marker_data['H'][i]['z']
				# if 'zero_config' in marker_data.keys():
				# 	self.calib_zero_config = np.array(marker_data['zero_config'])
				# 	self.robot.joint_upper_limit = self.robot.joint_upper_limit-self.calib_zero_config
				# 	self.robot.joint_lower_limit = self.robot.joint_lower_limit-self.calib_zero_config
		self.tool_marker_config_file=tool_marker_config_file
		self.T_tool_toolmarker = None # T^tool_toolmarker
		if len(tool_marker_config_file)>0:
			with open(tool_marker_config_file,'r') as file:
				marker_data = yaml.safe_load(file)
				self.tool_markers = marker_data['tool_markers']
				self.tool_markers_id = list(self.tool_markers.keys())
				self.tool_rigid_id = self.tool_markers_id[0].split('_')[1]
				if 'calib_tool_toolmarker_pose' in marker_data.keys():
					p = [marker_data['calib_tool_toolmarker_pose']['position']['x'],
						marker_data['calib_tool_toolmarker_pose']['position']['y'],
						marker_data['calib_tool_toolmarker_pose']['position']['z']]
					q = [marker_data['calib_tool_toolmarker_pose']['orientation']['w'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['x'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['y'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['z']]
					self.T_tool_toolmarker = Transform(q2R(q),p)
					# add d
					T_d1_d2 = Transform(np.eye(3),p=[0,0,d-15])
					self.T_tool_toolmarker = self.T_tool_toolmarker*T_d1_d2
				if 'calib_toolmarker_flange_pose' in marker_data.keys():
					p = [marker_data['calib_toolmarker_flange_pose']['position']['x'],
						marker_data['calib_toolmarker_flange_pose']['position']['y'],
						marker_data['calib_toolmarker_flange_pose']['position']['z']]
					q = [marker_data['calib_toolmarker_flange_pose']['orientation']['w'],
						marker_data['calib_toolmarker_flange_pose']['orientation']['x'],
						marker_data['calib_toolmarker_flange_pose']['orientation']['y'],
						marker_data['calib_toolmarker_flange_pose']['orientation']['z']]
					self.T_toolmarker_flange = Transform(q2R(q),p)

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
			# q = np.array(q)-self.calib_zero_config
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
		return robotjacobian(self.robot,q)

	def fwd_ph(self,q,ph_param):
     
		q=np.array(q)
    
		origin_P=copy.deepcopy(self.robot.P)
		origin_H=copy.deepcopy(self.robot.H)
    
		opt_P,opt_H = ph_param.predict(q[1:3])
		self.robot.P=copy.deepcopy(opt_P)
		self.robot.H=copy.deepcopy(opt_H)
		robot_T = fwdkin(self.robot,q)
  
		self.robot.P=copy.deepcopy(origin_P)
		self.robot.H=copy.deepcopy(origin_H)
  
		return robot_T

	def jacobian_ph(self,q,ph_param):
     
		q=np.array(q)
    
		origin_P=copy.deepcopy(self.robot.P)
		origin_H=copy.deepcopy(self.robot.H)
    
		opt_P,opt_H = ph_param.predict(q[1:3])
		self.robot.P=opt_P
		self.robot.H=opt_H
		J = robotjacobian(self.robot,q)
  
		self.robot.P=origin_P
		self.robot.H=origin_H
  
		return J

	def inv(self,p,R=np.eye(3),last_joints=None):
		pose=Transform(R,p)
		q_all=robot6_sphericalwrist_invkin(self.robot,pose,last_joints)
		
		return q_all

	###find a continous trajectory given Cartesion pose trajectory
	def find_curve_js(self,curve,curve_R,q_seed=None):
		q_inits=self.inv(curve[0],curve_R[0])
		curve_js_all=[]
		for q_init in q_inits:
			curve_js=np.zeros((len(curve),6))
			curve_js[0]=q_init
			for i in range(1,len(curve)):
				q_all=np.array(self.inv(curve[i],curve_R[i]))
				if len(q_all)==0:
					#if no solution
					print('no solution available')
					return

				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				if np.linalg.norm(q_all[order[0]]-curve_js[i-1])>0.5:
					break	#if large changes in q, not continuous
				else:
					curve_js[i]=q_all[order[0]]

			#check if all q found
			if i==len(curve)-1:
				curve_js_all.append(curve_js)
		
		if len(curve_js_all)==0:
			raise Exception('No Solution Found') 
		
		if q_seed is None:
			return curve_js_all
		else:
			if len(curve_js_all)==1:
				return curve_js_all[0]
			else:
				diff_min=[]
				for curve_js in curve_js_all:
					diff_min.append(np.linalg.norm(curve_js[0]-q_seed))

				return curve_js_all[np.argmin(diff_min)]
			


class positioner_obj(object):
	###robot object class
	def __init__(self,robot_name,def_path,tool_file_path='',base_transformation_file='',d=0,acc_dict_path='',pulse2deg_file_path='',\
				base_marker_config_file='',tool_marker_config_file=''):
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
			self.pulse2deg=np.abs(np.loadtxt(pulse2deg_file_path,delimiter=',')) #negate joint 2, 4, 6


		###set attributes
		self.upper_limit=self.robot.joint_upper_limit 
		self.lower_limit=self.robot.joint_lower_limit 
		self.joint_vel_limit=self.robot.joint_vel_limit 
		self.joint_acc_limit=self.robot.joint_acc_limit

		### load mocap marker config
		self.base_marker_config_file=base_marker_config_file
		self.T_base_basemarker = None # T^base_basemaker
		self.T_base_mocap = None # T^base_mocap
		if len(base_marker_config_file)>0:
			with open(base_marker_config_file,'r') as file:
				marker_data = yaml.safe_load(file)
				self.base_markers_id = marker_data['base_markers']
				self.base_rigid_id = self.base_markers_id[0].split('_')[1]
				self.calib_markers_id = marker_data['calibration_markers']
				if 'calib_base_basemarker_pose' in marker_data.keys():
					p = [marker_data['calib_base_basemarker_pose']['position']['x'],
						marker_data['calib_base_basemarker_pose']['position']['y'],
						marker_data['calib_base_basemarker_pose']['position']['z']]
					q = [marker_data['calib_base_basemarker_pose']['orientation']['w'],
						marker_data['calib_base_basemarker_pose']['orientation']['x'],
						marker_data['calib_base_basemarker_pose']['orientation']['y'],
						marker_data['calib_base_basemarker_pose']['orientation']['z']]
					self.T_base_basemarker = Transform(q2R(q),p)
				if 'calib_base_mocap_pose' in marker_data.keys():
					p = [marker_data['calib_base_mocap_pose']['position']['x'],
						marker_data['calib_base_mocap_pose']['position']['y'],
						marker_data['calib_base_mocap_pose']['position']['z']]
					q = [marker_data['calib_base_mocap_pose']['orientation']['w'],
						marker_data['calib_base_mocap_pose']['orientation']['x'],
						marker_data['calib_base_mocap_pose']['orientation']['y'],
						marker_data['calib_base_mocap_pose']['orientation']['z']]
					self.T_base_mocap = Transform(q2R(q),p)
				if 'calib_tool_flange_pose' in marker_data.keys():
					p = [marker_data['calib_tool_flange_pose']['position']['x'],
						marker_data['calib_tool_flange_pose']['position']['y'],
						marker_data['calib_tool_flange_pose']['position']['z']]
					q = [marker_data['calib_tool_flange_pose']['orientation']['w'],
						marker_data['calib_tool_flange_pose']['orientation']['x'],
						marker_data['calib_tool_flange_pose']['orientation']['y'],
						marker_data['calib_tool_flange_pose']['orientation']['z']]
					self.T_tool_flange = Transform(q2R(q),p)
				if 'P' in marker_data.keys():
					self.calib_P = np.zeros(self.robot.P.shape)
					for i in range(len(marker_data['P'])):
						self.calib_P[0,i] = marker_data['P'][i]['x']
						self.calib_P[1,i] = marker_data['P'][i]['y']
						self.calib_P[2,i] = marker_data['P'][i]['z']
				if 'H' in marker_data.keys():
					self.calib_H = np.zeros(self.robot.H.shape)
					for i in range(len(marker_data['H'])):
						self.calib_H[0,i] = marker_data['H'][i]['x']
						self.calib_H[1,i] = marker_data['H'][i]['y']
						self.calib_H[2,i] = marker_data['H'][i]['z']
				self.calib_zero_config=np.zeros(self.robot.H.shape[1])
				# if 'zero_config' in marker_data.keys():
				# 	self.calib_zero_config = np.array(marker_data['zero_config'])
				# 	self.robot.joint_upper_limit = self.robot.joint_upper_limit-self.calib_zero_config
				# 	self.robot.joint_lower_limit = self.robot.joint_lower_limit-self.calib_zero_config
		self.tool_marker_config_file=tool_marker_config_file
		self.T_tool_toolmarker = None # T^tool_toolmarker
		if len(tool_marker_config_file)>0:
			with open(tool_marker_config_file,'r') as file:
				marker_data = yaml.safe_load(file)
				self.tool_markers = marker_data['tool_markers']
				self.tool_markers_id = list(self.tool_markers.keys())
				self.tool_rigid_id = self.tool_markers_id[0].split('_')[1]
				if 'calib_tool_toolmarker_pose' in marker_data.keys():
					p = [marker_data['calib_tool_toolmarker_pose']['position']['x'],
						marker_data['calib_tool_toolmarker_pose']['position']['y'],
						marker_data['calib_tool_toolmarker_pose']['position']['z']]
					q = [marker_data['calib_tool_toolmarker_pose']['orientation']['w'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['x'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['y'],
						marker_data['calib_tool_toolmarker_pose']['orientation']['z']]
					self.T_tool_toolmarker = Transform(q2R(q),p)
					# add d
					T_d1_d2 = Transform(np.eye(3),p=[0,0,d])
					self.T_tool_toolmarker = self.T_tool_toolmarker*T_d1_d2

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

	def fwd_rotation(self,q_all):
		return Ry(-q_all[0])@Rz((-q_all[1]))

	# def inv(self,n,q_seed=np.zeros(2)):
	# 	###p_tcp: point of the desired tcp wrt. positioner eef
	# 	###n: normal direction at the desired tcp wrt. positioner eef
	# 	###[q1,q2]: result joint angles that brings n to [0,0,1]
	# 	if n[2]==1:	##if already up, infinite solutions
	# 		return np.array([0,q_seed[1]])
	# 	q2=np.arctan2(-n[1],n[0])
	# 	q1=np.arctan2(-n[0]*np.cos(q2)+n[1]*np.sin(q2),n[2])
	# 	# q1=np.arcsin(1/(-n[0]*np.cos(q2)+n[1]*np.sin(q2)+n[2]))

	# 	return np.array([-q1,-q2])		###2 solutions, 180 apart couple

	def inv(self,n,q_seed=None):
		###p_tcp: point of the desired tcp wrt. positioner eef
		###n: normal direction at the desired tcp wrt. positioner eef
		###[q1,q2]: result joint angles that brings n to [0,0,1]
		if n[2]==1:	##if already up, infinite solutions
			if q_seed is not None:
				return np.array([0-np.radians(15),q_seed[1]])
			else:
				return [np.array([0-np.radians(15),0])]
		q2=np.arctan2(n[1],n[0])
		q1=np.arctan2(n[0]*np.cos(q2)+n[1]*np.sin(q2),n[2])

		solutions = self.get_eq_solution([q1,q2])
		solutions[:,0]=solutions[:,0]-np.radians(15)		###manual adjustment for tilt angle

		if q_seed is not None:
			theta_dist = np.linalg.norm(np.subtract(solutions,q_seed), axis=1)
			return solutions[np.argsort(theta_dist)[0]]
		else:
			return solutions
	def get_eq_solution(self,q):
		###inifinite solution stack
		# solutions=np.vstack([np.linspace([q[0],q[1]-100*2*np.pi],[q[0],q[1]+100*2*np.pi],num=201),\
		# 					np.linspace([-q[0],q[1]-np.pi-100*2*np.pi],[-q[0],q[1]-np.pi+100*2*np.pi],num=201)])

		###extended 4pi solution stack
		# solutions=np.vstack([np.linspace([q[0],q[1]-2*2*np.pi],[q[0],q[1]+2*2*np.pi],num=5),\
		# 					np.linspace([-q[0],q[1]-np.pi-2*2*np.pi],[-q[0],q[1]-np.pi+2*2*np.pi],num=5)])

		###regular solution stack
		solutions=[q]
		if q[1]+2*np.pi<self.upper_limit[1]:
			solutions.append([q[0],q[1]+2*np.pi])
		if q[1]+np.pi<self.upper_limit[1]:
			solutions.append([-q[0],q[1]+np.pi])
		if q[1]-np.pi>self.lower_limit[1]:
			solutions.append([-q[0],q[1]-np.pi])
		if q[1]-2*np.pi>self.lower_limit[1]:
			solutions.append([q[0],q[1]-2*np.pi])
		
		return np.array(solutions)
	
	def jacobian(self,q):
		return robotjacobian(self.robot,q)

	###find a continous trajectory given Cartesion tool normal trajectory
	def find_curve_js(self,normals,q_seed=None):
		if normals[0][2]==1:# in case normal is already up, infinite solutions
			q_inits=[self.inv(normals[0],q_seed)]
		else:
			q_inits=self.inv(normals[0])
		curve_js_all=[]
		for q_init in q_inits:
			curve_js=np.zeros((len(normals),2))
			curve_js[0]=q_init
			for i in range(1,len(normals)):
				if normals[i][2]==1:# in case normal is already up, infinite solutions
					q_all=[np.array(self.inv(normals[i],curve_js[i-1]))]
				else:
					q_all=np.array(self.inv(normals[i]))
				if len(q_all)==0:
					#if no solution
					print('no solution available')
					return

				temp_q=q_all-curve_js[i-1]
				order=np.argsort(np.linalg.norm(temp_q,axis=1))
				if np.linalg.norm(q_all[order[0]]-curve_js[i-1])>np.pi/2:
					break	#if large changes in q, not continuous
				else:
					curve_js[i]=q_all[order[0]]

			#check if all q found
			if i==len(normals)-1:
				curve_js_all.append(curve_js)

		if len(curve_js_all)==0:
			raise Exception('No Solution Found') 
		
		if q_seed is None:
			return curve_js_all
		else:
			if len(curve_js_all)==1:
				return curve_js_all[0]
			else:
				curve_js_all=np.array(curve_js_all)

				diff_q0=np.abs(curve_js_all[:,0,0]-q_seed[0])
				diff_q0_min_ind=np.nonzero(diff_q0==np.min(diff_q0))[0]
				diff_q1=np.abs(curve_js_all[diff_q0_min_ind,0,1]-q_seed[1])
				diff_q1_min_ind=np.nonzero(diff_q1==np.min(diff_q1))[0]
				index=diff_q0_min_ind[diff_q1_min_ind[0]]
				return curve_js_all[index]

				# diff_min=[]
				# for curve_js in curve_js_all:
				# 	diff_min.append(np.linalg.norm(curve_js[0]-q_seed))
				# return curve_js_all[np.argmin(diff_min)]
			
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
	positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
		pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')
	
	print(positioner.fwd(np.zeros(2)))
	# q1=30
	# q2=50
	# q=np.radians([q1,q2])
	# print(robot.fwd(q))

	# # n=np.array([np.sqrt(2)/2,0,np.sqrt(2)/2])
	# n=np.array([np.sqrt(3)/3,np.sqrt(3)/3,np.sqrt(3)/3])
	# q_inv=robot.inv(n)
	# print(np.degrees(q_inv))
	# print(robot.fwd_rotation(q_inv))
	# print(robot.fwd(q_inv).R@n)

	# q_temp=-2
	# print(np.sin(q_temp))
	# print(-np.cos(q_temp))
	# print(robot.jacobian([q_temp,0]))

if __name__ == '__main__':
	main2()
