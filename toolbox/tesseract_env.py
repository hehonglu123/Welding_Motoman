#!/usr/bin/env python3
from tesseract_robotics.tesseract_environment import Environment, ChangeJointOriginCommand
from tesseract_robotics import tesseract_geometry
from tesseract_robotics.tesseract_common import Isometry3d, CollisionMarginData, Translation3d, Quaterniond, \
	ManipulatorInfo
from tesseract_robotics import tesseract_collision
from tesseract_robotics_viewer import TesseractViewer

import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s
import yaml, time, traceback, threading, sys, json
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import fminbound

from gazebo_model_resource_locator import GazeboModelResourceLocator
from robot_def import *

#convert 4x4 H matrix to 3x3 H matrix and inverse for mapping obj to robot frame
def H42H3(H):
	H3=np.linalg.inv(H[:2,:2])
	H3=np.hstack((H3,-np.dot(H3,np.array([[H[0][-1]],[H[1][-1]]]))))
	H3=np.vstack((H3,np.array([0,0,1])))
	return H3

class Tess_Env(object):
	def __init__(self,urdf_path):

		#link and joint names in urdf
		MA2010_link_names=["MA2010_base_link","MA2010_link_1_s","MA2010_link_2_l","MA2010_link_3_u","MA2010_link_4_r","MA2010_link_5_b","MA2010_link_6_t"]
		MA2010_joint_names=["MA2010_joint_1_s","MA2010_joint_2_l","MA2010_joint_3_u","MA2010_joint_4_r","MA2010_joint_5_b","MA2010_joint_6_t"]

		D500B_joint_names=["D500B_joint_1","D500B_joint_2"]
		D500B_link_names=["D500B_base_link","D500B_link_1","D500B_link_2"]
		#Robot dictionaries, all reference by name
		self.robot_linkname={'MA2010_A0':MA2010_link_names,'D500B':D500B_link_names}
		self.robot_jointname={'MA2010_A0':MA2010_joint_names,'D500B':D500B_joint_names}
		

		######tesseract environment setup:
		with open(urdf_path+'.urdf','r') as f:
			combined_urdf = f.read()
		with open(urdf_path+'.srdf','r') as f:
			combined_srdf = f.read()

		self.t_env= Environment()
		self.t_env.init(combined_urdf, combined_srdf, GazeboModelResourceLocator())
		self.scene_graph=self.t_env.getSceneGraph()


		#Tesseract reports all GJK/EPA distance within contact_distance threshold
		contact_distance=0.1
		monitored_link_names = self.t_env.getLinkNames()
		self.manager = self.t_env.getDiscreteContactManager()
		self.manager.setActiveCollisionObjects(monitored_link_names)
		self.manager.setCollisionMarginData(CollisionMarginData(contact_distance))


		#######viewer setup, for URDF setup verification in browser @ localhost:8000/#########################
		self.viewer = TesseractViewer()

		self.viewer.update_environment(self.t_env, [0,0,0])

		self.viewer.start_serve_background()

	def update_pose(self,model_name,H):
		###update model pose in tesseract environment
		cmd = ChangeJointOriginCommand(model_name+'_pose', Isometry3d(H))
		self.t_env.applyCommand(cmd)
		#refresh
		self.viewer.update_environment(self.t_env)

	def check_collision_single(self,robot_name,part_name,curve_js):
		###check collision for a single robot, including self collision and collision with part

		###iterate all joints config 

		for q in curve_js:
			self.t_env.setState(self.robot_jointname[robot_name], q)
			env_state = self.t_env.getState()
			self.manager.setCollisionObjectsTransform(env_state.link_transforms)

			result = tesseract_collision.ContactResultMap()
			contacts = self.manager.contactTest(result,tesseract_collision.ContactRequest(tesseract_collision.ContactTestType_ALL))
			result_vector = tesseract_collision.ContactResultVector()
			tesseract_collision.flattenResults(result,result_vector)
			###iterate all collision instances
			for c in result_vector:
				cond1=(robot_name in c.link_names[0]) and (robot_name in c.link_names[1])	#self collision
				cond2=(robot_name in c.link_names[0]) and (part_name in c.link_names[1])	#part collision
				cond3=(part_name in c.link_names[0]) and (robot_name in c.link_names[1])	#part collision
				collision_cond=c.distance<0 #actual collision

				if (cond1 or cond2 or cond3) and collision_cond:
					# print(c.link_names[0],c.link_names[1],c.distance)
					return True

		return False

	#######################################update joint angles in Tesseract Viewer###########################################
	def viewer_joints_update(self,robot_name,joints):
		self.viewer.update_joint_positions(self.robot_jointname[robot_name], np.array(joints))

	def viewer_trajectory(self,robot_name,curve_js):
		trajectory_json = dict()
		trajectory_json["use_time"] = True
		trajectory_json["loop_time"] = 20
		trajectory_json["joint_names"] = self.robot_jointname[robot_name]
		trajectory2 = np.hstack((curve_js,np.linspace(0,10,num=len(curve_js))[np.newaxis].T))
		trajectory_json["trajectory"] = trajectory2.tolist()
		self.viewer.trajectory_json=json.dumps(trajectory_json)

	def viewer_trajectory_dual(self,robot_name1,robot_name2,curve_js1,curve_js2):
		trajectory_json = dict()
		trajectory_json["use_time"] = True
		trajectory_json["loop_time"] = 20
		trajectory_json["joint_names"] = self.robot_jointname[robot_name1]+self.robot_jointname[robot_name2]
		trajectory2 = np.hstack((curve_js1,curve_js2,np.linspace(0,10,num=len(curve_js1))[np.newaxis].T))
		trajectory_json["trajectory"] = trajectory2.tolist()
		self.viewer.trajectory_json=json.dumps(trajectory_json)


def main():


	t=Tess_Env('../config/urdf/combined')				#create obj
	
	input("Press enter to quit")
	#stop background checker

def collision_test():
	t=Tess_Env('../config/urdf/combined')				#create obj
	###place part in place
	curve_pose=np.loadtxt('data/wood/baseline/curve_pose.csv',delimiter=',')
	curve_pose[:3,-1]=curve_pose[:3,-1]/1000.
	t.update_pose('curve_1',curve_pose)

	q=np.array([0,0.5,0.7,1,1,1.])
	t.viewer_joints_update('ABB_6640_180_255',q)
	curve_js=[q]
	print(t.check_collision_single('ABB_6640_180_255','curve_1',curve_js))

	input("Press enter to quit")

if __name__ == '__main__':
	main()











