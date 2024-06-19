import sys, glob, pickle, os, traceback, wave, time
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from pathlib import Path
import numpy as np
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from flir_toolbox import *
from StreamingSend import *
sys.path.append('../../sensor_fusion/')
from weldRRSensor import *

def main():

	##############################################################SENSORS####################################################################
	# weld state logging
	# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
	## RR sensor objects
	rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')

	###define start pose for 3 robtos
	measure_distance=500
	H2010_1440=H_inv(robot2.base_H)
	q_positioner_home=np.array([-15.*np.pi/180.,np.pi/2])
	p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	p_robot2_proj=p_positioner_home+np.array([0,0,50])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

	########################################################RR FRONIUS########################################################
	fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
	fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
	hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
	fronius_client.prepare_welder()
	weld_arcon=False
	########################################################RR STREAMING########################################################
	RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
	RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
	RR_robot = RR_robot_sub.GetDefaultClientWait(1)
	robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
	halt_mode = robot_const["RobotCommandMode"]["halt"]
	position_mode = robot_const["RobotCommandMode"]["position_command"]
	RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
	RR_robot.reset_errors()
	RR_robot.enable()
	RR_robot.command_mode = halt_mode
	time.sleep(0.1)
	RR_robot.command_mode = position_mode
	streaming_rate=125.
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)

	


	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])
	#ystart -850, yend -775
	p_start_base=np.array([1710,-825,-260])
	p_end_base=np.array([1590,-825,-260])
	p_start=np.array([1700,-825,-260])
	p_end=np.array([1600,-825,-260])
	q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])


	base_feedrate=300
	feedrate=150
	v_layer=15
	base_layer_height=3
	v_base=5
	layer_height=1.1
	#edge params, 1cm left and right
	feedrate_edge=150
	v_edge=15
	p_all=[]
	job_offset=450
	cond_all=[]

	####################################Base Layer ####################################
	for i in range(0,2):
		if i%2==0:
			p1=p_start_base+np.array([0,0,i*base_layer_height])
			p2=p_end_base+np.array([0,0,i*base_layer_height])
		else:
			p1=p_end_base+np.array([0,0,i*base_layer_height])
			p2=p_start_base+np.array([0,0,i*base_layer_height])

		#interpolate between p1 and p2
		p_all.extend(np.linspace(p1,p2,int(streaming_rate*np.linalg.norm(p2-p1)/v_base)),endpoint=False)
	
	q_all = robot.find_curve_js(np.array(p_all),[R]*len(p_all),q_seed)
	
	###jog to start point
	SS.jog2q(np.hstack((q_all[0],q2,q_positioner_home)))
	##############################################################Base Layers ####################################################################
	fronius_client.job_number = int(base_feedrate/10+job_offset)
	if weld_arcon:
		fronius_client.start_weld()

	for i in range(len(q_all)):
		SS.position_cmd(np.hstack((q_all[i],q2,q_positioner_home)),time.time())
	
	fronius_client.stop_weld()
	

	####################################Normal Layer ####################################
	q_all=[]
	cond_all=[]
	cond_indices=[]
	
	for i in range(2,12):
		if i%2==0:
			p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		else:
			p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
			p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

		p_edge1=p1+(p2-p1)/np.linalg.norm(p2-p1)*10
		p_edge2=p2-(p2-p1)/np.linalg.norm(p2-p1)*10


		#interpolate between p1 and p2
		cond_indices.append(len(p_all))
		cond_all.extend([int(feedrate_edge/10+job_offset)]*len(p_all))
		p_all.extend(np.linspace(p1,p_edge1,int(streaming_rate*np.linalg.norm(p_edge1-p1)/v_edge)),endpoint=False)
		cond_indices.append(len(p_all))
		cond_all.extend([int(feedrate/10+job_offset)]*len(p_all))
		p_all.extend(np.linspace(p_edge1,p_edge2,int(streaming_rate*np.linalg.norm(p_edge2-p_edge1)/v_layer)),endpoint=False)
		cond_indices.append(len(p_all))
		cond_all.extend([int(feedrate_edge/10+job_offset)]*len(p_all))
		p_all.extend(np.linspace(p_edge2,p2,int(streaming_rate*np.linalg.norm(p2-p_edge2)/v_edge)),endpoint=False)
		

	

	###jog to start point
	SS.jog2q(np.hstack((q_all[0],q2,q_positioner_home)))

	##############################################################Welding Layers ####################################################################
	fronius_client.job_number = cond_all[0]
	cond_all.pop(0)
	cond_indices.pop(0)

	rr_sensors.start_all_sensors()
	SS.joint_logging_flag=True
	if weld_arcon:
		fronius_client.start_weld()
	for i in range(len(q_all)):
		SS.position_cmd(np.hstack((q_all[i],q2,q_positioner_home)),time.time())
		if i in cond_indices:
			fronius_client.job_number = cond_all[cond_indices.index(i)]
	fronius_client.stop_weld()
	rr_sensors.stop_all_sensors()
	js_recording = SS.stop_recording()



	recorded_dir='../../../recorded_data/wallbf_%iipm_v%i_%iipm_v%i/'%(feedrate,v_layer,feedrate_edge,v_edge)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)

if __name__ == '__main__':
	main()