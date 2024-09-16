import sys, glob, pickle, os, traceback, wave, time
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../../toolbox/')
from robot_def import *
# from multi_robot import *
from lambda_calc import *
from flir_toolbox import *
from StreamingSend import *
sys.path.append('../../sensor_fusion/')
from weldRRSensor import *

def main():

	dataset='wall/'
	sliced_alg='dense_slice/'
	data_dir='../../data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)
	

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
	# p_positioner_home=positioner.fwd(q_positioner_home,world=True).p
	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js0_0.csv',delimiter=',')
	p_positioner_home=np.mean([robot.fwd(rob1_js[0]).p,robot.fwd(rob1_js[-1]).p],axis=0)
	p_robot2_proj=p_positioner_home+np.array([0,0,50])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]

	########################################################RR FRONIUS########################################################
	weld_arcon=True
	if weld_arcon:
		fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
		fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
		hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
		fronius_client.prepare_welder()
	
	########################################################RR STREAMING########################################################
	RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.1.114:59945?service=robot')
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	SS=StreamingSend(RR_robot_sub,streaming_rate=125.)

	


	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	base_feedrate=300
	feedrate=100
	v_layer=10
	base_layer_height=3
	v_base=5
	layer_height=1.1
	base_layer_height=2.
	#edge params, 1cm left and right
	edge_length=10
	feedrate_edge=100
	v_edge=10
	q_all=[]
	job_offset=100
	cond_all=[]

	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	base_slice_increment=int(base_layer_height/slicing_meta['line_resolution'])


	
	#########################################################BASELAYER find joint angles#########################################################
	slice_num=0
	for layer_num in range(2):
		x=0
		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		if layer_num%2==0:
			rob1_js=np.flip(rob1_js,axis=0)
			rob2_js=np.flip(rob2_js,axis=0)
			positioner_js=np.flip(positioner_js,axis=0)
			curve_sliced_relative=np.flip(curve_sliced_relative,axis=0)


		lam_relative=calc_lam_cs(curve_sliced_relative)

		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)

		breakpoints=SS.get_breakpoints(lam_relative_dense,v_base)


		
		###formulate streaming joint angles
		q_all.extend(np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints])))

		###adjust slice_num
		slice_num+=base_slice_increment

	q_all=np.array(q_all)[:,:6]

	###jog to start point
	print("BASELAYER CALCULATION FINISHED")
	SS.jog2q(np.hstack((q_all[0],q2,q_positioner_home)))
	##############################################################Base Layers Welding####################################################################
	if weld_arcon:
		fronius_client.job_number = int(base_feedrate/10+job_offset)
		fronius_client.start_weld()

	for i in range(len(q_all)):
		SS.position_cmd(np.hstack((q_all[i],q2,q_positioner_home)),time.perf_counter())
	
	if weld_arcon:
		fronius_client.stop_weld()
	
	print("BASELAYER WELDING FINISHED")
	####################################Normal Layer Joint Angles ####################################
	q_all=[]
	cond_all=[]
	cond_indices=[]
	
	slice_num=2*base_slice_increment
	layer_num=2
	while slice_num<slicing_meta['num_layers'] and layer_num<4:

		###############DETERMINE SECTION ORDER###########################
		sections=[0]
		####################DETERMINE CURVE ORDER##############################################
		for x in sections:
			rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			if layer_num%2==0:
				rob1_js=np.flip(rob1_js,axis=0)
				rob2_js=np.flip(rob2_js,axis=0)
				positioner_js=np.flip(positioner_js,axis=0)
				curve_sliced_relative=np.flip(curve_sliced_relative,axis=0)
			
			lam_relative=calc_lam_cs(curve_sliced_relative)

			lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
			#find index of edge at edge_length
			edge1_index=np.argmin(np.abs(lam_relative_dense-edge_length))
			edge2_index=np.argmin(np.abs(lam_relative_dense-(lam_relative[-1]-edge_length)))

			rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
			rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
			positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)

			breakpoints_edge1=SS.get_breakpoints(lam_relative_dense[:edge1_index],v_edge)
			breakpoints_edge2=SS.get_breakpoints(lam_relative_dense[edge2_index:],v_edge)
			breakpoints_layer=SS.get_breakpoints(lam_relative_dense[edge1_index:edge2_index],v_layer)
			breakpoints=np.hstack((breakpoints_edge1,breakpoints_layer+edge1_index,breakpoints_edge2+edge2_index))

		
		
		#welding job adjustment in the middle of the layer
		cond_indices.extend([len(q_all)+len(breakpoints_edge1),len(q_all)+len(breakpoints_edge1)+len(breakpoints_layer),len(q_all)+len(breakpoints)])
		cond_all.extend([int(feedrate_edge/10+job_offset)]*len(breakpoints_edge1))
		cond_all.extend([int(feedrate/10+job_offset)]*len(breakpoints_layer))
		cond_all.extend([int(feedrate_edge/10+job_offset)]*len(breakpoints_edge2))
		

		###formulate streaming joint angles
		q_all.extend(np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints])))

		slice_num+=int(nominal_slice_increment)
		layer_num+=1
	

	q_all=np.array(q_all)[:,:6]
	print("Layer CALCULATION FINISHED")

	###jog to start point
	SS.jog2q(np.hstack((q_all[0],q2,q_positioner_home)))

	##############################################################Welding Normal Layers ####################################################################

	rr_sensors.start_all_sensors()
	SS.start_recording()
	if weld_arcon:
		fronius_client.job_number = cond_all[0]
		cond_all.pop(0)
		cond_indices.pop(0)
		fronius_client.start_weld()
	for i in range(len(q_all)):
		SS.position_cmd(np.hstack((q_all[i],q2,q_positioner_home)),time.perf_counter())
		if i in cond_indices and weld_arcon:
			fronius_client.job_number = cond_all[cond_indices.index(i)]
		
	if weld_arcon:
		fronius_client.stop_weld()
	rr_sensors.stop_all_sensors()
	js_recording = SS.stop_recording()



	recorded_dir='../../../recorded_data/streaming/wallbf_%iipm_v%i_%iipm_v%i/'%(feedrate,v_layer,feedrate_edge,v_edge)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)

if __name__ == '__main__':
	main()