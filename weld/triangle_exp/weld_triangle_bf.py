import sys, time, pickle, os
sys.path.append('../../toolbox/')
from robot_def import *
from WeldSend import *
sys.path.append('../../sensor_fusion/')
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
from weldRRSensor import *


def main():

	dataset='triangle/'
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
	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js0_0.csv',delimiter=',')
	p_positioner_home=np.mean([robot.fwd(rob1_js[0]).p,robot.fwd(rob1_js[-1]).p],axis=0)
	p_robot2_proj=p_positioner_home+np.array([0,0,50])
	p2_in_base_frame=np.dot(H2010_1440[:3,:3],p_robot2_proj)+H2010_1440[:3,3]
	v_z=H2010_1440[:3,:3]@np.array([0,-0.96592582628,-0.2588190451]) ###pointing toward positioner's X with 15deg tiltd angle looking down
	v_y=VectorPlaneProjection(np.array([-1,0,0]),v_z)	###FLIR's Y pointing toward 1440's -X in 1440's base frame, projected on v_z's plane
	v_x=np.cross(v_y,v_z)
	p2_in_base_frame=p2_in_base_frame-measure_distance*v_z			###back project measure_distance-mm away from torch
	R2=np.vstack((v_x,v_y,v_z)).T
	q2=robot2.inv(p2_in_base_frame,R2,last_joints=np.zeros(6))[0]


	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	
	base_feedrate=300
	volume_per_distance=10
	v_layer=10
	feedrate=volume_per_distance*v_layer
	base_layer_height=3
	v_base=5
	layer_height=0.9
	num_layer=30
	#edge params, 1cm left and right
	feedrate_edge=feedrate
	v_edge=v_layer
	q_all=[]
	v_all=[]
	job_offset=450
	cond_all=[]
	primitives=[]

	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	base_slice_increment=int(base_layer_height/slicing_meta['line_resolution'])

	###jog to start point
	client=MotionProgramExecClient()
	ws=WeldSend(client)
	ws.jog_dual(robot2,positioner,q2,positioner_js[0],v=1)

	#########################################################BASELAYER Welding#########################################################
	# slice_num=0
	# for layer_num in range(2):
	# 	x=0
	# 	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
	# 	if layer_num%2==0:
	# 		rob1_js=np.flip(rob1_js,axis=0)


	# 	q_all.extend([rob1_js[0],rob1_js[-1]])
	# 	v_all.extend([1,v_base])
	# 	primitives.extend(['movej','movel'])
	# 	cond_all.extend([0,int(base_feedrate/10+job_offset)])
		
	# 	###adjust slice_num
	# 	slice_num+=base_slice_increment

	# ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.,blocking=True)
	

	####################################Normal Layer Welding####################################
	q_all=[]
	v_all=[]
	cond_all=[]
	primitives=[]
	slice_num=2*base_slice_increment
	layer_num=2
	while slice_num<slicing_meta['num_layers']:

		###############DETERMINE SECTION ORDER###########################
		sections=[0]
		####################DETERMINE CURVE ORDER##############################################
		for x in sections:
			rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			if layer_num%2==0:
				rob1_js=np.flip(rob1_js,axis=0)

			if layer_num==2:	#if start of first normal layer
				q_all.extend([rob1_js[0]])
				v_all.extend([1])
				primitives.extend(['movej'])
				cond_all.extend([0])

			q_all.extend([rob1_js[-1]])
			v_all.extend([v_layer])
			primitives.extend(['movel'])
			cond_all.extend([int(feedrate_edge/10+job_offset)])

		slice_num+=int(nominal_slice_increment)
		layer_num+=1
	
	
	##############################################################Log & Execution####################################################################
	rr_sensors.start_all_sensors()
	global_ts,robot_ts,joint_recording,job_line,_ = ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True,wait=0.)
	rr_sensors.stop_all_sensors()


	recorded_dir='../../../recorded_data/ER316L/trianglebf_%iipm_v%i_%iipm_v%i/'%(feedrate,v_layer,feedrate_edge,v_edge)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.hstack((global_ts,robot_ts,job_line,joint_recording)),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)


if __name__ == '__main__':
	main()