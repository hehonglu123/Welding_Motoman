import sys, time, pickle, os, glob
from motoman_def import *
from lambda_calc import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *

def main():

	dataset='tube/'
	sliced_alg='dense_slice/'
	data_dir='../../../geometry_data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	waypoint_distance=5 	###waypoint separation

	##############################################################SENSORS####################################################################
	# weld state logging
	# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# cam_ser=None
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
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
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

	
	flipped=False   #spiral direction
	
	base_feedrate=300
	VPD=10
	v_layer=14
	layer_feedrate=VPD*v_layer
	base_layer_height=3
	v_base=5
	layer_height=1.3
	num_base_layer=2        #2 base layer to establish adhesion to coupon
	num_support_layer=12     #support layer to raise the cylinder till visible by IR camera
	support_layer_height=1.5
	support_feedrate=150
	v_support=10
	weld_num_layer=20

	weld_arcon=True
	job_offset=450


	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	base_slice_increment=int(base_layer_height/slicing_meta['line_resolution'])
	support_slice_increment=int(support_layer_height/slicing_meta['line_resolution'])


	

	###jog rob2 & positioner to start point
	client=MotionProgramExecClient()
	ws=WeldSend(client)
	q_prev=client.getJointAnglesDB(positioner.pulse2deg)
	num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
	positioner_js+=num2p*2*np.pi
	ws.jog_dual(robot2,positioner,q2,positioner_js[0],v=1)


	#####################################################BASE & SUPPORT LAYER##########################################################################################
	slice_start=0
	slice_end=int(base_slice_increment*num_base_layer)
	primitives_base,q1_all_base,q2_all_base,v1_all_base,v2_all_base,cond_all_base = weld_spiral(robot,positioner,data_dir,v_base,base_feedrate,base_slice_increment,num_base_layer,slice_start,slice_end,job_offset,waypoint_distance,flipped,q_prev)

	q_prev=q2_all_base[-1]
	slice_start=int(num_base_layer*base_slice_increment)
	slice_end=int(num_base_layer*base_slice_increment+num_support_layer*support_slice_increment)
	primitives_support,q1_all_support,q2_all_support,v1_all_support,v2_all_support,cond_all_support = weld_spiral(robot,positioner,data_dir,v_support,support_feedrate,support_slice_increment,num_support_layer,slice_start,slice_end,job_offset,waypoint_distance,flipped,q_prev)
	
	ws.weld_segment_dual(primitives_base+primitives_support[1:],robot,positioner,q1_all_base+q1_all_support[1:],q2_all_base+q2_all_support[1:],v1_all_base+v1_all_support[1:],v2_all_base+v2_all_support[1:],cond_all_base+cond_all_support[1:],arc=weld_arcon)
	print("BASE & SUPPORT LAYER FINISHED")
	

	#####################################################LAYER Welding##########################################################################################
	q_prev=client.getJointAnglesDB(positioner.pulse2deg)
	slice_start=int(num_base_layer*base_slice_increment+num_support_layer*support_slice_increment)
	slice_end=slicing_meta['num_layers']
	primitives,q1_all,q2_all,v1_all,v2_all,cond_all = weld_spiral(robot,positioner,data_dir,v_layer,layer_feedrate,nominal_slice_increment,weld_num_layer,slice_start,slice_end,job_offset,waypoint_distance,flipped,q_prev)

	##############################################################Log & Execution####################################################################
	rr_sensors.start_all_sensors()
	global_ts,robot_ts,joint_recording,job_line,_ = ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all,arc=weld_arcon)
	rr_sensors.stop_all_sensors()


	recorded_dir='../../../recorded_data/ER316L/VPD%i/tubespiral_%iipm_v%i/'%(VPD,layer_feedrate,v_layer)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.hstack((global_ts.reshape(-1,1),robot_ts.reshape(-1,1),job_line.reshape(-1,1),joint_recording)),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)


if __name__ == '__main__':
	main()