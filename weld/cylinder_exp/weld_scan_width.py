import time, pickle, os
from motoman_def import *
from lambda_calc import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from redundancy_resolution_dual import redundancy_resolution_dual


def scan_path_gen(curve_sliced_relative,scan_stand_off_d):
	scan_R=[]
	scan_p=[]
	for i in range(len(curve_sliced_relative)):
		# get scan R
		Rz = curve_sliced_relative[i,3:]
		Rz = Rz/np.linalg.norm(Rz)
		if i<len(curve_sliced_relative)-1:
			Ry = curve_sliced_relative[i+1,:3]-curve_sliced_relative[i,:3]
		else:
			Ry = curve_sliced_relative[i,:3]-curve_sliced_relative[i-1,:3]
		Ry = (Ry-np.dot(Ry,Rz)*Rz)
		Ry=Ry/np.linalg.norm(Ry)

		Rx = np.cross(Ry,Rz)
		Rx = Rx/np.linalg.norm(Rx)
		scan_R.append(np.array([Rx,Ry,Rz]).T)
		scan_p.append(curve_sliced_relative[i,:3] - Rz*scan_stand_off_d)

	return np.array(scan_p),np.array(scan_R)

def weld_spiral(robot,positioner,data_dir,v,feedrate,slice_increment,num_layers,slice_start,slice_end,job_offset,waypoint_distance=5,flipped=False,q_prev=np.zeros(2)):
	q1_all=[]
	q2_all=[]
	v1_all=[]
	v2_all=[]
	cond_all=[]
	primitives=[]
	arcon_set=False
	###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	rob1_js_all_slices=[]
	positioner_js_all_slices=[]
	for i in range(0,slice_end):
		if not flipped:
			rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
			positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
		else:
			###spiral rotation direction
			rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
			positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))

	print("LAYERS PRELOAD FINISHED")

	slice_num=slice_start
	layer_counts=0
	while layer_counts<num_layers:

		####################DETERMINE CURVE ORDER##############################################
		x=0
		rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
		positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		if positioner_js.shape==(2,) and rob1_js.shape==(6,):
			continue
		
		###TRJAECTORY WARPING
		if slice_num>0:
			rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-slice_increment])
			positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
		if slice_num<slice_end-slice_increment:
			rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+slice_increment])
			positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)
				
		
			
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam1=calc_lam_js(rob1_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
		breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
		s1_all,s2_all=calc_individual_speed(v,lam1,lam2,lam_relative,breakpoints)
		# s1_all=[0.1]*len(s1_all)
		###find closest %2pi
		num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
		positioner_js+=num2p*2*np.pi
		###no need for acron/off when spiral, positioner not moving at all
		if not arcon_set:
			arcon_set=True
			q1_all.append(rob1_js[breakpoints[0]])
			q2_all.append(positioner_js[breakpoints[0]])
			v1_all.append(1)
			v2_all.append(1)
			cond_all.append(0)
			primitives.append('movej')
		

		q1_all.extend(rob1_js[breakpoints[1:]].tolist())
		q2_all.extend(positioner_js[breakpoints[1:]].tolist())
		v1_all.extend([1]*len(s1_all))
		cond_all.extend([int(feedrate/10)+job_offset]*(num_points_layer-1))
		primitives.extend(['movel']*(num_points_layer-1))

		for j in range(1,len(breakpoints)):
			positioner_w=v/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
		
		q_prev=copy.deepcopy(positioner_js[-1])
		layer_counts+=1
		slice_num+=slice_increment

	return primitives,q1_all,q2_all,v1_all,v2_all,cond_all

	
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
	robot2_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
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
	v_layer=20
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









	#################################################################################################################################################################################
	#################################################################################################################################################################################
	#################################################################################################################################################################################
	###########################################################ROBOT WIDTH SCAN######################################################################################################
	###########################################################Chose scanning Curve#######################################################
	num_layers=40
	layer_height=1.1
	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	slice_num=num_layers*nominal_slice_increment
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice%i_0.csv'%slice_num,delimiter=',')

	###########################################################Robot Control#######################################################
	client=MotionProgramExecClient()
	ws=WeldSend(client)

	##############################################################SENSORS####################################################################
	mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
	mti_client.setExposureTime("25")


	###########################################################Generate Scanning Path###########################################################
	r2_mid = np.radians([-7.48,27.4,-13.9,-50.7,-43,63.1])

	### scan parameters
	scan_speed=10 # scanning speed (mm/sec)
	scan_stand_off_d = 95 ## mm
	positioner_init=client.getJointAnglesDB(positioner.pulse2deg) ## init table
	# positioner_init=np.radians([-15,200]) ## init table
	scan_p,scan_R=scan_path_gen(curve_sliced_relative,scan_stand_off_d)
	

	positioner_pose=positioner.fwd(positioner_init,world=True)
	scan_p_world=[]
	scan_R_world=[]
	for i in range(len(scan_p)):
		scan_p_world.append(positioner_pose.p+np.dot(positioner_pose.R,scan_p[i]))
		scan_R_world.append(positioner_pose.R@scan_R[i])
	scan_p_world=np.array(scan_p_world)
	scan_R_world=np.array(scan_R_world)

	
	rrd=redundancy_resolution_dual(robot2_scan,positioner,scan_p,scan_R)
	q_out1, q_out2=rrd.dual_arm_6dof_stepwise(copy.deepcopy(r2_mid),positioner_init,w1=0.1,w2=0.01)	#more weights on robot2_scan to make it move slower

	lam1=calc_lam_js(q_out1,robot2_scan)
	lam2=calc_lam_js(q_out2,positioner)
	lam_relative=calc_lam_cs(scan_p)
	q_bp1,q_bp2,s1_all,s2_all=gen_motion_program_dual(lam1,lam2,lam_relative,q_out1,q_out2,v=10)

	## move to start
	ws.jog_single(robot,[0.,0.,np.pi/6,0.,0.,0.])
	ws.jog_dual(robot2_scan,positioner,r2_mid,q_bp2[0])
	ws.jog_dual(robot2_scan,positioner,q_bp1[0],q_bp2[0])

	## motion start
	mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
	# calibration motion
	target2=['MOVJ',np.degrees(q_bp2[1]),s2_all[0]]
	mp.MoveL(np.degrees(q_bp1[1]), scan_speed, 0, target2=target2)
	# routine motion
	for path_i in range(2,len(q_bp1)-1):
		target2=['MOVJ',np.degrees(q_bp2[path_i]),s2_all[path_i]]
		mp.MoveL(np.degrees(q_bp1[path_i]), s1_all[path_i], target2=target2)
	target2=['MOVJ',np.degrees(q_bp2[-1]),s2_all[-1]]
	mp.MoveL(np.degrees(q_bp1[-1]), s1_all[-1], 0, target2=target2)

	ws.client.execute_motion_program_nonblocking(mp)
	###streaming
	ws.client.StartStreaming()
	start_time=time.time()
	state_flag=0
	joint_recording=[]
	robot_ts=[]
	global_ts=[]
	mti_recording=[]
	while True:
		if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
			break 
		res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
		if res:
			joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
			state_flag=fb_data.controller_flags
			joint_recording.append(joint_angle)
			timestamp=fb_data.time
			robot_ts.append(timestamp)
			global_ts.append(time.perf_counter())
			###MTI scans YZ point from tool frame
			mti_recording.append(copy.deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
	ws.client.servoMH(False)
	
	mti_recording=np.array(mti_recording)
	joint_recording=np.array(joint_recording)

	positioner_up=client.getJointAnglesDB(positioner.pulse2deg)
	positioner_up[0]=np.radians(-15)
	ws.jog_dual(robot2_scan,positioner,r2_mid,positioner_up,v=1)
	###########################################################Save Scanning Data###########################################################
	np.savetxt(recorded_dir + 'scan_js_exe.csv',np.hstack((np.array(global_ts).reshape(-1,1),np.array(robot_ts).reshape(-1,1),joint_recording)),delimiter=',')
	with open(recorded_dir + 'mti_scans.pickle', 'wb') as file:
		pickle.dump(mti_recording, file)


if __name__ == '__main__':
	main()