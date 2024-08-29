import time, os, copy, traceback, yaml
from motoman_def import *
from lambda_calc import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from StreamingSend import *
from flir_toolbox import *
from robotics_utils import *


ir_updated_flag=False
ir_process_packet=None

def my_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return
	
def ir_process_cb(sub, value, ts):
	global ir_updated_flag, ir_process_packet

	ir_process_packet=copy.deepcopy(value)
	ir_updated_flag=True

def layer_idle_monitoring(rr_sensors,threshold):
	###monitoring the highest temperature, and wait until drop down to a certain threshold
	max_reading=99999
	while max_reading>threshold:
		time.sleep(0.1)
		max_reading=np.max(rr_sensors.ir_recording[-1])
	return

def main():
	global ir_updated_flag, ir_process_packet

	dataset='wall2/'
	sliced_alg='dense_slice/'
	data_dir='../../../../geometry_data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	open_loop=False
	weld_arcon=True

	##############################################################SENSORS####################################################################
	# weld state logging
	# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
	## RR sensor objects
	rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

	##############################################################FLIR PRORCESS####################################################################
	if not open_loop:
		sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
		ir_process_result=sub.SubscribeWire("ir_process_result")
		ir_process_result.WireValueChanged += ir_process_cb
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../../config/'
	flir_intrinsic=yaml.load(open(config_dir+'FLIR_A320.yaml'), Loader=yaml.FullLoader)
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_no_wire=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
	robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	robot2_flir_imaging=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir_imaging.csv',\
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

	########################################################RR FRONIUS########################################################
	
	if weld_arcon:
		fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
		fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
		hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
		fronius_client.prepare_welder()
	
	########################################################RR STREAMING########################################################
	# RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.12:59945?service=robot')
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	SS=StreamingSend(RR_robot_sub,streaming_rate=125.)

	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	
	base_feedrate=250
	VPD=10
	v_layer=10
	layer_feedrate=VPD*v_layer
	v_base=5
	layer_height=1.2
	num_base_layer=2        #2 base layer to establish adhesion to coupon
	num_support_layer=4     #support layer to raise the cylinder till visible by IR camera
	support_feedrate=150
	v_support=10
	job_offset=450
	

	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])

	
	###############################################################################################################################################################
	###############################################################################################################################################################
	#####################################################BASE & SUPPORT LAYER##########################################################################################
	slice_start=0
	slice_end=2
	slice_increment=1
	q1_cmd_all_base,positioner_cmd_all_base=weld_bf_streaming(SS,data_dir,v_base,slice_increment,num_base_layer,slice_start,slice_end,point_distance=point_distance,q_positioner_prev=SS.q_cur[-2:],layer_name='base_')
	q_cmd_all_base = np.hstack((q1_cmd_all_base, np.array([q2] * len(q1_cmd_all_base)),positioner_cmd_all_base))
	slice_start=0
	slice_end=4
	q1_cmd_all_support,positioner_cmd_all_support=weld_bf_streaming(SS,data_dir,v_support,slice_increment,num_support_layer,slice_start,slice_end,point_distance=point_distance,q_positioner_prev=SS.q_cur[-2:],layer_name='support_')
	q_cmd_all_support = np.hstack((q1_cmd_all_support, np.array([q2] * len(q1_cmd_all_support)),positioner_cmd_all_support))
	###jog to start point
	print("BASE-SUPPORT CALCULATION FINISHED")
	SS.jog2q(q_cmd_all_base[0])
	# plt.plot(np.hstack((q1_cmd_all_base[:,2],q1_cmd_all_support[:,2])))
	# plt.show()
	##############################################################BASE-SUPPORT Layers Welding####################################################################
	if weld_arcon:
		fronius_client.job_number = int(base_feedrate/10+job_offset)
		fronius_client.start_weld()
	for i in range(len(q_cmd_all_base)):
		SS.position_cmd(q_cmd_all_base[i],time.perf_counter())
	if weld_arcon:
		fronius_client.job_number = int(support_feedrate/10+job_offset)
		fronius_client.start_weld()
	for i in range(len(q_cmd_all_support)):
		SS.position_cmd(q_cmd_all_support[i],time.perf_counter())
	if weld_arcon:
		fronius_client.stop_weld()
	print("BASE-SUPPORT LAYER WELDING FINISHED")




	###############################################################################################################################################################
	###############################################################################################################################################################
	#####################################################LAYER Welding##########################################################################################
	###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	rob1_js_all_slices=[]
	positioner_js_all_slices=[]
	lam_relative_all_slices=[]
	v_cmd=v_layer
	for i in range(slicing_meta['num_layers']):
		rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
		positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
		lam_relative_all_slices.append(calc_lam_cs(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=',')))
		
	print("PRELOAD FINISHED")
	
	num_slice_start=0
	num_slice_end=slicing_meta['num_layers']

	slice_num=num_slice_start
	layer_counts=0
	nominal_wire_length=15
	prev_wire_length=nominal_wire_length
	v_gain=1e-3
	dv_max=2
	nominal_pixel_reading=25000
	continuous_stopping_threshold=nominal_pixel_reading+1500
	slice_increment=nominal_slice_increment
	feedrate_update_rate=1.	#Hz
	last_update_time=time.perf_counter()+5.
	arc_off=True
	high_temp_triggered=False
	high_temp_triggered_counts=0

	q_cmd_all=[]
	welding_cmd_all=[]

	q_positioner_prev=SS.q_cur[-2:]
	while slice_num<num_slice_end:
		print("CURRENT SLICE: ",slice_num)
		try:
			next_slice_nominal=min(slice_num+slice_increment,num_slice_end-1)
			####################DETERMINE CURVE ORDER##############################################
			rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
			positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
			if positioner_js.shape==(2,) and rob1_js.shape==(6,):
				continue
			if len(positioner_js_all_slices[next_slice_nominal])>2:
				###TRJAECTORY WARPING
				if layer_counts%2==0:
					if slice_num>num_slice_start:
						rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-slice_increment])
						positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-slice_increment])
						rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
					if slice_num<num_slice_end-slice_increment:
						rob1_js_next=copy.deepcopy(rob1_js_all_slices[next_slice_nominal])
						positioner_js_next=copy.deepcopy(positioner_js_all_slices[next_slice_nominal])
						rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)
				else:
					if slice_num>num_slice_start:
						rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-slice_increment])
						positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-slice_increment])
						rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=False)
					if slice_num<num_slice_end-slice_increment:
						rob1_js_next=copy.deepcopy(rob1_js_all_slices[next_slice_nominal])
						positioner_js_next=copy.deepcopy(positioner_js_all_slices[next_slice_nominal])
						rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=True)
					
					rob1_js=np.flip(rob1_js,axis=0)
					positioner_js=np.flip(positioner_js,axis=0)
			
			###find closest %2pi
			num2p=np.round((q_positioner_prev-positioner_js[0])/(2*np.pi))
			positioner_js+=num2p*2*np.pi


			if layer_counts==0:
				###jog to start point
				SS.jog2q(np.hstack((rob1_js[0],q2,positioner_js[0])))
				rr_sensors.start_all_sensors()
				SS.start_recording()
			if arc_off and weld_arcon:
				print("Welding Start")
				fronius_client.job_number = int(layer_feedrate/10+job_offset)
				fronius_client.start_weld()
				arc_off=False
			
			############################################################Welding Normal Layers ####################################################################
			lam_cur=0
			wire_length=[]
			pixel_reading=[]
			pixel_reading_layer=[]
			while lam_cur<lam_relative_all_slices[slice_num][-1] - v_cmd/SS.streaming_rate:

				loop_start=time.perf_counter()

				lam_cur+=v_cmd/SS.streaming_rate
				#get closest two indices and interpolate the joint angle
				lam_idx=np.where(lam_relative_all_slices[slice_num]>=lam_cur)[0][0]
				ratio=(lam_cur-lam_relative_all_slices[slice_num][lam_idx-1])/(lam_relative_all_slices[slice_num][lam_idx]-lam_relative_all_slices[slice_num][lam_idx-1])
				q1=rob1_js[lam_idx-1]*(1-ratio)+rob1_js[lam_idx]*ratio
				q_positioner=positioner_js[lam_idx-1]*(1-ratio)+positioner_js[lam_idx]*ratio

				q_cmd=np.hstack((q1,q2,q_positioner))
				if ir_updated_flag:			###process IR info and update welding parameters
					ir_updated_flag=False
					torch_pose=robot_no_wire.fwd(SS.q_cur[:6])
					IR_pose=robot2_flir_imaging.fwd(SS.q_cur[6:-2],world=True)
					IR_vector=IR_pose.R@form_vector(ir_process_packet.arc_centroid[0],ir_process_packet.arc_centroid[1],flir_intrinsic)
					wire_tip=line_intersection(torch_pose.p,torch_pose.R[:,2],IR_pose.p,IR_vector)
					wire_length.append(np.linalg.norm(wire_tip-torch_pose.p))
					pixel_reading.append(ir_process_packet.flame_reading)
					pixel_reading_layer.append(ir_process_packet.flame_reading)

				###update welding param
				if time.perf_counter()-last_update_time>1./feedrate_update_rate:
					welding_cmd_all.append(np.hstack((time.perf_counter(),layer_counts,v_cmd,layer_feedrate)))
					print("Interval Average Pixel Reading: ",np.mean(pixel_reading))
					if not np.isnan(np.mean(pixel_reading)):
						v_cmd=v_layer+v_gain*(nominal_pixel_reading-np.mean(pixel_reading))
						v_cmd=min(max(v_cmd,max(5,v_cmd-dv_max)),min(20,v_cmd+dv_max))
						if high_temp_triggered:
							v_cmd=5
						layer_feedrate=VPD*v_cmd
						fronius_client.async_set_job_number(int(layer_feedrate/10)+job_offset, my_handler)
						# print("Adjusted Speed: ",v_cmd)
						# print("ADJUSTED feedrate: ",layer_feedrate)
					pixel_reading=[]
					last_update_time=time.perf_counter()
				
				###Position Command
				q_cmd_all.append(np.hstack((time.perf_counter(),layer_counts,q_cmd)))
				if lam_cur>lam_relative_all_slices[slice_num][-1]-v_cmd/SS.streaming_rate:
					SS.position_cmd(q_cmd)
				else:
					SS.position_cmd(q_cmd,loop_start)

			###choose next slice
			print("Layer Average Wire Length: ",np.mean(wire_length))
			if open_loop:
				slice_increment=nominal_slice_increment
			else:
				act_layer_height=slice_increment*slicing_meta['line_resolution']-(np.mean(wire_length)-prev_wire_length)
				print("MEASURED ACTUAL LAYER HEIGHT: ",act_layer_height)
				slice_increment=(nominal_wire_length-np.mean(wire_length)+act_layer_height)/slicing_meta['line_resolution']
				###safety bound
				slice_increment=int(min(max(slice_increment,1),1.1*nominal_slice_increment))


			print("ADJUSTED slice_increment: ",slice_increment)


			###LAYER IDLE CONDITION
			if weld_arcon:
				if np.mean(pixel_reading_layer)>continuous_stopping_threshold:
					print("Continuous Stopping Condition Reached")
					fronius_client.stop_weld()
					arc_off=True
					high_temp_triggered_counts+=1
					if high_temp_triggered_counts>2:
						high_temp_triggered=True
					layer_idle_monitoring(rr_sensors,nominal_pixel_reading-2000)
					print("Continuous Stopping Condition Cleared")

		

			###loop conditions
			q_positioner_prev=copy.deepcopy(positioner_js[-1])
			layer_counts+=1
			slice_num+=slice_increment
			pixel_reading_layer=[]
			prev_wire_length=np.mean(wire_length)

			
		except:
			traceback.print_exc()
			if weld_arcon:
				fronius_client.stop_weld()
				fronius_client.release_welder()
			SS.deinitialize_robot()
			break
			

	############################################################LOGGING####################################################################
	if weld_arcon:
		fronius_client.stop_weld()
		fronius_client.release_welder()
	SS.deinitialize_robot()
	rr_sensors.stop_all_sensors()
	js_recording = SS.stop_recording()

	if not open_loop:
		recorded_dir='../../../../recorded_data/ER316L/streaming/'+dataset+'bf_T%i/'%(nominal_pixel_reading)
	else:
		recorded_dir='../../../../recorded_data/ER316L/streaming/'+dataset+'bf_ol_v%i_f%i/'%(v_layer,layer_feedrate)

	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_cmd.csv',np.array(q_cmd_all),delimiter=',')
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	np.savetxt(recorded_dir+'weld_cmd.csv',np.array(welding_cmd_all),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)

if __name__ == '__main__':
	main()