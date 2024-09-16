import time, os, copy
from motoman_def import *
from lambda_calc import *
from RobotRaconteur.Client import *
from weldRRSensor import *
from dual_robot import *
from traj_manipulation import *
from StreamingSend import *

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

def main():
	global ir_updated_flag, ir_process_packet

	dataset='tube/'
	sliced_alg='dense_slice/'
	data_dir='../../../../geometry_data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)


	##############################################################SENSORS####################################################################
	# weld state logging
	# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	cam_ser=RRN.ConnectService('rr+tcp://localhost:60827/?service=camera')
	# mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
	## RR sensor objects
	rr_sensors = WeldRRSensor(weld_service=None,cam_service=cam_ser,microphone_service=None)

	##############################################################FLIR PRORCESS####################################################################
	sub=RRN.SubscribeService('rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS')
	ir_process_result=sub.SubscribeWire("ir_process_result")
	ir_process_result.WireValueChanged += ir_process_cb
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../../config/'
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

	########################################################RR FRONIUS########################################################
	weld_arcon=True
	if weld_arcon:
		fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
		fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
		hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
		fronius_client.prepare_welder()
	
	########################################################RR STREAMING########################################################
	RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
	point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
	SS=StreamingSend(RR_robot_sub,streaming_rate=125.)

	#################################################################robot 1 welding params####################################################################
	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	
	flipped=False   #spiral direction
	base_feedrate=250
	volume_per_distance=10
	v_layer=10
	feedrate=volume_per_distance*v_layer
	base_layer_height=3
	v_base=5
	layer_height=1.1
	num_base_layer=10       #10layers to avoid clamp blocking IR view
	num_layer=20
	q_cmd_all=[]
	welding_cmd_all=[]
	job_offset=450


	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	base_slice_increment=int(base_layer_height/slicing_meta['line_resolution'])


	q_positioner_prev=SS.q_cur[-2:]

	
	###############################################################################################################################################################
	###############################################################################################################################################################
	#####################################################BASE LAYER##########################################################################################
	###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	rob1_js_all_slices=[]
	positioner_js_all_slices=[]
	for i in range(0,num_base_layer*base_slice_increment):
		if not flipped:
			rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
			positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
		else:
			###spiral rotation direction
			rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
			positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))

	print("PRELOAD FINISHED")

	num_layer_end=num_base_layer*base_slice_increment
	for slice_num in range(0,num_layer_end,base_slice_increment):

		####################DETERMINE CURVE ORDER##############################################
		x=0
		rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
		positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
		if positioner_js.shape==(2,) and rob1_js.shape==(6,):
			continue
		
		###TRJAECTORY WARPING
		if slice_num>0:
			rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-base_slice_increment])
			positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-base_slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
		if slice_num<num_layer_end-base_slice_increment:
			rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+base_slice_increment])
			positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+base_slice_increment])
			rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)
				
		
			
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
		breakpoints=SS.get_breakpoints(lam_relative_dense,v_base)

		###find closest %2pi
		num2p=np.round((q_positioner_prev-positioner_js_dense[0])/(2*np.pi))
		positioner_js_dense+=num2p*2*np.pi
		
		###formulate streaming joint angles
		q_cmd_all.extend(np.hstack((rob1_js_dense[breakpoints],[q2]*len(breakpoints),positioner_js_dense[breakpoints])))
		
		q_positioner_prev=copy.deepcopy(positioner_js_dense[-1])

	q_cmd_all=np.array(q_cmd_all)

	###jog to start point
	print("BASELAYER CALCULATION FINISHED")
	SS.jog2q(q_cmd_all[0])
	##############################################################Base Layers Welding####################################################################
	if weld_arcon:
		fronius_client.job_number = int(base_feedrate/10+job_offset)
		fronius_client.start_weld()
	for i in range(len(q_cmd_all)):
		SS.position_cmd(q_cmd_all[i],time.perf_counter())
	if weld_arcon:
		fronius_client.stop_weld()
	print("BASELAYER WELDING FINISHED")




	###############################################################################################################################################################
	###############################################################################################################################################################
	#####################################################LAYER Welding##########################################################################################
	###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
	rob1_js_all_slices=[]
	positioner_js_all_slices=[]
	lam_relative_all_slices=[]
	v_cmd=v_layer
	for i in range(slicing_meta['num_layers']):
		if not flipped:
			rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
			positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
			lam_relative_all_slices.append(calc_lam_cs(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=',')))
		else:
			###spiral rotation direction
			rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
			positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))
			lam_relative_all_slices.append(calc_lam_cs(np.flip(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=','),axis=0)))

		
	print("PRELOAD FINISHED")
	
	
	num_layer_start=int(num_base_layer*base_slice_increment)
	num_layer_end=slicing_meta['num_layers']-1
	slice_num=num_layer_start
	layer_counts=0
	wire_length_gain=2.
	nominal_wire_length=20
	v_gain=1e-3
	nominal_pixel_reading=25000
	slice_increment=nominal_slice_increment
	feedrate_update_rate=1.	#Hz
	last_update_time=time.perf_counter()+5.

	while layer_counts<num_layer:
		try:
			####################DETERMINE CURVE ORDER##############################################
			rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
			positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
			if positioner_js.shape==(2,) and rob1_js.shape==(6,):
				continue
			
			###TRJAECTORY WARPING
			if slice_num>num_layer_start:
				rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-slice_increment])
				positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-slice_increment])
				rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_prev,positioner_js_prev,reversed=True)
			if slice_num<num_layer_end-slice_increment:
				rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+slice_increment])
				positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+slice_increment])
				rob1_js,positioner_js=warp_traj2(rob1_js,positioner_js,rob1_js_next,positioner_js_next,reversed=False)

			
			###find closest %2pi
			num2p=np.round((q_positioner_prev-positioner_js[0])/(2*np.pi))
			positioner_js+=num2p*2*np.pi


			if layer_counts==0:
				###jog to start point
				SS.jog2q(np.hstack((rob1_js[0],q2,positioner_js[0])))
				rr_sensors.start_all_sensors()
				SS.start_recording()
				if weld_arcon:
					print("Welding Start")
					fronius_client.job_number = int(feedrate/10+job_offset)
					fronius_client.start_weld()
			
			############################################################Welding Normal Layers ####################################################################
			lam_cur=0
			wire_length=[]
			pixel_reading=[]
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
					wire_length.append(np.linalg.norm(ir_process_packet.weld_pool-ir_process_packet.torch_bottom))
					pixel_reading.append(ir_process_packet.flame_reading)

				###update welding param
				if time.perf_counter()-last_update_time>1./feedrate_update_rate:
					welding_cmd_all.append(np.hstack((time.perf_counter(),layer_counts,v_cmd,feedrate)))
					print("Layer Average Pixel Reading: ",np.mean(pixel_reading))
					if not np.isnan(np.mean(pixel_reading)):
						v_cmd=v_layer+v_gain*(nominal_pixel_reading-np.mean(pixel_reading))
						v_cmd=min(max(v_cmd,5),17)
						feedrate=volume_per_distance*v_cmd
						fronius_client.async_set_job_number(int(feedrate/10)+job_offset, my_handler)
						print("Adjusted Speed: ",v_cmd)
						print("ADJUSTED feedrate: ",feedrate)
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
			slice_increment=nominal_slice_increment+wire_length_gain*(nominal_wire_length-np.mean(wire_length))
			slice_increment=int(min(max(slice_increment,0.5*nominal_slice_increment),2*nominal_slice_increment))
			print("ADJUSTED slice_increment: ",slice_increment)


			###loop conditions
			q_positioner_prev=copy.deepcopy(positioner_js[-1])
			layer_counts+=1
			slice_num+=slice_increment
		
		except KeyboardInterrupt:
			break

	############################################################LOGGING####################################################################
	if weld_arcon:
		fronius_client.stop_weld()
	rr_sensors.stop_all_sensors()
	js_recording = SS.stop_recording()


	recorded_dir='../../../../recorded_data/ER316L/streaming/tubespiral_T%i/'%(nominal_pixel_reading)
	os.makedirs(recorded_dir,exist_ok=True)
	np.savetxt(recorded_dir+'weld_js_cmd.csv',np.array(q_cmd_all),delimiter=',')
	np.savetxt(recorded_dir+'weld_js_exe.csv',np.array(js_recording),delimiter=',')
	np.savetxt(recorded_dir+'weld_cmd.csv',np.array(welding_cmd_all),delimiter=',')
	rr_sensors.save_all_sensors(recorded_dir)


if __name__ == '__main__':
	main()