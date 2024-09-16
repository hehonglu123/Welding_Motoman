import sys, glob, pickle, os, traceback, wave
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from pathlib import Path
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from flir_toolbox import *
from StreamingSend import *
sys.path.append('../../sensor_fusion/')
from weldRRSensor import *

def new_frame(pipe_ep):
	global flir_logging, flir_ts, image_consts
	#Loop to get the newest frame
	while (pipe_ep.Available > 0):
		#Receive the packet
		rr_img=pipe_ep.ReceivePacket()
		if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono8"]:
			# Simple uint8 image
			mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		elif rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
			data_u16 = np.array(rr_img.data.view(np.uint16))
			mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
		
		ir_format = rr_img.image_info.extended["ir_format"].data

		if ir_format == "temperature_linear_10mK":
			display_mat = (mat * 0.01) - 273.15    
		elif ir_format == "temperature_linear_100mK":
			display_mat = (mat * 0.1) - 273.15    
		else:
			display_mat = mat

		#Convert the packet to an image and set the global variable
		flir_logging.append(display_mat)
		flir_ts.append(rr_img.image_info.data_header.ts['seconds']+rr_img.image_info.data_header.ts['nanoseconds']*1e-9)

############################################################WELDING PARAMETERS########################################################

dataset='wall/'
sliced_alg='dense_slice/'
data_dir='../../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
#load template
template = cv2.imread('../../FLIR/tracking/torch_template.png',0)


robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../../config/MA1440_A0_robot_default_config.yml',tool_file_path='../../config/flir.csv',\
		pulse2deg_file_path='../../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../../config/MA1440_pose_mocap.csv')
positioner=positioner_obj('D500B',def_path='../../config/D500B_robot_default_config.yml',tool_file_path='../../config/positioner_tcp.csv',\
	pulse2deg_file_path='../../config/D500B_pulse2deg_real.csv',base_transformation_file='../../config/D500B_pose.csv')


########################################################RR Microphone########################################################
microphone = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
########################################################RR FLIR########################################################
flir=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
flir.setf_param("focus_pos", RR.VarValue(int(2000),"int32"))
flir.setf_param("object_distance", RR.VarValue(0.4,"double"))
flir.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
flir.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
flir.setf_param("relative_humidity", RR.VarValue(50,"double"))
flir.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
flir.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
flir.setf_param("current_case", RR.VarValue(2,"int32"))
# flir.setf_param("ir_format", RR.VarValue("temperature_linear_100mK","string"))
flir.setf_param("ir_format", RR.VarValue("radiometric","string"))
flir.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
flir.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
flir.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))
global image_consts, flir_logging, flir_ts
image_consts = RRN.GetConstants('com.robotraconteur.image', flir)
p=flir.frame_stream.Connect(-1)
#Set the callback for when a new pipe packet is received to the
#new_frame function
flir_logging=[]
flir_ts=[]
p.PacketReceivedEvent+=new_frame
try:
	flir.start_streaming()
except: pass
########################################################RR FRONIUS########################################################
fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.prepare_welder()
########################################################RR CURRENT########################################################
current_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
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

##########################################SENSORS LOGGIGN########################################################
rr_sensors = WeldRRSensor(weld_service=fronius_sub,cam_service=None,microphone_service=microphone,current_service=current_sub)


###########################################layer welding############################################
q14=np.zeros(14)
res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[:6]

layer_counts=0
slice_num=0
welding_started=False
###job=feedrate/10+200
job_offset=300
nominal_feedrate=150
nominal_vd_relative=6
nominal_wire_length=25 #pixels
nominal_temp_below=500

###set up control parameters
base_feedrate=200
base_vd=2
feedrate=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=60
feedrate_max=300
nominal_slice_increment=int(1.3/slicing_meta['line_resolution'])
slice_inc_gain=3.

###BASELAYER WELDING
# slice_num=0
# for i in range(2):
# 	x=0
# 	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')

	
# 	lam_relative=calc_lam_cs(curve_sliced_relative)

# 	lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
# 	rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
# 	rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
# 	positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)

# 	breakpoints=SS.get_breakpoints(lam_relative_dense,base_vd)

# 	###find which end to start
# 	if np.linalg.norm(q14[:6]-rob1_js_dense[0])>np.linalg.norm(q14[:6]-rob1_js_dense[-1]):
# 		breakpoints=np.flip(breakpoints)

# 	###monitoring parameters
# 	wire_length=[]
# 	temp_below=[]
	
# 	fronius_client.job_number = int(base_feedrate/10)+job_offset
# 	###start welding at the first layer, then non-stop
# 	if not welding_started:
# 		SS.jog2q(np.hstack((rob1_js_dense[breakpoints[0]],rob2_js_dense[breakpoints[0]],positioner_js_dense[breakpoints[0]])))
# 		fronius_client.start_weld()
# 		welding_started=True

# 	timestamp_robot=[]
# 	joint_recording=[]
# 	for bp_idx in range(len(breakpoints)):
		
# 		if bp_idx<10:	###streaming 10 points before process FLIR monitoring
# 			robot_timestamp,q14=SS.position_cmd(np.hstack((rob1_js_dense[breakpoints[bp_idx]],rob2_js_dense[breakpoints[bp_idx]],positioner_js_dense[breakpoints[bp_idx]])),time.time())
# 		else:
# 			point_stream_start_time=time.time()
# 			robot_timestamp,q14=SS.position_cmd(np.hstack((rob1_js_dense[breakpoints[bp_idx]],rob2_js_dense[breakpoints[bp_idx]],positioner_js_dense[breakpoints[bp_idx]])))
# 			####################################FLIR PROCESSING####################################
# 			#torch detection
# 			torch_loc=torch_detect(flir_logging[-1],template)
# 			torch_eef=[torch_loc[0]+template.shape[1]-3,torch_loc[1]+int(template.shape[0]/2)]
# 			centroid, bbox, pixels=weld_detection(flir_logging[-1])
# 			if centroid is not None:
# 				###find wire tip
# 				indices=[np.argmax(pixels[0]),np.argmin(pixels[0])]
# 				idx=indices[np.argmin(np.abs(pixels[0][indices]-torch_eef[1]))]
# 				tip=[pixels[1][idx],pixels[0][idx]] #x,y coordinates
# 				wire_length.append(abs(tip[0]-torch_eef[0]))
# 				###find temperature below
# 				bbox_below_size=5
# 				centroid_below=(int(tip[0]+bbox[2]/2+bbox_below_size/2),tip[1])
# 				temp_in_bbox=counts2temp(flir_logging[-1][int(centroid_below[1]-bbox_below_size):int(centroid_below[1]+bbox_below_size),int(centroid_below[0]-bbox_below_size):int(centroid_below[0]+bbox_below_size)].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.85)
# 				temp_below.append(np.average(temp_in_bbox))
			
# 			if bp_idx<len(breakpoints)-1: ###no wait at last point
# 				###busy wait for accurate 8ms streaming
# 				while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
# 					continue
# 		timestamp_robot.append(robot_timestamp)
# 		joint_recording.append(q14)

# 	###LOGGING
# 	local_recorded_dir='recorded_data/wall_recording/'
# 	os.makedirs(local_recorded_dir,exist_ok=True)
# 	np.savetxt(local_recorded_dir+'slice_%i_%i_joint.csv'%(slice_num,x),np.hstack(((np.array(timestamp_robot)-timestamp_robot[0]).reshape((-1,1)),np.array(joint_recording))),delimiter=',')
# 	flir_ts_rec=np.array(flir_ts)-flir_ts[0]
# 	np.savetxt(local_recorded_dir+'slice_%i_%i_flir_ts.csv'%(slice_num,x),flir_ts_rec,delimiter=',')
# 	with open(local_recorded_dir+'slice_%i_%i_flir.pickle'%(slice_num,x), 'wb') as file:
# 		pickle.dump(flir_logging, file)

# 	wire_length_avg=np.average(wire_length)
# 	temp_below_avg=np.average(temp_below)
# 	###proportional feedback
# 	feedrate=min(max(feedrate+1*(nominal_temp_below-temp_below_avg),feedrate_min),feedrate_max)
# 	slice_increment=max(nominal_slice_increment+slice_inc_gain*(nominal_wire_length-wire_length_avg),1)
# 	print('WIRE LENGTH: ',wire_length_avg,'TEMP BELOW: ',temp_below_avg,'FEEDRATE: ',feedrate,'SLICE INC: ',slice_increment)
# 	slice_num+=int(nominal_slice_increment)

	

slice_num=30
while slice_num<slicing_meta['num_layers']:

	###############DETERMINE SECTION ORDER###########################
	sections=[0]
	####################DETERMINE CURVE ORDER##############################################
	for x in sections:
		try:
			rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
			curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')

			
			lam_relative=calc_lam_cs(curve_sliced_relative)

			lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
			rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
			rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
			positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)

			breakpoints=SS.get_breakpoints(lam_relative_dense,vd_relative)

			###find which end to start
			if np.linalg.norm(q14[:6]-rob1_js_dense[0])>np.linalg.norm(q14[:6]-rob1_js_dense[-1]):
				breakpoints=np.flip(breakpoints)

			###monitoring parameters
			wire_length=[]
			temp_below=[]
			audio_recording=[]
			timestamp_robot=[]
			joint_recording=[]
			flir_logging=[]
			flir_ts=[]

			fronius_client.job_number = int(feedrate/10)+job_offset
			###start welding at the first layer, then non-stop
			if not welding_started:
				SS.jog2q(np.hstack((rob1_js_dense[breakpoints[0]],rob2_js_dense[breakpoints[0]],positioner_js_dense[breakpoints[0]])))
				welding_started=True
				fronius_client.start_weld()
				
			rr_sensors.start_all_sensors()
			for bp_idx in range(len(breakpoints)):
				
				if bp_idx<10:	###streaming 10 points before process FLIR monitoring
					robot_timestamp,q14=SS.position_cmd(np.hstack((rob1_js_dense[breakpoints[bp_idx]],rob2_js_dense[breakpoints[bp_idx]],positioner_js_dense[breakpoints[bp_idx]])),time.time())
				else:
					point_stream_start_time=time.time()
					robot_timestamp,q14=SS.position_cmd(np.hstack((rob1_js_dense[breakpoints[bp_idx]],rob2_js_dense[breakpoints[bp_idx]],positioner_js_dense[breakpoints[bp_idx]])))
					####################################FLIR PROCESSING####################################
					#torch detection
					torch_loc=torch_detect(flir_logging[-1],template)
					torch_eef=[torch_loc[0]+template.shape[1]-3,torch_loc[1]+int(template.shape[0]/2)]
					centroid, bbox, pixels=weld_detection(flir_logging[-1])
					if centroid is not None:
						###find wire tip
						indices=[np.argmax(pixels[0]),np.argmin(pixels[0])]
						idx=indices[np.argmin(np.abs(pixels[0][indices]-torch_eef[1]))]
						tip=[pixels[1][idx],pixels[0][idx]] #x,y coordinates
						wire_length.append(abs(tip[0]-torch_eef[0]))
						###find temperature below
						bbox_below_size=5
						centroid_below=(int(tip[0]+bbox[2]/2+bbox_below_size/2),tip[1])
						temp_in_bbox=counts2temp(flir_logging[-1][int(centroid_below[1]-bbox_below_size):int(centroid_below[1]+bbox_below_size),int(centroid_below[0]-bbox_below_size):int(centroid_below[0]+bbox_below_size)].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.85)
						temp_below.append(np.average(temp_in_bbox))
						
					if bp_idx<len(breakpoints)-1: ###no wait at last point
						###busy wait for accurate 8ms streaming
						while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
							continue
					timestamp_robot.append(robot_timestamp)
					joint_recording.append(q14)
			

			###LOGGING
			rr_sensors.stop_all_sensors()
			recorded_dir='recorded_data/wall_recording/'
			layer_data_dir=recorded_dir+'layer_'+str(slice_num)+'/'
			Path(layer_data_dir).mkdir(exist_ok=True)
			flir_ts_rec=np.array(flir_ts)-flir_ts[0]
			np.savetxt(layer_data_dir+'ir_stamps.csv',flir_ts_rec,delimiter=',')
			with open(layer_data_dir+'ir_recording.pickle', 'wb') as file:
				pickle.dump(flir_logging, file)
			np.savetxt(layer_data_dir+'joint_recording.csv',np.hstack(((np.array(timestamp_robot)-timestamp_robot[0]).reshape(-1, 1),np.array(joint_recording))),delimiter=',')
			rr_sensors.save_all_sensors(layer_data_dir)

			wire_length_avg=np.average(wire_length)
			temp_below_avg=np.average(temp_below)
			###proportional feedback
			# feedrate=min(max(feedrate+1*(nominal_temp_below-temp_below_avg),feedrate_min),feedrate_max)
			if slice_num>100:
				slice_increment=max(nominal_slice_increment+slice_inc_gain*(nominal_wire_length-wire_length_avg),-nominal_slice_increment+2)
			else:
				slice_increment=nominal_slice_increment
			# vd_relative=feedrate/10-3
			print('WIRE LENGTH: ',wire_length_avg,'TEMP BELOW: ',temp_below_avg,'FEEDRATE: ',feedrate,'VD: ',vd_relative,'SLICE INC: ',slice_increment)
			slice_num+=int(slice_increment)

		except:
			traceback.print_exc()
			fronius_client.stop_weld()
		

	
	

fronius_client.stop_weld()

	

	
