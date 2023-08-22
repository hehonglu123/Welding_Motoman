import sys, glob, wave, pickle
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../../toolbox/')
from utils import *
from robot_def import *
from lambda_calc import *
from multi_robot import *
from flir_toolbox import *
from dx200_motion_program_exec_client import *
from StreamingSend import *


def warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_x,rob2_js_x,positioner_js_x,reversed=False):
	if positioner_js_x.shape==(2,) and rob1_js_x.shape==(6,):
		return rob1_js,rob2_js,positioner_js
	traj_length_x_half=int(len(rob1_js_x)/2)
	traj_length_half=int(len(rob1_js)/2)
	if reversed:
		rob1_js[:traj_length_half]=spiralize(rob1_js[:traj_length_half],rob1_js_x[:traj_length_x_half],reversed)
		rob2_js[:traj_length_half]=spiralize(rob2_js[:traj_length_half],rob2_js_x[:traj_length_x_half],reversed)
		positioner_js[:traj_length_half]=spiralize(positioner_js[:traj_length_half],positioner_js_x[:traj_length_x_half],reversed)
	else:
		rob1_js[traj_length_half:]=spiralize(rob1_js[traj_length_half:],rob1_js_x[traj_length_x_half:],reversed)
		rob2_js[traj_length_half:]=spiralize(rob2_js[traj_length_half:],rob2_js_x[traj_length_x_half:],reversed)
		positioner_js[traj_length_half:]=spiralize(positioner_js[traj_length_half:],positioner_js_x[traj_length_x_half:],reversed)
	return rob1_js,rob2_js,positioner_js


timestamp=[]
voltage=[]
current=[]
feedrate=[]
energy=[]

def wire_cb(sub, value, ts):
	global timestamp, voltage, current, feedrate, energy

	timestamp.append(value.ts['microseconds'][0])
	voltage.append(value.welding_voltage)
	current.append(value.welding_current)
	feedrate.append(value.wire_speed)
	energy.append(value.welding_energy)



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


dataset='diamond/'
sliced_alg='cont_sections/'
data_dir='../../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/'

layer_height_num=int(1.8/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../../config/MA1440_A0_robot_default_config.yml',tool_file_path='../../config/flir.csv',\
		pulse2deg_file_path='../../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../../config/D500B_robot_default_config.yml',tool_file_path='../../config/positioner_tcp.csv',\
	pulse2deg_file_path='../../config/D500B_pulse2deg_real.csv',base_transformation_file='../../config/D500B_pose.csv')

########################################################RR Microphone########################################################
samplerate = 44000
channels = 1
audio_recording=[]
def microphone_new_frame(pipe_ep):
	global audio_recording
	#Loop to get the newest frame
	while (pipe_ep.Available > 0):
		#Receive the packet
		audio_recording.extend(pipe_ep.ReceivePacket().audio_data)
microphone = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
p_microphone = microphone.microphone_stream.Connect(-1)
p_microphone.PacketReceivedEvent+=microphone_new_frame
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
global image_consts
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
welder_state_sub=fronius_sub.SubscribeWire("welder_state")
welder_state_sub.WireValueChanged += wire_cb
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.prepare_welder()
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


###set up control parameters
job_offset=200
nominal_feedrate=210
nominal_vd_relative=0.5
nominal_wire_length=25 #pixels
nominal_temp_below=500
base_feedrate_cmd=270
base_vd=5
feedrate_cmd=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=60
feedrate_max=300
nominal_slice_increment=int(1.8/slicing_meta['line_resolution'])
slice_inc_gain=3.

###########################################layer welding############################################

res, robot_state, _ = RR_robot_state.TryGetInValue()
q14=robot_state.joint_position
# q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only

welding_started=False
######################################################BASE LAYER##########################################################################################
# slice_num=0
# num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_*.csv'))
# for x in range(0,num_sections,layer_width_num):
# 	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
# 		continue
# 	if x>0:
# 		rob1_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		rob2_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		positioner_js_prev=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
# 		if x<num_sections-layer_width_num:
# 			rob1_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			rob2_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			positioner_js_next=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
	
# 	###find closest %2pi
# 	num2p=np.round((q14[-2:]-positioner_js[0])/(2*np.pi))
# 	positioner_js+=num2p*2*np.pi

# 	lam_relative=calc_lam_cs(curve_sliced_relative)
# 	lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
# 	curve_js_all_dense=interp1d(lam_relative,np.hstack((rob1_js,rob2_js,positioner_js)),kind='cubic',axis=0)(lam_relative_dense)
# 	breakpoints=SS.get_breakpoints(lam_relative_dense,base_vd)

# 	###start welding at the first layer, then non-stop
# 	fronius_client.job_number = int(base_feedrate_cmd/10)+job_offset
# 	if not welding_started:
# 		SS.jog2q(curve_js_all_dense[breakpoints[0]])
# 		welding_started=True
# 		fronius_client.start_weld()
# 	SS.traj_streaming(curve_js_all_dense[breakpoints],ctrl_joints=np.ones(14))

# fronius_client.stop_weld()
######################################################LAYER WELDING##########################################################################################
####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]
lam_relative_all_slices=[]
lam_relative_dense_all_slices=[]
for i in range(0,slicing_meta['num_layers']-1):
	rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
	rob2_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','))
	positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'_0.csv',delimiter=',')
	lam_relative=calc_lam_cs(curve_sliced_relative)
	lam_relative_all_slices.append(lam_relative)
	lam_relative_dense_all_slices.append(np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance)))

slice_num=1
# while slice_num<slicing_meta['num_layers']:
while slice_num<13:
	x=0
	
	rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
	rob2_js=copy.deepcopy(rob2_js_all_slices[slice_num])
	positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
		continue
	
	###TRJAECTORY WARPING
	if slice_num>10:
		rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-layer_height_num])
		rob2_js_prev=copy.deepcopy(rob2_js_all_slices[slice_num-layer_height_num])
		positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-layer_height_num])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
	if slice_num<slicing_meta['num_layers']-layer_height_num:
		rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+layer_height_num])
		rob2_js_next=copy.deepcopy(rob2_js_all_slices[slice_num+layer_height_num])
		positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+layer_height_num])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
	###find closest %2pi
	num2p=np.round((q14[-2:]-positioner_js[0])/(2*np.pi))
	positioner_js+=num2p*2*np.pi

	curve_js_all_dense=interp1d(lam_relative_all_slices[slice_num],np.hstack((rob1_js,rob2_js,positioner_js)),kind='cubic',axis=0)(lam_relative_dense_all_slices[slice_num])
	breakpoints=SS.get_breakpoints(lam_relative_dense_all_slices[slice_num],vd_relative)

	###monitoring parameters
	wire_length=[]
	temp_below=[]
	audio_recording=[]
	robot_ts=[]
	robot_js=[]
	flir_logging=[]
	flir_ts=[]

	fronius_client.job_number = int(feedrate_cmd/10)+job_offset
	###start welding at the first layer, then non-stop
	if not welding_started:
		#jog above
		waypoint_pose=robot.fwd(curve_js_all_dense[0,:6])
		waypoint_pose.p[-1]+=50
		waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_js_all_dense[breakpoints[0],:6])[0]
		SS.jog2q(np.hstack((waypoint_q,curve_js_all_dense[0,6:])))
		SS.jog2q(curve_js_all_dense[breakpoints[0]])
		welding_started=True
		fronius_client.start_weld()
		time.sleep(0.2)
	
	try:
		for bp_idx in range(len(breakpoints)):
			
			if bp_idx<10:	###streaming 10 points before process FLIR monitoring
				robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[breakpoints[bp_idx]],time.time())
			else:
				point_stream_start_time=time.time()
				robot_timestamp,q14=SS.position_cmd(curve_js_all_dense[breakpoints[bp_idx]])
				####################################FLIR PROCESSING####################################
				#TODO: make sure processing time within 8ms
				centroid, bbox=flame_detection(flir_logging[-1])
				if centroid is not None:
					bbox_below_size=5
					centroid_below=(int(centroid[0]+bbox[2]/2+bbox_below_size/2),centroid[1])
					temp_in_bbox=counts2temp(flir_logging[-1][int(centroid_below[1]-bbox_below_size):int(centroid_below[1]+bbox_below_size),int(centroid_below[0]-bbox_below_size):int(centroid_below[0]+bbox_below_size)].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13)
					temp_below.append(np.average(temp_in_bbox))
					wire_length.append(centroid[0]-139)
				
				if bp_idx<len(breakpoints)-1: ###no wait at last point
					###busy wait for accurate 8ms streaming
					while time.time()-point_stream_start_time<1/SS.streaming_rate-0.0005:
						continue
				robot_ts.append(robot_timestamp)
				robot_js.append(q14)
	
		###LOGGING
		# local_recorded_dir='recorded_data/cup_recording_spiral/'
		# os.makedirs(local_recorded_dir,exist_ok=True)
		# np.savetxt(local_recorded_dir+'slice_%i_%i_joint.csv'%(slice_num,x),np.hstack((np.array(robot_ts)-robot_ts[0].reshape((-1,1)),np.array(robot_js))),delimiter=',')
		# flir_ts_rec=np.array(flir_ts)-flir_ts[0]
		# np.savetxt(local_recorded_dir+'slice_%i_%i_flir_ts.csv'%(slice_num,x),flir_ts_rec,delimiter=',')
		# with open(local_recorded_dir+'slice_%i_%i_flir.pickle'%(slice_num,x), 'wb') as file:
		# 	pickle.dump(flir_logging, file)
		# first_channel = np.concatenate(audio_recording)
		# first_channel_int16=(first_channel*32767).astype(np.int16)
		# with wave.open(local_recorded_dir+'slice_%i_%i_microphone.wav'%(slice_num,x), 'wb') as wav_file:
		# 	# Set the WAV file parameters
		# 	wav_file.setnchannels(channels)
		# 	wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
		# 	wav_file.setframerate(samplerate)

		# 	# Write the audio data to the WAV file
		# 	wav_file.writeframes(first_channel_int16.tobytes())

		# wire_length_avg=np.average(wire_length)
		# temp_below_avg=np.average(temp_below)
		# ###proportional feedback
		# # feedrate=min(max(feedrate+1*(nominal_temp_below-temp_below_avg),feedrate_min),feedrate_max)
		# slice_increment=max(nominal_slice_increment+slice_inc_gain*(nominal_wire_length-wire_length_avg),-nominal_slice_increment+2)
		# # vd_relative=feedrate/10-3
		
		
		
		
		feedrate_cmd-=20
		vd_relative+=1
		vd_relative=min(10,vd_relative)
		feedrate_cmd=max(feedrate_cmd,100)
		slice_num+=int(nominal_slice_increment)
		print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)


	except:
		traceback.print_exc()
		fronius_client.stop_weld()



fronius_client.stop_weld()
