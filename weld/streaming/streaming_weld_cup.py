import sys, glob
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from StreamingSend import *

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


dataset='cup/'
sliced_alg='circular_slice_shifted/'
data_dir='../../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/'

waypoint_distance=7 	###waypoint separation
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
fronius_client.job_number = 200
fronius_client.prepare_welder()
vd_relative=5
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


###########################################layer welding############################################
layer_start=11
layers2weld=50
layer_counts=layer_start
num_layer_start=int(layer_start*layer_height_num)	###modify layer num here
num_layer_end=int((layer_start+layers2weld)*layer_height_num)
res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[-2:]
# q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only
timestamp_robot=[]
joint_recording=[]

if num_layer_start<=1*layer_height_num:
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
else:
	num_sections=1

for layer in range(num_layer_start,num_layer_end,layer_height_num):

	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	for x in range(0,num_sections,layer_width_num):
		
		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		if curve_sliced_relative.shape==(6,):
			continue
			
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
		breakpoints=SS.get_breakpoints(lam_relative_dense,vd_relative)


		###find which end to start depending on layer count
		if layer_counts%2==1:
			breakpoints=np.flip(breakpoints)



		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections!=num_sections_prev:
			waypoint_pose=robot.fwd(rob1_js_dense[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js_dense[breakpoints[0]])[0]
			SS.jog2q(np.hstack((waypoint_q,rob2_js_dense[breakpoints[0]],positioner_js_dense[breakpoints[0]])))


		curve_js_all=np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints]))


		SS.jog2q(curve_js_all[0])
		q_prev=positioner_js_dense[breakpoints[-1]]

		flir_logging=[]
		flir_ts=[]
		audio_recording=[]
		##########WELDING#######
		fronius_client.start_weld()
		robot_ts,robot_js=SS.traj_streaming(curve_js_all,ctrl_joints=np.ones(14))

		time.sleep(0.1)
		fronius_client.stop_weld()

		local_recorded_dir='recorded_data/cup_recording/'
		os.makedirs(local_recorded_dir,exist_ok=True)
		np.savetxt(local_recorded_dir+'slice_%i_%i_joint.csv'%(layer,x),np.hstack((robot_ts.reshape((-1,1)),robot_js)),delimiter=',')
		flir_ts=np.array(flir_ts)-flir_ts[0]
		np.savetxt(local_recorded_dir+'slice_%i_%i_flir_ts.csv'%(layer,x),flir_ts,delimiter=',')
		with open(local_recorded_dir+'slice_%i_%i_flir.pickle'%(layer,x), 'wb') as file:
			pickle.dump(flir_logging, file)
		
		first_channel = np.concatenate(audio_recording)
		first_channel_int16=(first_channel*32767).astype(np.int16)
		with wave.open('output.wav', 'wb') as wav_file:
			# Set the WAV file parameters
			wav_file.setnchannels(channels)
			wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
			wav_file.setframerate(samplerate)

			# Write the audio data to the WAV file
			wav_file.writeframes(first_channel_int16.tobytes())

		layer_counts+=1
