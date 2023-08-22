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
sliced_alg='circular_slice/'
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
feedrate_cmd=170
vd_relative=1
fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
welder_state_sub=fronius_sub.SubscribeWire("welder_state")
welder_state_sub.WireValueChanged += wire_cb
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.job_number = int(feedrate_cmd/10)+200
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


###########################################layer welding############################################
layer_start=2
layers2weld=50
layer_counts=layer_start
num_layer_start=int(layer_start*layer_height_num)	###modify layer num here
num_layer_end=min(int((layer_start+layers2weld)*layer_height_num),slicing_meta['num_layers'])
res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[-2:]
# q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only

timestamp_robot=[]
joint_recording=[]
curve_js_all=[]
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
		traj_length=len(rob1_js)
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		if positioner_js.shape==(2,) and rob1_js.shape==(6,):
			continue
		
		###TRJAECTORY WARPING
		if x>0:###if multiple sections
			rob1_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
			rob2_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
			positioner_js_prev=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
			if positioner_js_prev.shape==(2,) and rob1_js_prev.shape==(6,):
				continue
			traj_length_prev=len(rob1_js_prev)
			rob1_js[:int(traj_length/2)]=spiralize(rob1_js[:int(traj_length/2)],rob1_js_prev[:int(traj_length_prev/2)],reversed=True)
			rob2_js[:int(traj_length/2)]=spiralize(rob2_js[:int(traj_length/2)],rob2_js_prev[:int(traj_length_prev/2)],reversed=True)
			positioner_js[:int(traj_length/2)]=spiralize(positioner_js[:int(traj_length/2)],positioner_js_prev[:int(traj_length_prev/2)],reversed=True)
			if x<num_sections-layer_width_num:
				rob1_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
				rob2_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
				positioner_js_next=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
				traj_length_next=len(rob1_js_next)
				rob1_js[int(traj_length/2):]=spiralize(rob1_js[int(traj_length/2):],rob1_js_next[int(traj_length_next/2):])
				rob2_js[int(traj_length/2):]=spiralize(rob2_js[int(traj_length/2):],rob2_js_next[int(traj_length_next/2):])
				positioner_js[int(traj_length/2):]=spiralize(positioner_js[int(traj_length/2):],positioner_js_next[int(traj_length_next/2):])

		if layer>1:
			rob1_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer-layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
			rob2_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer-layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
			positioner_js_prev=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer-layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
			
			if positioner_js_prev.shape==(2,) and rob1_js_prev.shape==(6,):
				continue
			traj_length_prev=len(rob1_js_prev)
			rob1_js[:int(traj_length/2)]=spiralize(rob1_js[:int(traj_length/2)],rob1_js_prev[:int(traj_length_prev/2)],reversed=True)
			rob2_js[:int(traj_length/2)]=spiralize(rob2_js[:int(traj_length/2)],rob2_js_prev[:int(traj_length_prev/2)],reversed=True)
			positioner_js[:int(traj_length/2)]=spiralize(positioner_js[:int(traj_length/2)],positioner_js_prev[:int(traj_length_prev/2)],reversed=True)
			if layer<num_layer_end-layer_height_num:
				rob1_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer+layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
				rob2_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer+layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
				positioner_js_next=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer+layer_height_num)+'_'+str(x)+'.csv',delimiter=',')
				traj_length_next=len(rob1_js_next)
				rob1_js[int(traj_length/2):]=spiralize(rob1_js[int(traj_length/2):],rob1_js_next[int(traj_length_next/2):])
				rob2_js[int(traj_length/2):]=spiralize(rob2_js[int(traj_length/2):],rob2_js_next[int(traj_length_next/2):])
				positioner_js[int(traj_length/2):]=spiralize(positioner_js[int(traj_length/2):],positioner_js_next[int(traj_length_next/2):])
				
		###find closest %2pi
		num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
		positioner_js+=num2p*2*np.pi
			
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
		breakpoints=SS.get_breakpoints(lam_relative_dense,vd_relative)

		q_prev=positioner_js_dense[breakpoints[-1]]

		curve_js_all.append(np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints])))
	
	vd_relative+=1
	vd_relative=min(vd_relative,20)

curve_js_all=np.vstack(curve_js_all)

#jog above
waypoint_pose=robot.fwd(curve_js_all[0,:6])
waypoint_pose.p[-1]+=50
waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js_dense[breakpoints[0]])[0]
SS.jog2q(np.hstack((waypoint_q,curve_js_all[0,6:])))



SS.jog2q(curve_js_all[0])




flir_logging=[]
flir_ts=[]
audio_recording=[]
##########WELDING#######
fronius_client.start_weld()
###flag checking###
time.sleep(0.5)
# while True:
# 	state, _ = fronius_client.welder_state.PeekInValue()
# 	flags = state.welder_state_flags
# 	hflags = state.welder_state_flags >> 32
	
# 	if hflags & 512:
# 		break
		
robot_ts,robot_js=SS.traj_streaming(curve_js_all,ctrl_joints=np.ones(14))

time.sleep(0.1)
fronius_client.stop_weld()


