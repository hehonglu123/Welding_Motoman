import sys, glob, pickle, os
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from StreamingSend import *
sys.path.append('../../sensor_fusion/')
from weldRRSensor import *

def my_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return
	
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
dataset='blade0.1/'
sliced_alg='auto_slice/'
data_dir='../../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/'
Path(recorded_dir).mkdir(exist_ok=True)

robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../../config/MA1440_A0_robot_default_config.yml',tool_file_path='../../config/flir.csv',\
		pulse2deg_file_path='../../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../../config/MA1440_pose.csv')
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
fronius_client = RRN.ConnectService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client.job_number = 420
fronius_client.prepare_welder()
welding_started=False

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


# ###set up control parameters
job_offset=400 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
nominal_feedrate=80
nominal_vd_relative=0.1
nominal_wire_length=25 #pixels
nominal_temp_below=500
base_feedrate_cmd=300
base_vd=3
feedrate_cmd=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=100
feedrate_max=300
nominal_slice_increment=int(1.3/slicing_meta['line_resolution'])
slice_inc_gain=3.
vd_max=6

###########################################base layer welding############################################
# num_baselayer=2
# feedrate_cmd=base_feedrate_cmd
# fronius_client.async_set_job_number(int(feedrate_cmd/10)+job_offset, my_handler)
# q_prev=np.array([-3.791547245558870571e-01,7.167996965635117235e-01,2.745092098742105691e-01,2.111291009755724701e-01,-7.843516348888318612e-01,-5.300740197588397207e-01])
# for base_layer in range(num_baselayer):
# 	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_*.csv'))
# 	for x in range(num_sections):
# 		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')

# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')

# 		lam_relative=calc_lam_cs(curve_sliced_relative)


# 		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
# 		rob1_js_dense=interp1d(lam_relative,rob1_js,axis=0)(lam_relative_dense)
# 		rob2_js_dense=interp1d(lam_relative,rob2_js,axis=0)(lam_relative_dense)
# 		positioner_js_dense=interp1d(lam_relative,positioner_js,axis=0)(lam_relative_dense)


# 		breakpoints=SS.get_breakpoints(lam_relative_dense,vd_relative)

# 		###find which end to start
# 		if np.linalg.norm(q_prev-rob1_js[0])>np.linalg.norm(q_prev-rob1_js[-1]):
# 			breakpoints=np.flip(breakpoints)

# 		curve_js_all=np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints]))
# 		if not welding_started:
# 			SS.jog2q(curve_js_all[0])
# 			welding_started=True
# 			fronius_client.start_weld()
			
		
# 		##########WELDING#######
# 		SS.traj_streaming(curve_js_all,ctrl_joints=np.ones(14))
# 		q_prev=rob1_js_dense[breakpoints[-1]]

# fronius_client.stop_weld()

###########################################layer welding############################################
###memory logging
robot_logging_all=[]
weld_logging_all=[]
current_logging_all=[]
audio_logging_all=[]
flir_logging_all=[]
flir_ts_logging_all=[]
slice_logging_all=[]
# now=None
####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]
lam_relative_all_slices=[]
lam_relative_dense_all_slices=[]
for slice_num in range(0,slicing_meta['num_layers']-1):
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_*.csv'))
	rob1_js_slice_sections=[]
	rob2_js_slice_sections=[]
	positioner_js_slice_sections=[]
	lam_relative_slice_sections=[]
	lam_relative_dense_slice_sections=[]

	for x in range(num_sections):
		rob1_js_slice_sections.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_0.csv',delimiter=','))
		rob2_js_slice_sections.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_0.csv',delimiter=','))
		positioner_js_slice_sections.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_0.csv',delimiter=','))

		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_0.csv',delimiter=',')
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_slice_sections.append(lam_relative)
		lam_relative_dense_slice_sections.append(np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance)))
	
	rob1_js_all_slices.append(rob1_js_slice_sections)
	rob2_js_all_slices.append(rob2_js_slice_sections)
	positioner_js_all_slices.append(positioner_js_slice_sections)
	lam_relative_all_slices.append(lam_relative_slice_sections)
	lam_relative_dense_all_slices.append(lam_relative_dense_slice_sections)


res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[:6]
slice_num=0
num_sections=1
while True:
	###change feedrate
	fronius_client.async_set_job_number(int(feedrate_cmd/10)+job_offset, my_handler)

	num_sections_prev=num_sections
	num_sections=len(lam_relative_all_slices[slice_num])

	###############DETERMINE SECTION ORDER###########################
	if num_sections==1:
		sections=[0]
	else:
		endpoints=[]
		rob1_js_first=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_0.csv',delimiter=',')
		rob1_js_last=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(num_sections-1)+'.csv',delimiter=',')
		endpoints=np.array([rob1_js_first[0],rob1_js_first[-1],rob1_js_last[0],rob1_js_last[-1]])
		clost_idx=np.argmin(np.linalg.norm(endpoints-q_prev,axis=1))
		if clost_idx>1:
			sections=reversed(range(num_sections))
		else:
			sections=range(num_sections)

	####################DETERMINE CURVE ORDER##############################################
	for x in sections:
		rob1_js=rob1_js_all_slices[slice_num][x]
		rob2_js=rob2_js_all_slices[slice_num][x]
		positioner_js=positioner_js_all_slices[slice_num][x]
		
		lam_relative=lam_relative_all_slices[slice_num][x]

		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		rob1_js_dense=interp1d(lam_relative,rob1_js,kind='cubic',axis=0)(lam_relative_dense)
		rob2_js_dense=interp1d(lam_relative,rob2_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)

		breakpoints=SS.get_breakpoints(lam_relative_dense,vd_relative)

		###find which end to start
		if np.linalg.norm(q_prev-rob1_js[0])>np.linalg.norm(q_prev-rob1_js[-1]):
			breakpoints=np.flip(breakpoints)

		
		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections>1 or num_sections<num_sections_prev:
			fronius_client.stop_weld()
			welding_started=False
			waypoint_pose=robot.fwd(rob1_js_dense[breakpoints[0]])
			waypoint_pose.p[-1]+=30
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js_dense[breakpoints[0]])[0]
			SS.jog2q(np.hstack((waypoint_q,rob2_js_dense[breakpoints[0]],positioner_js_dense[breakpoints[0]])))


		curve_js_all=np.hstack((rob1_js_dense[breakpoints],rob2_js_dense[breakpoints],positioner_js_dense[breakpoints]))
		SS.jog2q(curve_js_all[0])
		
		flir_logging=[]
		flir_ts=[]
		try:
			##########WELDING#######
			if not welding_started:
				welding_started=True
				fronius_client.start_weld()
			robot_ts,robot_js=SS.traj_streaming(curve_js_all,ctrl_joints=np.ones(14))

			###save in memory,
			# now=time.time()
			current_timestamp=np.array(rr_sensors.current_timestamp).flatten()-rr_sensors.current_timestamp[0]
			min_length=min(len(current_timestamp),len(rr_sensors.current))
			current_data=np.array([current_timestamp[:min_length], rr_sensors.current[:min_length]]).T

			weld_timestamp=np.array(rr_sensors.weld_timestamp).flatten()-rr_sensors.weld_timestamp[0]
			min_length=min(len(current_timestamp),len(rr_sensors.weld_voltage),len(rr_sensors.weld_current),len(rr_sensors.weld_feedrate),len(rr_sensors.weld_energy))
			welding_data=np.array([weld_timestamp[:min_length], rr_sensors.weld_voltage[:min_length], rr_sensors.weld_current[:min_length], rr_sensors.weld_feedrate[:min_length], rr_sensors.weld_energy[:min_length]]).T
			
			robot_ts=np.array(robot_ts)
			robot_ts=robot_ts-robot_ts[0]
			robot_js=np.array(robot_js)
			flir_ts=np.array(flir_ts)
			flir_ts=flir_ts-flir_ts[0]
			robot_logging_all.append(np.hstack((robot_ts.reshape(-1, 1),robot_js)))
			weld_logging_all.append(welding_data)
			current_logging_all.append(current_data)
			audio_logging_all.append(rr_sensors.audio_recording)
			flir_logging_all.append(flir_logging)
			flir_ts_logging_all.append(flir_ts)
			slice_logging_all.append((slice_num,x))

			
			####CONTROL PARAMETERS
			slice_num+=int(nominal_slice_increment)
			print('SLICE_NUM: ',slice_num)
			# print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)
		
			q_prev=rob1_js_dense[breakpoints[-1]]

		except:
			traceback.print_exc()
			fronius_client.stop_weld()
			rr_sensors.stop_all_sensors()
			break
	fronius_client.stop_weld()
	rr_sensors.stop_all_sensors()

	for i in range(len(slice_logging_all)):
		rr_sensors.save_data_streaming(recorded_dir,current_logging_all[i],weld_logging_all[i],audio_logging_all[i],robot_logging_all[i],flir_logging_all[i],flir_ts_logging_all[i],slice_logging_all[i][0],slice_logging_all[i][1])


	
