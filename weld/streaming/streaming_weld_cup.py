import sys, glob
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../toolbox/')
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


logging=False
if logging:
	sub=RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	obj = sub.GetDefaultClientWait(3)      #connect, timeout=30s
	welder_state_sub=sub.SubscribeWire("welder_state")
	welder_state_sub.WireValueChanged += wire_cb


dataset='cup/'
sliced_alg='circular_slice_shifted/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/cup_ER316L/'

waypoint_distance=5 	###waypoint separation
layer_height_num=int(1.5/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

########################################################RR FRONIUS########################################################
fronius_client = RRN.ConnectService('rr+tcp://192.168.55.10:60823?service=welder')
hflags_const = RRN.GetConstants("experimental.fronius", fronius_client)["WelderStateHighFlags"]
fronius_client.job_number = 200
fronius_client.prepare_welder()
vd_relative=15
########################################################RR STREAMING########################################################

RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
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
num_layer_start=int(0*layer_height_num)	###modify layer num here
num_layer_end=int(1*layer_height_num)
res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[:6]
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
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		if len(curve_sliced_js)<2:
			continue
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		vd_relative=5
		lam1=calc_lam_js(curve_sliced_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)
		lam_relative_dense=np.linspace(0,lam_relative[-1],num=int(lam_relative[-1]/point_distance))
		curve_sliced_js_dense=interp1d(lam_relative,curve_sliced_js,kind='cubic',axis=0)(lam_relative_dense)
		positioner_js_dense=interp1d(lam_relative,positioner_js,kind='cubic',axis=0)(lam_relative_dense)
		breakpoints=get_breapoints(lam_relative_dense,vd_relative,streaming_rate)


		###find which end to start depending on how close to joint limit
		if positioner.upper_limit[1]-q_prev[1]<q_prev[1]-positioner.lower_limit[1]:
			breakpoints=np.flip(breakpoints)



		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections!=num_sections_prev:
			waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			q1=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
			q2=positioner_js[breakpoints[0]]
			SS.jog2q(np.hstack((q1,np.pi/2,[0]*5,q2)))
		
		curve_js_all=np.hstack((curve_sliced_js_dense[breakpoints],0.5*np.pi*np.ones((len(breakpoints),1)),np.zeros((len(breakpoints),5)),positioner_js_dense[breakpoints]))
		
	

		####DATA LOGGING
		if logging:
			local_recorded_dir=recorded_dir+'slice%i_%i/'%(layer,x)
			os.makedirs(local_recorded_dir,exist_ok=True)
			timestamp=[]
			voltage=[]
			current=[]
			feedrate=[]
			energy=[]

		SS.jog2q(curve_js_all[0])

		##########WELDING#######
		while True:
			state, _ = fronius_client.welder_state.PeekInValue()
			hflags = state.welder_state_flags >> 32
			if hflags & 0x200:		###make sure robot_motion_release ready
				break

		fronius_client.start_weld()
		ts,js=SS.traj_streaming(curve_js_all)
		timestamp_robot.extend(ts)
		joint_recording.extend(js)
		time.sleep(0.44)
		fronius_client.stop_weld()

		q_prev=curve_sliced_js_dense[breakpoints[-1]]
		
		if logging:
			np.savetxt(local_recorded_dir +'welder_info.csv',
						np.array([timestamp, voltage, current, feedrate, energy]).T, delimiter=',',
						header='timestamp,voltage,current,feedrate,energy', comments='')
			np.savetxt(local_recorded_dir+'joint_recording.csv',np.hstack((timestamp_robot.reshape(-1, 1),job_line.reshape(-1, 1),joint_recording)),delimiter=',')