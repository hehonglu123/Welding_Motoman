import sys, glob
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from StreamingSend import *


def extend_simple(curve_sliced_js,positioner_js,curve_sliced_relative,lam_relative,d=10):
	pose0=robot.fwd(curve_sliced_js[0])
	pose1=robot.fwd(curve_sliced_js[1])
	vd_start=pose0.p-pose1.p
	vd_start/=np.linalg.norm(vd_start)
	q_start=robot.inv(pose0.p+d*vd_start,pose0.R,curve_sliced_js[0])[0]

	pose_1=robot.fwd(curve_sliced_js[-1])
	pose_2=robot.fwd(curve_sliced_js[-2])
	vd_end=pose_1.p-pose_2.p
	vd_end/=np.linalg.norm(vd_end)
	q_end=robot.inv(pose_1.p+d*vd_end,pose0.R,curve_sliced_js[-1])[0]

	return q_start, q_end

############################################################WELDING PARAMETERS########################################################
dataset='blade0.1/'
sliced_alg='auto_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

layer_height_num=int(1.5/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose_mocap.csv')

########################################################RR FRONIUS########################################################
fronius_client = RRN.ConnectService('rr+tcp://192.168.55.21:60823?service=welder')
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

###########################################base layer welding############################################
# num_baselayer=2
# q_prev=np.array([-3.791547245558870571e-01,7.167996965635117235e-01,2.745092098742105691e-01,2.111291009755724701e-01,-7.843516348888318612e-01,-5.300740197588397207e-01])
# for base_layer in range(num_baselayer):
# 	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_*.csv'))
# 	for x in range(num_sections):
# 		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')

# 		lam_relative=calc_lam_cs(curve_sliced_relative)

# 		breakpoints=np.arange(0,len(lam_relative))

# 		###find which end to start
# 		if np.linalg.norm(q_prev-curve_sliced_js[0])>np.linalg.norm(q_prev-curve_sliced_js[-1]):
# 			breakpoints=np.flip(breakpoints)

# 		SS.jog2q(np.hstack((curve_sliced_js[breakpoints[0]],np.pi/2,[0]*5,positioner_js[breakpoints[0]])))

# 		##########WELDING#######
# 		fronius_client.start_weld()
# 		SS.traj_tracking(lam_relative,np.hstack((curve_sliced_js[breakpoints],positioner_js[breakpoints])),vd_relative,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,1,1]))
# 		time.sleep(0.2)
# 		fronius_client.stop_weld()

# 		q_prev=curve_sliced_js[breakpoints[-1]]

###########################################layer welding############################################
res, robot_state, _ = RR_robot_state.TryGetInValue()
q_prev=robot_state.joint_position[:6]

num_layer_start=int(0*layer_height_num)
num_layer_end=int(3*layer_height_num)
num_sections=1
for layer in range(num_layer_start,num_layer_end,layer_height_num):
	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	###############DETERMINE SECTION ORDER###########################
	if num_sections==1:
		sections=[0]
	else:
		endpoints=[]
		curve_sliced_js_first=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_0.csv',delimiter=',')
		curve_sliced_js_last=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(num_sections-1)+'.csv',delimiter=',')
		endpoints=np.array([curve_sliced_js_first[0],curve_sliced_js_first[-1],curve_sliced_js_last[0],curve_sliced_js_last[-1]])
		clost_idx=np.argmin(np.linalg.norm(endpoints-q_prev,axis=1))
		if clost_idx>1:
			sections=reversed(range(num_sections))
		else:
			sections=range(num_sections)

	####################DETERMINE CURVE ORDER##############################################
	for x in sections:
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		
		lam_relative=calc_lam_cs(curve_sliced_relative)

		breakpoints=np.arange(0,len(lam_relative))

		###find which end to start
		if np.linalg.norm(q_prev-curve_sliced_js[0])>np.linalg.norm(q_prev-curve_sliced_js[-1]):
			breakpoints=np.flip(breakpoints)

		
		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections>1 or num_sections<num_sections_prev:
			waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
			waypoint_pose.p[-1]+=30
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]

			SS.jog2q(np.hstack((waypoint_q,np.pi/2,[0]*5,positioner_js[breakpoints[0]])))

		SS.jog2q(np.hstack((curve_sliced_js[breakpoints[0]],np.pi/2,[0]*5,positioner_js[breakpoints[0]])))


		##########WELDING#######
		fronius_client.start_weld()
		SS.traj_tracking(lam_relative,np.hstack((curve_sliced_js[breakpoints],positioner_js[breakpoints])),vd_relative,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,1,1]))
		time.sleep(0.4)
		fronius_client.stop_weld()

		q_prev=curve_sliced_js[breakpoints[-1]]
	
