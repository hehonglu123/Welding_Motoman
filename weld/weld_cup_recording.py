import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *

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
line_width=0.1
layer_height_num=int(1.3/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])
curve_sliced_js=[]
positioner_js=[]

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

client=MotionProgramExecClient()

###########################################layer welding############################################
num_layer_start=int(30*layer_height_num)
num_layer_end=int(42*layer_height_num)
q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev=np.array([9.53E-02,-2.71E+00])

if num_layer_start<=1:
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
else:
	num_sections=1

for layer in range(num_layer_start,num_layer_end,layer_height_num):
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	for x in range(0,num_sections,layer_width_num):
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		vd_relative=8
		lam1=calc_lam_js(curve_sliced_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

		###find which end to start depending on how close to joint limit
		if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
			breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		else:
			breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

		s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections!=num_sections_prev:
			target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),10]
			waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
			mp.MoveL(np.degrees(waypoint_q), 10,target2=target2)

		target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),15]
		mp.MoveL(np.degrees(curve_sliced_js[breakpoints[0]]), s1_all[0],target2=target2)

		mp.setArc(True,cond_num=410)
		for j in range(1,len(breakpoints)):
			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),min(100,100*positioner_w/positioner.joint_vel_limit[1])]
			mp.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), max(s1_all[j],0.1),target2=target2)
		mp.setArc(False)

		q_prev=positioner_js[breakpoints[-1]]
	

		####DATA LOGGING
		local_recorded_dir=recorded_dir+'slice%i_%i/'%(layer,x)
		if not os.path.isdir(local_recorded_dir):
			os.mkdir(local_recorded_dir)

		timestamp=[]
		voltage=[]
		current=[]
		feedrate=[]
		energy=[]
		timestamp_robot,joint_recording,job_line,_=client.execute_motion_program(mp)
		np.savetxt(local_recorded_dir +'welder_info.csv',
					   np.array([timestamp, voltage, current, feedrate, energy]).T, delimiter=',',
					   header='timestamp,voltage,current,feedrate,energy', comments='')
		np.savetxt(local_recorded_dir+'joint_recording.csv',np.hstack((timestamp_robot.reshape(-1, 1),job_line.reshape(-1, 1),joint_recording)),delimiter=',')