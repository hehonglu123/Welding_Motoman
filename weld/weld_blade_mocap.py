import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from MocapPoseListener import *


dataset='blade0.1/'
sliced_alg='NX_slice2/'
data_dir='../data/'+dataset+sliced_alg
cmd_dir=data_dir+'cmd/50J/'

waypoint_distance=5 	###waypoint separation
curve_sliced_js=[]
positioner_js=[]
config_dir='../config/'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun_old.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15,  base_marker_config_file=config_dir+'MA2010_marker_config.yaml',\
		tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv',\
	base_marker_config_file=config_dir+'D500B_marker_config.yaml',\
	tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')

mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
client=MotionProgramExecClient()

mocap_url = 'rr+tcp://192.168.55.10:59823?service=optitrack_mocap'
mocap_url = mocap_url
mocap_cli = RRN.ConnectService(mocap_url)
mpl_obj = MocapPoseListener(mocap_cli,[robot,positioner],collect_base_window=240)

###########################################base layer welding############################################
###move to starting position
# curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js0_0.csv',delimiter=',')
# positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js0_0.csv',delimiter=',')
# target2=['MOVJ',np.degrees(positioner_js[0]),10]
# mp.MoveJ(np.degrees(curve_sliced_js[0]), 2,target2=target2)

###baselayer
# num_baselayer=2
# mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
# for base_layer in range(num_baselayer):
# 	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_*.csv'))
# 	for x in range(num_sections):
# 		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')
# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'_'+str(x)+'.csv',delimiter=',')

# 		vd_relative=5
# 		lam1=calc_lam_js(curve_sliced_js,robot)
# 		lam2=calc_lam_js(positioner_js,positioner)
# 		lam_relative=calc_lam_cs(curve_sliced_relative)

# 		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
# 		if base_layer % 2==1:
# 		    breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
# 		else:
# 		    breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

# 		s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

# 		target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),10]
# 		mp.MoveL(np.degrees(curve_sliced_js[breakpoints[0]]), s1_all[0],target2=target2)
# 		for j in range(1,len(breakpoints)):
# 		    target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
# 		    mp.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), s1_all[j],target2=target2)

###########################################layer welding############################################
mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
num_layer_start=0
num_layer_end=5
for layer in range(num_layer_start,num_layer_end):
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))
	if layer % 2==1:
		sections=reversed(range(num_sections))
	else:
		sections=range(num_sections)
	for x in sections:
		curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		vd_relative=20
		lam1=calc_lam_js(curve_sliced_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
		if layer % 2==1:
		    breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		else:
		    breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

		s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)
		
		###move to intermidieate waypoint for collision avoidance
		if num_sections>1:
			target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),30]
			waypoint_pose=robot.fwd(curve_sliced_js[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			waypoint_q=robot.inv(waypoint_pose.p,waypoint_pose.R,curve_sliced_js[breakpoints[0]])[0]
			mp.MoveL(np.degrees(waypoint_q), 25,target2=target2)

		target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),30]
		mp.MoveL(np.degrees(curve_sliced_js[breakpoints[0]]), s1_all[0],target2=target2)

		for j in range(1,len(breakpoints)):
		    target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
		    mp.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), s1_all[j],target2=target2)


mpl_obj.run_pose_listener()

timestamp,joint_recording,job_line,_=client.execute_motion_program(mp) 
mpl_obj.stop_pose_listener()
curve_exe_dict,curve_exe_R_dict,timestamp_dict = mpl_obj.get_robots_traj()

curve_exe = np.array(curve_exe_dict[robot.robot_name])
curve_exe_R = np.array(curve_exe_R_dict[robot.robot_name])
timestamp = np.array(timestamp_dict[robot.robot_name])
len_min=min(len(timestamp),len(curve_exe),len(curve_exe_R))
curve_exe=curve_exe[:len_min]
timestamp=timestamp[:len_min]
curve_exe_R=curve_exe_R[:len_min]
np.savetxt('recorded_data/mocap_robot.csv',np.hstack((timestamp.reshape(-1, 1),curve_exe,R2w(curve_exe_R,np.eye(3)))),delimiter=',')

curve_exe = np.array(curve_exe_dict[positioner.robot_name])
curve_exe_R = np.array(curve_exe_R_dict[positioner.robot_name])
timestamp = np.array(timestamp_dict[positioner.robot_name])
len_min=min(len(timestamp),len(curve_exe),len(curve_exe_R))
curve_exe=curve_exe[:len_min]
timestamp=timestamp[:len_min]
curve_exe_R=curve_exe_R[:len_min]
np.savetxt('recorded_data/mocap_positioner.csv',np.hstack((timestamp.reshape(-1, 1),curve_exe,R2w(curve_exe_R,np.eye(3)))),delimiter=',')

# np.savetxt('joint_recording.csv',np.hstack((timestamp.reshape(-1, 1),job_line.reshape(-1, 1),joint_recording)),delimiter=',')