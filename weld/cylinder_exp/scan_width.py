from copy import deepcopy
from pathlib import Path
import pickle, sys, time, datetime, traceback, glob
sys.path.append('../../scan/scan_tools/')

from motoman_def import *
from scan_utils import *
from robotics_utils import *
from weldRRSensor import *
from WeldSend import *
from dual_robot import *
from lambda_calc import *
from redundancy_resolution_dual import redundancy_resolution_dual
from dx200_motion_program_exec_client import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import numpy as np

def scan_path_gen(curve_sliced_relative,scan_stand_off_d):
	scan_R=[]
	scan_p=[]
	for i in range(len(curve_sliced_relative)):
		# get scan R
		Rz = curve_sliced_relative[i,3:]
		Rz = Rz/np.linalg.norm(Rz)
		if i<len(curve_sliced_relative)-1:
			Ry = curve_sliced_relative[i+1,:3]-curve_sliced_relative[i,:3]
		else:
			Ry = curve_sliced_relative[i,:3]-curve_sliced_relative[i-1,:3]
		Ry = (Ry-np.dot(Ry,Rz)*Rz)
		Ry=Ry/np.linalg.norm(Ry)

		Rx = np.cross(Ry,Rz)
		Rx = Rx/np.linalg.norm(Rx)
		scan_R.append(np.array([Rx,Ry,Rz]).T)
		scan_p.append(curve_sliced_relative[i,:3] - Rz*scan_stand_off_d)

	return np.array(scan_p),np.array(scan_R)


def main():
	dataset='tube/'
	sliced_alg='dense_slice/'
	data_dir='../../../geometry_data/'+dataset+sliced_alg
	with open(data_dir+'slicing.yml', 'r') as file:
		slicing_meta = yaml.safe_load(file)

	waypoint_distance=5 	###waypoint separation
	

	
	##############################################################Robot####################################################################
	###robot kinematics def
	config_dir='../../config/'
	robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
		pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
	robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
		pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
	positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose_mocap.csv')


	###########################################################Chose scanning Curve#######################################################
	num_layers=40
	layer_height=1.1
	nominal_slice_increment=int(layer_height/slicing_meta['line_resolution'])
	slice_num=num_layers*nominal_slice_increment
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice%i_0.csv'%slice_num,delimiter=',')

	###########################################################Robot Control#######################################################
	client=MotionProgramExecClient()
	ws=WeldSend(client)

	##############################################################SENSORS####################################################################
	mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
	mti_client.setExposureTime("25")


	###########################################################Generate Scanning Path###########################################################
	r2_mid = np.radians([-7.48,27.4,-13.9,-50.7,-43,63.1])

	### scan parameters
	scan_speed=10 # scanning speed (mm/sec)
	scan_stand_off_d = 95 ## mm
	positioner_init=client.getJointAnglesDB(positioner.pulse2deg) ## init table
	# positioner_init=np.radians([-15,200]) ## init table
	scan_p,scan_R=scan_path_gen(curve_sliced_relative,scan_stand_off_d)
	

	positioner_pose=positioner.fwd(positioner_init,world=True)
	scan_p_world=[]
	scan_R_world=[]
	for i in range(len(scan_p)):
		scan_p_world.append(positioner_pose.p+np.dot(positioner_pose.R,scan_p[i]))
		scan_R_world.append(positioner_pose.R@scan_R[i])
	scan_p_world=np.array(scan_p_world)
	scan_R_world=np.array(scan_R_world)

	###check the generated path
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# # ax.plot(scan_p_world[:,0],scan_p_world[:,1],scan_p_world[:,2])
	# # ax.quiver(scan_p_world[:,0],scan_p_world[:,1],scan_p_world[:,2],scan_R_world[:,0,-1],scan_R_world[:,1,-1],scan_R_world[:,2,-1],color='r')
	# ax.plot(scan_p[:,0],scan_p[:,1],scan_p[:,2])
	# ax.quiver(scan_p[:,0],scan_p[:,1],scan_p[:,2],scan_R[:,0,-1],scan_R[:,1,-1],scan_R[:,2,-1],color='r')
	# set_axes_equal(ax)
	# plt.show()

	
	rrd=redundancy_resolution_dual(robot_scan,positioner,scan_p,scan_R)
	q_out1, q_out2=rrd.dual_arm_6dof_stepwise(copy.deepcopy(r2_mid),positioner_init,w1=0.1,w2=0.01)	#more weights on robot_scan to make it move slower

	lam1=calc_lam_js(q_out1,robot_scan)
	lam2=calc_lam_js(q_out2,positioner)
	lam_relative=calc_lam_cs(scan_p)
	q_bp1,q_bp2,s1_all,s2_all=gen_motion_program_dual(lam1,lam2,lam_relative,q_out1,q_out2,v=10)

	## move to start
	ws.jog_single(robot,[0.,0.,np.pi/6,0.,0.,0.])
	ws.jog_dual(robot_scan,positioner,r2_mid,q_bp2[0])
	ws.jog_dual(robot_scan,positioner,q_bp1[0],q_bp2[0])

	input("Press Enter to start moving and scanning")

	## motion start
	mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
	# calibration motion
	target2=['MOVJ',np.degrees(q_bp2[1]),s2_all[0]]
	mp.MoveL(np.degrees(q_bp1[1]), scan_speed, 0, target2=target2)
	# routine motion
	for path_i in range(2,len(q_bp1)-1):
		target2=['MOVJ',np.degrees(q_bp2[path_i]),s2_all[path_i]]
		mp.MoveL(np.degrees(q_bp1[path_i]), s1_all[path_i], target2=target2)
	target2=['MOVJ',np.degrees(q_bp2[-1]),s2_all[-1]]
	mp.MoveL(np.degrees(q_bp1[-1]), s1_all[-1], 0, target2=target2)

	ws.client.execute_motion_program_nonblocking(mp)
	###streaming
	ws.client.StartStreaming()
	start_time=time.time()
	state_flag=0
	joint_recording=[]
	robot_ts=[]
	global_ts=[]
	mti_recording=[]
	while True:
		if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
			break 
		res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
		if res:
			joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
			state_flag=fb_data.controller_flags
			joint_recording.append(joint_angle)
			timestamp=fb_data.time
			robot_ts.append(timestamp)
			global_ts.append(time.perf_counter())
			###MTI scans YZ point from tool frame
			mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
	ws.client.servoMH(False)
	
	mti_recording=np.array(mti_recording)
	joint_recording=np.array(joint_recording)


	ws.jog_single(robot_scan,r2_mid)


	###########################################################Save Scanning Data###########################################################
	out_scan_dir = 'scans/'
	## save traj
	Path(out_scan_dir).mkdir(exist_ok=True)
	# save poses
	np.savetxt(out_scan_dir + 'scan_js_exe.csv',np.hstack((np.array(global_ts).reshape(-1,1),np.array(robot_ts).reshape(-1,1),joint_recording)),delimiter=',')
	with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
		pickle.dump(mti_recording, file)


if __name__ == '__main__':
	main()