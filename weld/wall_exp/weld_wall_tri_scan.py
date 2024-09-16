import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from scan_utils import *
from scanPathGen import *
from scanProcess import *
from weldRRSensor import *
from RobotRaconteur.Client import *
from copy import deepcopy
from pathlib import Path
import datetime


dataset='wall/'
sliced_alg='dense_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
vd_relative=15
feedrate_job=205

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
recorded_dir=data_dir+'weld_scan_job%i_v%i'%(feedrate_job,vd_relative)+formatted_time+'/'

datestr = input('Input date')
if datestr=='':
	pass
else:
	recorded_dir=data_dir+'weld_scan_job%i_v%i'%(feedrate_job,vd_relative)+datestr+'/'

waypoint_distance=5 	###waypoint separation
layer_height_num=int(1.5/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15,\
    base_marker_config_file='../config/MA2010_marker_config.yaml',tool_marker_config_file='../config/weldgun_marker_config.yaml')
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv',\
    base_marker_config_file='../config/MA1440_marker_config.yaml')
robot2_mti=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv',\
    base_marker_config_file='../config/MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv',\
    base_marker_config_file='../config/D500B_marker_config.yaml',tool_marker_config_file='../config/positioner_tcp_marker_config.yaml')

# Table_home_T = positioner.fwd(np.radians([-15,180]))
# T_S1TCP_R1Base = np.linalg.inv(np.matmul(positioner.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
# T_R1Base_S1TCP = np.linalg.inv(T_S1TCP_R1Base)

#### change base H to calibrated ones ####
robot_scan_base = robot.T_base_basemarker.inv()*robot2.T_base_basemarker
robot2.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
robot2_mti.base_H = H_from_RT(robot_scan_base.R,robot_scan_base.p)
positioner_base = robot.T_base_basemarker.inv()*positioner.T_base_basemarker
positioner.base_H = H_from_RT(positioner_base.R,positioner_base.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

client=MotionProgramExecClient()
ws=WeldSend(client)

## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
# weld state logging
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
cam_ser=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
mic_ser = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,cam_service=cam_ser,microphone_service=mic_ser)

# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")

###########################################layer welding############################################
layer_count = 22
num_layer_start=int(layer_count*layer_height_num)	###modify layer num here
num_layer_end=int((layer_count+10)*layer_height_num)
# q_prev=client.getJointAnglesDB(positioner.pulse2deg)
q_prev=np.array([-3.791544713877046391e-01,7.156749523014762637e-01,2.756772964158371586e-01,2.106493295914119712e-01,-7.865937103692784982e-01,-5.293956242391706368e-01])	###for motosim tests only

## for scanning ##
h_largest=0
Transz0_H=np.array([[ 9.99999340e-01, -1.74246690e-06,  1.14895353e-03,  1.40279850e-03],
 [-1.74246690e-06,  9.99995400e-01,  3.03312933e-03,  3.70325619e-03],
 [-1.14895353e-03, -3.03312933e-03,  9.99994740e-01,  1.22092938e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
##################

if num_layer_start<=1*layer_height_num:
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
else:
	num_sections=1


for layer in range(num_layer_start,num_layer_end,layer_height_num):
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

	num_sections_prev=num_sections
	num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer)+'_*.csv'))

	####################DETERMINE CURVE ORDER##############################################
	for x in range(0,num_sections,layer_width_num):
		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))
		if len(rob1_js)<2:
			continue
		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')

		
		lam1=calc_lam_js(rob1_js,robot)
		lam2=calc_lam_js(positioner_js,positioner)
		lam_relative=calc_lam_cs(curve_sliced_relative)

		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

		###find which end to start depending on how close to joint limit
		# if positioner.upper_limit[1]-q_prev[1]>q_prev[1]-positioner.lower_limit[1]:
		# 	breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
		# else:
		# 	breakpoints=np.linspace(len(rob1_js)-1,0,num=num_points_layer).astype(int)
		if layer_count%2==0:
			breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
		else:
			breakpoints=np.linspace(len(rob1_js)-1,0,num=num_points_layer).astype(int)

		s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

		###move to intermidieate waypoint for collision avoidance if multiple section
		if num_sections!=num_sections_prev:
			waypoint_pose=robot.fwd(rob1_js[breakpoints[0]])
			waypoint_pose.p[-1]+=50
			q1=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js[breakpoints[0]])[0]
			q2=positioner_js[breakpoints[0]]
			ws.jog_dual(robot,positioner,q1,q2)

		q1_all=[rob1_js[breakpoints[0]]]
		q2_all=[rob2_js[breakpoints[0]]]
		positioner_all=[positioner_js[breakpoints[0]]]
		v1_all=[1]
		v2_all=[10]
		primitives=['movej']
		for j in range(1,len(breakpoints)):
			q1_all.append(rob1_js[breakpoints[j]])
			q2_all.append(rob2_js[breakpoints[j]])
			positioner_all.append(positioner_js[breakpoints[j]])
			v1_all.append(max(s1_all[j-1],0.1))
			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
			primitives.append('movel')

		q_prev=positioner_js[breakpoints[-1]]
	
		###robot1=robot2 speed tests
		ws.jog_tri(robot,positioner,robot2,q1_all[0],positioner_all[0],q2_all[0],v=3)	#jog to starting positioner first
		rr_sensors.start_all_sensors()
		timestamp_robot,joint_recording,job_line,_=ws.weld_segment_tri(primitives,robot,positioner,robot2,q1_all,positioner_all,q2_all,v1_all,v1_all,cond_all=[feedrate_job],arc=True)
		rr_sensors.stop_all_sensors()
		

		Path(recorded_dir).mkdir(exist_ok=True)
		layer_data_dir=recorded_dir+'layer_'+str(layer)+'/'
		Path(layer_data_dir).mkdir(exist_ok=True)
		np.savetxt(layer_data_dir+'joint_recording.csv',np.hstack((timestamp_robot.reshape(-1, 1),job_line.reshape(-1, 1),joint_recording)),delimiter=',')
		rr_sensors.save_all_sensors(layer_data_dir)

	######## scanning ##########
	ws.jog_single(robot,np.array([-8.135922244967886741e-01,7.096733413840118354e-01,3.570605700073341549e-01,1.795958126158156976e-01,-8.661845429601626734e-01,-4.639865155930678053e-01]),v=3)
	for x in range(0,num_sections,layer_width_num):
		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
	# 	curve_sliced_relative=[np.array([  -3.19476273e+01,  1.72700000e+00,  4.25800473e+01,  1.55554573e-04,
    #    -6.31394918e-20, -9.99881509e-01]), np.array([ 3.30446707e+01,  1.72700000e+00,  4.25800473e+01,  1.55554573e-04,
    #    -6.31394918e-20, -9.99881509e-01])]

		scan_speed=10 # scanning speed (mm/sec)
		scan_stand_off_d = 95 ## mm
		Rz_angle = np.radians(0) # point direction w.r.t welds
		Ry_angle = np.radians(0) # rotate in y a bit
		bounds_theta = np.radians(1) ## circular motion at start and end
		all_scan_angle = np.radians([0]) ## scan angle
		q_init_table=np.radians([-15,20]) ## init table
		save_output_points = True
		### scanning path module
		spg = ScanPathGen(robot2_mti,positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta)
		mti_Rpath = np.array([[ -1.,0.,0.],   
					[ 0.,1.,0.],
					[0.,0.,-1.]])
		# mti_Rpath = np.eye(3)
		scan_p,scan_R,q_out1,q_out2=spg.gen_scan_path([curve_sliced_relative],[0],all_scan_angle,\
                            solve_js_method=0,q_init_table=q_init_table,R_path=mti_Rpath,scan_path_dir=None)
		q_bp1,q_bp2,s1_all,s2_all=spg.gen_motion_program(q_out1,q_out2,scan_p,scan_speed,init_sync_move=0)

		# to_start_speed=7
		# mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2.pulse2deg,pulse2deg_2=positioner.pulse2deg)
		# target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
		# mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
		# robot_client.execute_motion_program(mp)
		ws.jog_dual(robot2_mti,positioner,q_bp1[0][0],q_bp2[0][0],v=3)

		scan_motion_scan_st = time.time()

		## motion start
		mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2_mti.pulse2deg,pulse2deg_2=positioner.pulse2deg)
		# calibration motion
		target2=['MOVJ',np.degrees(q_bp2[1][0]),s2_all[0]]
		mp.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
		# routine motion
		for path_i in range(2,len(q_bp1)-1):
			target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
			mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
		target2=['MOVJ',np.degrees(q_bp2[-1][0]),s2_all[-1]]
		mp.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)

		ws.client.execute_motion_program_nonblocking(mp)
		###streaming
		ws.client.StartStreaming()
		start_time=time.time()
		state_flag=0
		joint_recording=[]
		robot_stamps=[]
		mti_recording=[]
		r_pulse2deg = np.append(robot2_mti.pulse2deg,positioner.pulse2deg)
		while True:
			if state_flag & 0x08 == 0 and time.time()-start_time>1.:
				print("break")
				break
			res, data = ws.client.receive_from_robot(0.01)
			if res:
				joint_angle=np.radians(np.divide(np.array(data[26:34]),r_pulse2deg))
				state_flag=data[16]
				joint_recording.append(joint_angle)
				timestamp=data[0]+data[1]*1e-9
				robot_stamps.append(timestamp)
				###MTI scans YZ point from tool frame
				mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
		robot_stamps=np.array(robot_stamps)-robot_stamps[0]
		ws.client.servoMH(False)
		mti_recording=np.array(mti_recording)
		q_out_exe=joint_recording

		# move robot to home
		q2=np.array([-2.704468035842202411e-01,9.330144509521144380e-01,-3.376213326477142118e-01,-1.228474839331376023e+00,-1.395732731587226549e+00,2.846773448527548656e+00])
		q3=np.radians([-15,90])
		# ## move to home
		# to_home_speed=7
		# mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot2_mti.pulse2deg,pulse2deg_2=positioner.pulse2deg)
		# target2=['MOVJ',q3,to_home_speed]
		# mp.MoveJ(q2, to_home_speed, 0, target2=target2)
		# robot_client.execute_motion_program(mp)
		ws.jog_dual(robot2_mti,positioner,q2,q3,v=3)
		#####################

		print("Total exe len:",len(q_out_exe))
		out_scan_dir = layer_data_dir+'scans/'
		## save traj
		Path(out_scan_dir).mkdir(exist_ok=True)
		# save poses
		np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
		np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
		with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
			pickle.dump(mti_recording, file)
		print('Total scans:',len(mti_recording))
  
		curve_x_start = deepcopy(curve_sliced_relative[0][0])
		curve_x_end = deepcopy(curve_sliced_relative[-1][0])
		# Transz0_H=np.array([[ 9.99974559e-01, -7.29664987e-06, -7.13309345e-03, -1.06461758e-02],
		#                     [-7.29664987e-06,  9.99997907e-01, -2.04583032e-03, -3.05341146e-03],
		#                     [ 7.13309345e-03,  2.04583032e-03,  9.99972466e-01,  1.49246365e+00],
		#                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
		z_height_start=h_largest-5
		crop_extend=10
		crop_min=(curve_x_start-crop_extend,-30,-10)
		crop_max=(curve_x_end+crop_extend,30,z_height_start+30)
		crop_h_min=(curve_x_start-crop_extend,-20,-10)
		crop_h_max=(curve_x_end+crop_extend,20,z_height_start+30)

		try:
			scan_process = ScanProcess(robot2_mti,positioner)
			pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
			pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
												min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
			profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
			print("Transz0_H:",Transz0_H)

			save_output_points=True
			if save_output_points:
				o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)
				np.save(out_scan_dir+'height_profile.npy',profile_height)
			# visualize_pcd([pcd])
			plt.scatter(profile_height[:,0],profile_height[:,1])
			plt.show()
			h_largest=np.max(profile_height[:,1])
			h_mean=np.mean(profile_height[:,1])
			print("H largest:",h_largest)
			print("H mean:",h_mean)
		except Exception as e:
			print(e)
			h_largest = curve_sliced_relative[0][2]+1.8
	
	layer_count+=1