import sys, glob
from pathlib import Path

sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from WeldSend import *
from traj_manipulation import *


dataset='funnel/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/'
Path(recorded_dir).mkdir(exist_ok=True)
waypoint_distance=5
layer_width_num=int(4/slicing_meta['line_resolution'])
layer_height_num=int(1.3/slicing_meta['line_resolution'])

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
		pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

# ########################################################RR Microphone########################################################
# microphone = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
# ########################################################RR CURRENT########################################################
# current_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
# ########################################################RR FRONIUS########################################################
# fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
# fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s
# ##########################################SENSORS LOGGIGN########################################################
# rr_sensors = WeldRRSensor(weld_service=fronius_sub,cam_service=None,microphone_service=microphone,current_service=current_sub)


###set up control parameters
job_offset=200 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
nominal_feedrate=200
nominal_vd_relative=12
nominal_wire_length=25 #pixels
nominal_temp_below=500
base_feedrate_cmd=300
base_vd=3
feedrate_cmd=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=200
feedrate_max=300
nominal_slice_increment=int(1.3/slicing_meta['line_resolution'])
slice_inc_gain=3.
vd_max=12


###########################################layer welding############################################
client=MotionProgramExecClient()
ws=WeldSend(client)
q1_all=[]
q2_all=[]
v1_all=[]
v2_all=[]
cond_all=[]
primitives=[]
arcon_set=False
layer_start=178
layer_end=190
num_layer_start=int(layer_start*layer_height_num)
num_layer_end=int(layer_end*layer_height_num)

q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev=[0,0]


####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[[]]*layer_start
rob2_js_all_slices=[[]]*layer_start
positioner_js_all_slices=[[]]*layer_start
for i in range(num_layer_start,num_layer_end,nominal_slice_increment):
	rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	rob2_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))


for slice_num in range(num_layer_start,num_layer_end,nominal_slice_increment):

	####################DETERMINE CURVE ORDER##############################################
	x=0
	rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
	rob2_js=copy.deepcopy(rob2_js_all_slices[slice_num])
	positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
		continue
	
	###TRJAECTORY WARPING
	if slice_num>layer_start:
		rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-nominal_slice_increment])
		rob2_js_prev=copy.deepcopy(rob2_js_all_slices[slice_num-nominal_slice_increment])
		positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-nominal_slice_increment])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
	if slice_num<layer_end-nominal_slice_increment:
		rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+nominal_slice_increment])
		rob2_js_next=copy.deepcopy(rob2_js_all_slices[slice_num+nominal_slice_increment])
		positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+nominal_slice_increment])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
			
	
		
	lam_relative=calc_lam_cs(curve_sliced_relative)
	lam1=calc_lam_js(rob1_js,robot)
	lam2=calc_lam_js(positioner_js,positioner)
	num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
	breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)

	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

	###find closest %2pi
	num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
	positioner_js+=num2p*2*np.pi
	###no need for acron/off when spiral, positioner not moving at all
	if not arcon_set:
		arcon_set=True
		q1_all.append(rob1_js[breakpoints[0]])
		q2_all.append(positioner_js[breakpoints[0]])
		v1_all.append(20)
		v2_all.append(5)
		cond_all.append(0)
		primitives.append('movej')


	

	q1_all.extend(rob1_js[breakpoints[1:]].tolist())
	q2_all.extend(positioner_js[breakpoints[1:]].tolist())
	v1_all.extend(s1_all)
	cond_all.extend([int(feedrate_cmd/10)+job_offset]*(num_points_layer-1))
	primitives.extend(['movel']*(num_points_layer-1))

	if slice_num<30:
		v2_all.extend([0]*len(s2_all))
	else:
		for j in range(1,len(breakpoints)):
			positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
			v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
	
	feedrate_cmd-=20
	vd_relative+=2
	vd_relative=min(vd_max,vd_relative)
	feedrate_cmd=max(feedrate_cmd,feedrate_min)
	print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)

	q_prev=copy.deepcopy(positioner_js[-1])

timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all,arc=True)
