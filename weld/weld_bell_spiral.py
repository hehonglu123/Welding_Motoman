import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *
from traj_manipulation import *

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

dataset='bell/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/bell_ER316L/'

waypoint_distance=5 	###waypoint separation
layer_height_num=int(1.2/slicing_meta['line_resolution'])
layer_width_num=int(3/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose_mocap.csv')

client=MotionProgramExecClient()
ws=WeldSend(client)

client=MotionProgramExecClient()
ws=WeldSend(client)
q1_all=[]
q2_all=[]
v1_all=[]
v2_all=[]
cond_all=[]
primitives=[]
arcon_set=False


q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev=[0,0]


###set up control parameters
job_offset=200 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
nominal_feedrate=170
nominal_vd_relative=5
nominal_wire_length=25 #pixels
nominal_temp_below=500
base_feedrate_cmd=300
base_vd=8
feedrate_cmd=nominal_feedrate
vd_relative=nominal_vd_relative
feedrate_gain=0.5
feedrate_min=100
feedrate_max=300
nominal_slice_increment=int(1.0/slicing_meta['line_resolution'])
slice_inc_gain=3.
vd_max=9
feedrate_cmd_adjustment=-50
vd_relative_adjustment=2

# ###set up control parameters, no baselayer needed for 70S6!!!!
# job_offset=300 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
# nominal_feedrate=250
# nominal_vd_relative=4
# nominal_wire_length=25 #pixels
# nominal_temp_below=500
# base_feedrate_cmd=300
# base_vd=8
# feedrate_cmd=nominal_feedrate
# vd_relative=nominal_vd_relative
# feedrate_gain=0.5
# feedrate_min=100
# feedrate_max=150
# nominal_slice_increment=int(1.1/slicing_meta['line_resolution'])
# slice_inc_gain=3.
# vd_max=6
# feedrate_cmd_adjustment=-100
# vd_relative_adjustment=2


# ###set up control parameters, no baselayer needed for 316L!!!!
# job_offset=400 		###200 for Aluminum ER4043, 300 for Steel Alloy ER70S-6, 400 for Stainless Steel ER316L
# nominal_feedrate=80
# nominal_vd_relative=2
# nominal_wire_length=25 #pixels
# nominal_temp_below=500
# base_feedrate_cmd=300
# base_vd=8
# feedrate_cmd=nominal_feedrate
# vd_relative=nominal_vd_relative
# feedrate_gain=0.5
# feedrate_min=80
# feedrate_max=300
# nominal_slice_increment=int(1.1/slicing_meta['line_resolution'])
# slice_inc_gain=3.
# vd_max=8
# feedrate_cmd_adjustment=0.7
# vd_relative_adjustment=1


######################################################BASE LAYER##########################################################################################
# slice_num=0
# num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_*.csv'))
# mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)

# try:
# 	for x in range(0,num_sections,layer_width_num):
# 		rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 		rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 		positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 		curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 		if positioner_js.shape==(2,) and rob1_js.shape==(6,):
# 			continue
# 		if x>0:
# 			rob1_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 			rob2_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 			positioner_js_prev=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 			rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
# 			if x<num_sections-layer_width_num:
# 				rob1_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 				rob2_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 				positioner_js_next=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 				rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
			
# 		lam_relative=calc_lam_cs(curve_sliced_relative)
# 		lam1=calc_lam_js(rob1_js,robot)
# 		lam2=calc_lam_js(positioner_js,positioner)
# 		num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))
# 		breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)

# 		s1_all,s2_all=calc_individual_speed(base_vd,lam1,lam2,lam_relative,breakpoints)

# 		###find closest %2pi
# 		num2p=np.round((q_prev-positioner_js[0])/(2*np.pi))
# 		positioner_js+=num2p*2*np.pi
# 		###no need for acron/off when spiral, positioner not moving at all
# 		if not arcon_set:
# 			arcon_set=True
# 			q1_all.append(rob1_js[breakpoints[0]])
# 			q2_all.append(positioner_js[breakpoints[0]])
# 			v1_all.append(20)
# 			v2_all.append(5)
# 			cond_all.append(0)
# 			primitives.append('movej')


		

# 		q1_all.extend(rob1_js[breakpoints[1:]].tolist())
# 		q2_all.extend(positioner_js[breakpoints[1:]].tolist())
# 		v1_all.extend(s1_all)
# 		v2_all.extend([0]*len(s1_all))
# 		cond_all.extend([int(base_feedrate_cmd/10)+job_offset]*(num_points_layer-1))
# 		primitives.extend(['movel']*(num_points_layer-1))

# 	q_prev=positioner_js[breakpoints[-1]]
# 	timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all,arc=True)

# except:
# 	traceback.print_exc()


##################################################### LAYER Welding##########################################################################################
###PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]
lam_relative_all_slices=[]
lam_relative_dense_all_slices=[]
for i in range(0,slicing_meta['num_layers']):
	rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
	rob2_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','))
	positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))
	# rob1_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	# rob2_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	# positioner_js_all_slices.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','),axis=0))
	
print("PRELOAD FINISHED")

num_layer_start=int(1*nominal_slice_increment)
num_layer_end=slicing_meta['num_layers']
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
	if slice_num>num_layer_start:
		rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-nominal_slice_increment])
		rob2_js_prev=copy.deepcopy(rob2_js_all_slices[slice_num-nominal_slice_increment])
		positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-nominal_slice_increment])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
	if slice_num<num_layer_end-nominal_slice_increment:
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
	# s1_all=[0.1]*len(s1_all)
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
	# v1_all.extend(s1_all)
	v1_all.extend([1]*len(s1_all))
	cond_all.extend([int(feedrate_cmd/10)+job_offset]*(num_points_layer-1))
	primitives.extend(['movel']*(num_points_layer-1))

	for j in range(1,len(breakpoints)):
		positioner_w=vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
		v2_all.append(min(100,100*positioner_w/positioner.joint_vel_limit[1]))
	
	feedrate_cmd+=feedrate_cmd_adjustment
	vd_relative+=vd_relative_adjustment
	vd_relative=min(vd_max,vd_relative)
	feedrate_cmd=min(feedrate_max,max(feedrate_cmd,feedrate_min))
	print('FEEDRATE: ',feedrate_cmd,'VD: ',vd_relative)

	q_prev=copy.deepcopy(positioner_js[-1])

timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all,arc=True)
