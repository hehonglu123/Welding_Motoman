import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from traj_manipulation import *
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


logging=False
if logging:
	sub=RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
	obj = sub.GetDefaultClientWait(3)      #connect, timeout=30s
	welder_state_sub=sub.SubscribeWire("welder_state")
	welder_state_sub.WireValueChanged += wire_cb


dataset='diamond/'
sliced_alg='cont_sections/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
recorded_dir='recorded_data/cup_ER316L/'

waypoint_distance=4	###waypoint separation
layer_height_num=int(1.8/slicing_meta['line_resolution'])


robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
		pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

layer_height_num=int(1.8/slicing_meta['line_resolution'])
layer_width_num=int(4/slicing_meta['line_resolution'])


client=MotionProgramExecClient()
ws=WeldSend(client)

####PRELOAD ALL SLICES TO SAVE INPROCESS TIME
rob1_js_all_slices=[]
rob2_js_all_slices=[]
positioner_js_all_slices=[]

for i in range(0,slicing_meta['num_layers']-1):
	rob1_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_0.csv',delimiter=','))
	rob2_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_0.csv',delimiter=','))
	positioner_js_all_slices.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_0.csv',delimiter=','))




######################################################BASE LAYER##########################################################################################
# base_feedrate_cmd=300
# base_vd_relative=10
# slice_num=0
# num_sections=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_*.csv'))
# q1_all=[]
# q2_all=[]
# v1_all=[]
# v2_all=[]
# primitives=[]
# q_prev=client.getJointAnglesDB(positioner.pulse2deg)

# for x in range(0,num_sections,layer_width_num):
# 	rob1_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')
# 	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
# 		continue
# 	if x>0:
# 		rob1_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		rob2_js_prev=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		positioner_js_prev=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x-layer_width_num)+'.csv',delimiter=',')
# 		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
# 		if x<num_sections-layer_width_num:
# 			rob1_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			rob2_js_next=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			positioner_js_next=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(slice_num)+'_'+str(x+layer_width_num)+'.csv',delimiter=',')
# 			rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
	
	

# 	lam1=calc_lam_js(rob1_js,robot)
# 	lam2=calc_lam_js(positioner_js,positioner)
# 	lam_relative=calc_lam_cs(curve_sliced_relative)

# 	num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

# 	breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
   
# 	s1_all,s2_all=calc_individual_speed(base_vd_relative,lam1,lam2,lam_relative,breakpoints)

# 	###find closest %2pi
# 	num2p=np.round((q_prev[1]-positioner_js[0,1])/(2*np.pi))
# 	positioner_js[:,1]=positioner_js[:,1]+num2p*2*np.pi
# 	if len(q1_all)==0:
# 		q1_all.extend([rob1_js[breakpoints[0]]])
# 		q2_all.extend([positioner_js[breakpoints[0]]])
# 		v1_all.extend([1])
# 		v2_all.extend([10])
# 		primitives.extend(['movej'])

# 	for j in range(1,len(breakpoints)):
# 		q1_all.extend([rob1_js[breakpoints[j]]])
# 		q2_all.extend([positioner_js[breakpoints[j]]])
# 		v1_all.extend([max(s1_all[j-1],0.1)])
# 		positioner_w=8*base_vd_relative/np.linalg.norm(curve_sliced_relative[breakpoints[j]][:2])
# 		v2_all.extend([min(100,100*positioner_w/positioner.joint_vel_limit[1])])
# 		# v2_all.append(50)
# 		primitives.extend(['movel'])
		
# timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(base_feedrate_cmd/10+200)],arc=True)



###########################################layer welding############################################
vd_relative=5
feedrate=100
layer_start=38
layers2weld=1
layer_counts=layer_start
num_layer_start=int(layer_start*layer_height_num)	###modify layer num here
num_layer_end=int((layer_start+layers2weld)*layer_height_num)

q_prev=client.getJointAnglesDB(positioner.pulse2deg)
# q_prev=np.array([9.53E-02,-2.71E+00])	###for motosim tests only

q1_all=[]
q2_all=[]
v1_all=[]
v2_all=[]
primitives=[]
primitives.extend(['movej'])
v1_all.extend([1])
v2_all.extend([10])
for slice_num in range(num_layer_start,num_layer_end,layer_height_num):
	####################DETERMINE CURVE ORDER##############################################
	x=0
	rob1_js=copy.deepcopy(rob1_js_all_slices[slice_num])
	rob2_js=copy.deepcopy(rob2_js_all_slices[slice_num])
	positioner_js=copy.deepcopy(positioner_js_all_slices[slice_num])
	if positioner_js.shape==(2,) and rob1_js.shape==(6,):
		continue
	curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(slice_num)+'_'+str(x)+'.csv',delimiter=',')

	
	lam1=calc_lam_js(rob1_js,robot)
	lam2=calc_lam_js(positioner_js,positioner)
	lam_relative=calc_lam_cs(curve_sliced_relative)

	num_points_layer=max(2,int(lam_relative[-1]/waypoint_distance))

	breakpoints=np.linspace(0,len(rob1_js)-1,num=num_points_layer).astype(int)
   
	s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)

	###for diamond shape only due to discontinuity at start/end
	s1_all=np.insert(s1_all,0,5+np.max(s1_all))
	breakpoints[-1]=breakpoints[-1]-int((breakpoints[-1]-breakpoints[-2])/2)
	if len(q1_all)==0:
		breakpoints[0]=breakpoints[0]+int((breakpoints[1]-breakpoints[0])/2)


	###TRJAECTORY WARPING
	if layer_counts>layer_start:
		rob1_js_prev=copy.deepcopy(rob1_js_all_slices[slice_num-layer_height_num])
		rob2_js_prev=copy.deepcopy(rob2_js_all_slices[slice_num-layer_height_num])
		positioner_js_prev=copy.deepcopy(positioner_js_all_slices[slice_num-layer_height_num])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_prev,rob2_js_prev,positioner_js_prev,reversed=True)
	if layer_counts<layer_start+layers2weld-1:
		rob1_js_next=copy.deepcopy(rob1_js_all_slices[slice_num+layer_height_num])
		rob2_js_next=copy.deepcopy(rob2_js_all_slices[slice_num+layer_height_num])
		positioner_js_next=copy.deepcopy(positioner_js_all_slices[slice_num+layer_height_num])
		rob1_js,rob2_js,positioner_js=warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_next,rob2_js_next,positioner_js_next,reversed=False)
	###find closest %2pi
	num2p=np.round((q_prev[1]-positioner_js[0,1])/(2*np.pi))
	positioner_js[:,1]=positioner_js[:,1]+num2p*2*np.pi

	###move to intermidieate waypoint for collision avoidance if multiple section
	if abs(q_prev[0]-positioner_js[0,0])>0.1:
		waypoint_pose=robot.fwd(rob1_js[breakpoints[0]])
		waypoint_pose.p[-1]+=50
		q1=robot.inv(waypoint_pose.p,waypoint_pose.R,rob1_js[breakpoints[0]])[0]
		q2=positioner_js[breakpoints[0]]
		ws.jog_dual(robot,positioner,q1,q2)


	if len(q1_all)==0:
		q1_all.extend([rob1_js[breakpoints[0]]])
		q2_all.extend([positioner_js[breakpoints[0]]])
	
		
	for j in range(0,len(breakpoints)):
		q1_all.extend([rob1_js[breakpoints[j]]])
		q2_all.extend([positioner_js[breakpoints[j]]])
		v1=max(s1_all[j],0.1)
		if v1>2*vd_relative:
			v1_all.extend([v1/2])
		else:
			v1_all.extend([v1])
		v2_all.append(100)
		primitives.extend(['movel'])

	q_prev=positioner_js[-1]
	layer_counts+=1

timestamp_robot,joint_recording,job_line,_=ws.weld_segment_dual(primitives,robot,positioner,q1_all,q2_all,v1_all,v2_all,cond_all=[int(feedrate/10+200)],arc=True)


