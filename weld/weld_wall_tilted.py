import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *


q_positioner_baselayer=np.radians([-15,180])
tilt_angle=np.radians(30)

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

pose_positioner_baselayer=positioner.fwd(q_positioner_baselayer,world=True)

R=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])
p_start=np.array([-50,0,0])
p_end=np.array([50,0,0])


q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()

base_layer_height=2
layer_height=0.8

# for i in range(1,2):
# 	if i==0:
# 		p_temp=transform_curve(np.array([p_start+np.array([0,0,50])]),H_from_RT(pose_positioner_baselayer.R,pose_positioner_baselayer.p))[0]
# 		q_temp=np.degrees(robot.inv(p_temp,R,q_seed)[0])
		
# 		mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
# 		target2=['MOVJ',np.degrees(q_positioner_baselayer),1]
# 		mp.MoveJ(q_temp, 1,target2=target2)

# 		client.execute_motion_program(mp)


# 	mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg, tool_num=12)
# 	if i%2==0:
# 		p1=p_start+np.array([0,0,i*base_layer_height])
# 		p2=p_end+np.array([0,0,i*base_layer_height])
# 	else:
# 		p1=p_end+np.array([0,0,i*base_layer_height])
# 		p2=p_start+np.array([0,0,i*base_layer_height])
	
# 	start_end=transform_curve(np.array([p1,p2]),H_from_RT(pose_positioner_baselayer.R,pose_positioner_baselayer.p))

# 	p1=start_end[0]
# 	p2=start_end[1]

# 	q_init=np.degrees(robot.inv(p1,R,q_seed)[0])
# 	q_end=np.degrees(robot.inv(p2,R,q_seed)[0])
# 	mp.MoveJ(q_init,1,0)
# 	mp.setArc(True,cond_num=410)
# 	mp.MoveL(q_end,5,0)
# 	mp.setArc(False)
# 	client.execute_motion_program(mp)

for i in range(7,9):

	if i%2==0:
		p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
		p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		q_positioner_tilted=q_positioner_baselayer+np.array([tilt_angle,0])
	else:
		p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
		q_positioner_tilted=q_positioner_baselayer+np.array([tilt_angle,-np.pi])

	pose_positioner_tilted=positioner.fwd(q_positioner_tilted,world=True)
	p_temp=transform_curve(np.array([p1+np.array([0,0,70])]),H_from_RT(pose_positioner_tilted.R,pose_positioner_tilted.p))[0]
	q_temp=np.degrees(robot.inv(p_temp,R,q_seed)[0])
	
	mp=MotionProgram(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg, tool_num = 12)
	target2=['MOVJ',np.degrees(q_positioner_tilted),1]
	mp.MoveJ(q_temp, 1,target2=target2)
	client.execute_motion_program(mp)

	mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg, tool_num=12)
	start_end=transform_curve(np.array([p1,p2]),H_from_RT(pose_positioner_tilted.R,pose_positioner_tilted.p))
	p1=start_end[0]
	p2=start_end[1]

	q_init=np.degrees(robot.inv(p1,R,q_seed)[0])
	q_end=np.degrees(robot.inv(p2,R,q_seed)[0])
	mp.MoveJ(q_init,1,0)
	mp.setArc(True,cond_num=402)
	mp.MoveL(q_end,10,0)
	mp.setArc(False)
	client.execute_motion_program(mp)
