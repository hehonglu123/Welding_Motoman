import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
p_start=np.array([1615,-860,-260])
p_end=np.array([1615,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
q_init=np.degrees(robot.inv(p_start,R,q_seed)[0])
q_end=np.degrees(robot.inv(p_end,R,q_seed)[0])

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)

base_layer_height=2
layer_height=0.8

# for i in range(2):
# 	if i%2==0:
# 		p1=p_start+np.array([0,0,i*base_layer_height])
# 		p2=p_end+np.array([0,0,i*base_layer_height])
# 	else:
# 		p1=p_end+np.array([0,0,i*base_layer_height])
# 		p2=p_start+np.array([0,0,i*base_layer_height])

# 	q_init=np.degrees(robot.inv(p1,R,q_seed)[0])
# 	q_end=np.degrees(robot.inv(p2,R,q_seed)[0])
# 	client.MoveJ(q_init,1,0)
# 	client.SetArc(True,cond_num=301)
# 	client.MoveL(q_end,5,0)
# 	client.SetArc(False)
# 	client.execute_motion_program()

for i in range(10,12):
	if i%2==0:
		p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
		p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
	else:
		p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
		p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

	q_init=np.degrees(robot.inv(p1,R,q_seed)[0])
	q_end=np.degrees(robot.inv(p2,R,q_seed)[0])
	client.MoveJ(q_init,1,0)
	client.SetArc(True,cond_num=300)
	client.MoveL(q_end,10,0)
	client.SetArc(False)
	client.execute_motion_program()
