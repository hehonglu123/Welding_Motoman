import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
p_start=np.array([1650,-860,-250])
p_end=np.array([1650,-760,-250])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
q_init=np.degrees(robot.inv(p_start,R,q_seed)[0])
q_end=np.degrees(robot.inv(p_end,R,q_seed)[0])

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)


client.MoveJ(q_init,1,0)
client.SetArc(True,cond_num=301)
client.MoveL(q_end,5,0)
client.SetArc(False)

client.execute_motion_program()
