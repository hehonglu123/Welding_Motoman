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


mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=[1.341416193724337745e+03,1.907685083229250267e+03,1.592916090846681982e+03,1.022871664227330484e+03,9.802549195016306385e+02,4.547554799861444508e+02])
client=MotionProgramExecClient()

mp.MoveJ(q_init,1,0)
# mp.SetArc(True,cond_num=410)
mp.MoveL(q_end,5,0)
# mp.SetArc(False)

client.execute_motion_program(mp)
