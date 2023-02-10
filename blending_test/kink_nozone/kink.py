import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])

p_start=np.array([1648,-1290,-231])
p_mid=np.array([1648-50/np.sqrt(3),-1240,-231])
p_end=np.array([1648,-1190,-231])
p_dense=np.linspace(p_start,p_end,num=50)

qseed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
q_start=robot.inv(p_start, R, qseed)[0]
q_mid=robot.inv(p_mid, R, qseed)[0]
q_end=robot.inv(p_end, R, qseed)[0]


def jog2start():
	client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
	client.ProgStart(r"""AAA""")
	client.MoveJ(np.degrees(q_start),1,0)
	client.ProgFinish(r"""AAA""")
	client.ProgSave(".","AAA",False)
	client.execute_motion_program("AAA.JBI")

def blending_zone_test():

	jog2start()

	client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
	client.MoveL(np.degrees(q_mid),10)
	client.MoveL(np.degrees(q_end),10,0)
	client.ProgEnd()
	timestamp, curve_exe_js=client.execute_motion_program("AAA.JBI")
	np.savetxt('blending_zone_test/nozone.csv',np.hstack((timestamp.reshape(-1, 1),curve_exe_js)),delimiter=',')



if __name__ == '__main__':
	blending_zone_test()