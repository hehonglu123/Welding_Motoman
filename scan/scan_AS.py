import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

import cv2
import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_1440_A0',def_path='../config/MA_1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp2.csv',\
	pulse2deg_file_path='../config/MA_1440_A0_pulse2deg.csv')

q4=np.array([43.5893,72.1362,45.2749,-84.0966,24.3644,94.2091])
q5=np.array([34.6291,55.5756,15.4033,-28.8363,24.0298,3.6855])
q6=np.array([27.3821,51.3582,-19.8428,-21.2525,71.6314,-62.8669])


client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot.pulse2deg)

###TODO: fix tool definition
# client.motoman.DONT_USE_SETTOOL=False
# client.motoman.setTool(Pose([0,0,450,0,0,0]), None, 'welder')
client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA1440 Base""")

client.MoveJ(q4,1,0)
client.MoveC(q4, q5, q6, 10,1)

client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

print(client.execute_motion_program("AAA.JBI"))