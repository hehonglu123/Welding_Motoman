import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

q1=np.array([])
q2=np.array([])
q3=np.array([])
q4=np.array([])
q5=np.array([])

robot=robot_obj('MA_1440_A0',def_path='../config/MA_1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp.csv',\
	pulse2deg_file_path='../config/MA_1440_A0_pulse2deg.csv')
ms=MotionSend()

client=

ms.exec_motions(robot,['movej'],[robot.fwd(q1).p],[q1],10,0)
client.
