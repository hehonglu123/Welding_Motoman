import sys
sys.path.append('toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

robot1=robot_obj('MA2010_A0',def_path='config/MA2010_A0_robot_default_config.yml',tool_file_path='config/weldgun.csv',\
	pulse2deg_file_path='config/MA2010_A0_pulse2deg_real.csv')

robot2=robot_obj('MA2010_A0',def_path='config/MA2010_A0_robot_default_config.yml',tool_file_path='config/weldgun.csv',\
	pulse2deg_file_path='config/MA1440_A0_pulse2deg_real.csv')

station_pulse2deg=np.abs(np.loadtxt('config/D500B_pulse2deg_real.csv'))

q1=np.zeros(6)
q2=np.zeros(6)
q2[0]=90
q3=[-15,180]


	
client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot1.pulse2deg)
client.MoveJ(q1,5,0)
client.execute_motion_program("AAA.JBI")
np.savetxt('js_recording.csv',np.hstack((timestamp.reshape(-1, 1),joint_recording)),delimiter=',')
# print(timestamp)
# print(joint_recording)

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB2',pulse2deg=robot2.pulse2deg)
client.MoveJ(q2,5,0)
client.execute_motion_program("AAA.JBI")


client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='ST1',pulse2deg=station_pulse2deg)
client.MoveJ(q3,10,0)
client.execute_motion_program("AAA.JBI")


