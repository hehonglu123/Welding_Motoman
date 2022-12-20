import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

start_q=np.zeros(6)
end_q=0.1*np.ones(6)
client=MotionProgramExecClient('127.0.0.1',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)

###TODO: fix tool definition
# client.robodk_rob.DONT_USE_SETTOOL=False
# client.robodk_rob.setTool(Pose([0,0,450,0,0,0]), None, 'welder')
client.robodk_rob.ACTIVE_TOOL=1

client.robodk_rob.ProgStart(r"""AAA""")
client.robodk_rob.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
client.robodk_rob.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(start_q),5,0)
client.robodk_rob.SetArc(True)
client.robodk_rob.MoveL(Pose([0,0,0,0,0,0]),np.degrees(end_q),5,0)
client.robodk_rob.SetArc(False)
client.robodk_rob.ProgFinish(r"""AAA""")
client.robodk_rob.ProgSave(".","AAA",False)

# client.execute_motion_program("AAA.JBI")
