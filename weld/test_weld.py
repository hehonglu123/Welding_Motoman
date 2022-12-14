import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')


# start_q=np.array([-35.4291,-56.6333,40.5194,-4.5177,-52.2505,11.6546])
# end_q=np.array([-36.8918,-61.1844,48.1628,-3.6876,-55.2334,9.6293])

start_q=np.array([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
end_q=np.array([-36.8918,61.1844,48.1628,3.6876,-55.2334,-9.6293])

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)


client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
client.MoveJ(Pose([0,0,0,0,0,0]),start_q,1,0)
client.SetArc(True,AC=110,AVP=100,T=0.,V=250)
client.MoveL(Pose([0,0,0,0,0,0]),end_q,20,0)
client.SetArc(False)
client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

client.execute_motion_program("AAA.JBI")
