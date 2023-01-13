import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

##########################################FIGURE 8 MOVEC TEST###############################################
R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
p1=[1648,-1340,-220]
p2=[1698,-1290,-220]
p3=[1648,-1240,-220]
p4=[1598,-1190,-220]
p5=[1648,-1140,-220]
p6=[1698,-1190,-220]
p7=[1598,-1290,-220]
qseed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
q1=robot.inv(p1, R, qseed)[0]
q2=robot.inv(p2, R, qseed)[0]
q3=robot.inv(p3, R, qseed)[0]
q4=robot.inv(p4, R, qseed)[0]
q5=robot.inv(p5, R, qseed)[0]
q6=robot.inv(p6, R, qseed)[0]
q7=robot.inv(p7, R, qseed)[0]


client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)

client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
client.MoveC(np.degrees(q1),np.degrees(q2),np.degrees(q3),20,0)
client.MoveC(np.degrees(q3),np.degrees(q4),np.degrees(q5),10,0)
client.MoveC(np.degrees(q5),np.degrees(q6),np.degrees(q3),10,0)
client.MoveC(np.degrees(q3),np.degrees(q7),np.degrees(q1),10,0)



client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

client.execute_motion_program("AAA.JBI")
