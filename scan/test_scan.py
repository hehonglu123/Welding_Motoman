import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

sin30=.5
cos30=np.cos(np.radians(30))
sin60=np.sin(np.radians(60))
cos60=.5

robot=robot_obj('MA_1440_A0',def_path='../config/MA_1440_A0_robot_default_config.yml',tool_file_path='../config/scanner_tcp.csv',\
	pulse2deg_file_path='../config/MA_1440_A0_pulse2deg.csv')

center=np.array([1000,0,-200])
radius=300
p1=np.array([center[0]-radius*cos60,center[1],center[2]+radius*sin60])
p2=np.array([center[0],center[1],center[2]+radius])
p3=np.array([center[0]+radius*cos30,center[1],center[2]+radius*sin30])
R1=np.array([[sin60,0,cos60],\
			[ 0,-1,0],\
			[cos60,0,-sin60]])
R2=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
R3=np.array([[sin30,0,-cos30],\
			[ 0,-1,0],\
			[-cos30,0,-sin30]])

# q1=robot.inv(p1,R1,last_joints=0.1*np.ones(6))[0]
# print(robot.fwd(q1))
q2=robot.inv(p2,R2,last_joints=0.1*np.ones(6))[0]
print(q2)
q3=robot.inv(p3,R3,last_joints=q2)[0]
print(q3)

client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot.pulse2deg)

###TODO: fix tool definition
# client.robodk_rob.DONT_USE_SETTOOL=False
# client.robodk_rob.setTool(Pose([0,0,450,0,0,0]), None, 'welder')
client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA1440 Base""")
# client.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(q1),5,0)
client.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(q2),5,0)
client.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(q3),5,0)
client.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(q2),5,0)
client.MoveJ(Pose([0,0,0,0,0,0]),np.degrees(q3),5,0)
client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

# client.execute_motion_program("AAA.JBI")
# client.disconnectMH()