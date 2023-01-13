import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')


speed=[5,10,25]
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

final_layer_length=80
final_layer_shift=np.array([0,final_layer_length,0])/2


center=np.array([1648,-1240,-228])
x_offset=np.array([35,0,0])

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
final_layer=[center-final_layer_shift,center+final_layer_shift]

final_layers=[]
for i in range(3):
	for j in range(2):
		final_layers.append([final_layer[j]+(i-1)*x_offset])
final_layers=np.array(final_layers).reshape((3,2,3))

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)


client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

q_init=np.degrees(robot.inv(final_layers[0][0],R,q_seed)[0])
client.MoveJ(q_init,1,0)	

# for i in range(len(final_layers)):
# 	q_init=np.degrees(robot.inv(final_layers[i][0],R,q_seed)[0])

# 	q_end=np.degrees(robot.inv(final_layers[i][1],R,q_seed)[0])
# 	client.MoveJ(q_init,1,0)
# 	# client.SetArc(True)
# 	client.MoveL(q_end,speed[i],0)
# 	# client.SetArc(False)


client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

client.execute_motion_program("AAA.JBI")
