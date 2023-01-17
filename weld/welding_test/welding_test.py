import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

base_layer_length=130
base_layer_shift=np.array([0,base_layer_length,0])/2


center=np.array([1648,-1240,-231])
x_offset=np.array([35,0,0])
layer_height=1
R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
first_layer=[center-base_layer_shift,center+base_layer_shift]
second_layer=[]
for p in first_layer:
	second_layer.append(p+np.array([0,0,layer_height]))

first_layers=[]
second_layers=[]
for i in range(3):
	for j in range(2):
		first_layers.append([first_layer[j]+(i-1)*x_offset])
		second_layers.append([second_layer[j]+(i-1)*x_offset])
first_layers=np.array(first_layers).reshape((3,2,3))
second_layers=np.array(second_layers).reshape((3,2,3))

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)


client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

for i in range(len(first_layers)):
	q_init=np.degrees(robot.inv(first_layers[i][0],R,q_seed)[0])

	q_end=np.degrees(robot.inv(first_layers[i][1],R,q_seed)[0])
	client.MoveJ(q_init,1,0)
	client.SetArc(True)
	client.MoveL(q_end,10,0)
	client.SetArc(False)
for i in range(len(second_layers)):
	q_init=np.degrees(robot.inv(second_layers[i][0],R,q_seed)[0])
	q_end=np.degrees(robot.inv(second_layers[i][1],R,q_seed)[0])
	client.MoveJ(q_init,1,0)
	client.SetArc(True)
	client.MoveL(q_end,10,0)
	client.SetArc(False)

client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

client.execute_motion_program("AAA.JBI")
