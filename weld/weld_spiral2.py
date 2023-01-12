import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

data_dir='../data/spiral_cylinder/'
solution_dir='baseline/'

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

curve_js=np.loadtxt(data_dir+solution_dir+'curve_sliced_js/0.csv',delimiter=',')

# print(robot.pulse2deg)
client=MotionProgramExecClient(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)

###TODO: fix tool definition
# client.motoman.DONT_USE_SETTOOL=False
# client.motoman.setTool(Pose([0,0,450,0,0,0]), None, 'welder')
client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
q0=np.degrees(curve_js[0])
client.MoveJ(q0,2,0)
client.SetArc(True)

indices=np.linspace(0,len(curve_js),3*50,endpoint=False).astype(int)[:100]

for i in range(len(indices)):
	if i<10:
		speed=5
	else:
		speed=50
	target_id1 = client.add_target_joints(np.degrees(curve_js[indices[i]]))
	client.addline("MOVC C%05d %s%s" % (target_id1, "V=%.1f" % speed, ' PL=%i' % round(min(1, 8))))

client.SetArc(False)
client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

print(client.execute_motion_program("AAA.JBI"))
