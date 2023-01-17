import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from result_analysis import *
import matplotlib.pyplot as plt


robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])

center=np.array([1678,-1240,-231])

base_layer_length=130
base_layer_shift=np.array([0,base_layer_length,0])/2

final_layer_length=80
final_layer_shift=np.array([0,final_layer_length,0])/2

p1=center-base_layer_shift
p2=center-final_layer_shift
p3=center+final_layer_shift
p4=center+base_layer_shift

q1=np.degrees(robot.inv(p1,R,q_seed)[0])
q2=np.degrees(robot.inv(p2,R,q_seed)[0])
q3=np.degrees(robot.inv(p3,R,q_seed)[0])
q4=np.degrees(robot.inv(p4,R,q_seed)[0])

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)


client.ACTIVE_TOOL=1

client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

client.MoveJ(q1,1,0)
client.MoveL(q2,10,0.5)
# client.SetArc(True)
client.MoveL(q3,10,0.5)
# client.SetArc(False)
client.MoveL(q4,10,0)
	


client.ProgFinish(r"""AAA""")
client.ProgSave(".","AAA",False)

timestamp, curve_exe_js=client.execute_motion_program("AAA.JBI")

lam, curve_exe, curve_exe_R, speed=logged_data_analysis(robot,timestamp,np.radians(curve_exe_js))

start_idx=np.argmin(np.linalg.norm(curve_exe-p1,axis=1))

plt.title('Speed')
plt.plot(lam[start_idx+1:],speed[start_idx:])
plt.show()