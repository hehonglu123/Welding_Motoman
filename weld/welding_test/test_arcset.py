import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
p_start=np.array([1665,-1290,-231])
p_end=np.array([1665,-1190,-231])

p_dense=np.linspace(p_start,p_end,num=3)
qseed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
q_dense=[]
for p in p_dense:
	q_dense.append(robot.inv(p, R, qseed)[0])

# ipm=np.arange(100,250,100)
ipm=[250,100]
client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)

client.MoveJ(np.degrees(q_dense[0]),1,0)
# client.SetArc(True,cond_num=ipm[0])

for i in range(1,len(q_dense)-1):
	client.MoveL(np.degrees(q_dense[i]),5,0)
	# client.ChangeArc(ipm[i])

client.MoveL(np.degrees(q_dense[-1]),5,0)
# client.SetArc(False)
client.ProgEnd()

# client.StartStreaming()
timestamp,joint_recording=client.execute_motion_program()
print(joint_recording)
np.savetxt('weldchange.csv',np.hstack((timestamp.reshape(-1,1),joint_recording)),delimiter=',')