import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from lambda_calc import *

data_dir='../data/wall/'
solution_dir='baseline/'

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

# print(robot.fwd(np.radians([69.70880615,7.860881676,-27.5568493,-7.707783068,-31.93097882,13.85803702])))

client=MotionProgramExecClient()

ws=WeldSend(client)
ws.wire_cut(robot,speed=5)

R=np.array([[-0.7071, 0.7071, -0.    ],
			[ 0.7071, 0.7071,  0.    ],
			[0.,      0.,     -1.    ]])

###tilt tool by X Degrees
# R=Ry(np.radians(-30))@R

p_start=np.array([1650,-850,-250])
p_end=np.array([1650,-780,-250])



p_all=np.linspace(p_start,p_end,num=10)
p_all=np.array([p_start]*10)
lam=calc_lam_cs(p_all)

q_all=ws.touchsense(robot,p_all+np.array([0,0,10]),p_all-np.array([0,0,10]),R)

p_act=robot.fwd(q_all).p_all

# plt.plot(lam,p_act[:,-1])
plt.plot(p_act[:,-1])
plt.show()