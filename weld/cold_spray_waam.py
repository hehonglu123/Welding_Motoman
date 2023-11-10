import sys
sys.path.append('../toolbox/')
from robot_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
# p_start=np.array([1650,-840,-260])
# p_end=np.array([1650,-780,-260])
p_start=np.array([1670,-840,-260])
p_end=np.array([1630,-840,-260])
# p_start=np.array([1650,-860,-260])
# p_end=np.array([1650,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)

feedrate = 100
base_layer_height=1.5
layer_height=1.0
q_all=[]
v_all=[]
cond_all=[]
primitives=[]

for i in range(4,5):
	if i%2==0:
		p1=p_start+np.array([0,0,i*base_layer_height])
		p2=p_end+np.array([0,0,i*base_layer_height])
	else:
		p1=p_end+np.array([0,0,i*base_layer_height])
		p2=p_start+np.array([0,0,i*base_layer_height])

	
	q_init=robot.inv(p1,R,q_seed)[0]
	q_end=robot.inv(p2,R,q_seed)[0]

	# p_mid1=p1+5*(p2-p1)/np.linalg.norm(p2-p1)
	# p_mid2=p2-5*(p2-p1)/np.linalg.norm(p2-p1)
	# q_mid1=robot.inv(p_mid1,R,q_seed)[0]
	# q_mid2=robot.inv(p_mid2,R,q_seed)[0]

	q_all.extend([q_init,q_end])
	v_all.extend([1,2])
	primitives.extend(['movej','movel'])
	cond_all.extend([0,int(feedrate/10)+200])

# for i in range(2,3):
# 	if i%2==0:
# 		p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
# 		p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
# 	else:
# 		p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
# 		p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

# 	q_init=robot.inv(p1,R,q_seed)[0]
# 	q_end=robot.inv(p2,R,q_seed)[0]
# 	q_all.extend([q_init,q_end])
# 	v_all.extend([1,15])
# 	primitives.extend(['movej','movel'])
# 	cond_all.extend([0,210])

print('primitives',primitives)
print('q_all',q_all)
print('v_all',v_all)
print('cond_all',cond_all)
ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True)