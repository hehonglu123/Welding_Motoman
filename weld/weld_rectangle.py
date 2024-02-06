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



q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
p_start=np.array([1620,-880,-260])
p_end=np.array([1620,-750,-260])
client=MotionProgramExecClient()
ws=WeldSend(client)

feedrate = int(501)
base_layer_height=1.5
layer_height=1.0
q_all=[]
v_all=[]
cond_all=[]
primitives=[]
p_all=[]
# quit
for i in range(31,32):
    if i%2==0:
        p_start=np.array([1620,-880,-260])
        p_end=np.array([1620,-750,-260])
    else:
        p_start=np.array([1690,-880,-260])
        p_end=np.array([1690,-750,-260])
    for n in range(0,7):
        if i%2==0:
            p1=p_start+np.array([0,0,i*layer_height])
            p2=p_end+np.array([0,0,i*layer_height])
            p3=p_end+np.array([5,0,i*layer_height])
            p4=p_start+np.array([5,0,i*layer_height])
            p5=p_start+np.array([10,0,i*layer_height])
            print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
            p_start=np.array([1620+(10*(n+1)),-880,-260])
            p_end=np.array([1620+(10*(n+1)),-750,-260])
        else:

            p1=p_start+np.array([0,0,i*layer_height])
            p2=p_end+np.array([0,0,i*layer_height])
            p3=p_end+np.array([-5,0,i*layer_height])
            p4=p_start+np.array([-5,0,i*layer_height])
            p5=p_start+np.array([-10,0,i*layer_height])
            print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
            p_start=np.array([1690+(-10*(n+1)),-880,-260])
            p_end=np.array([1690+(-10*(n+1)),-750,-260])
        q_1=robot.inv(p1,R,q_seed)[0]
        q_2=robot.inv(p2,R,q_seed)[0]
        q_3=robot.inv(p3,R,q_seed)[0]
        q_4=robot.inv(p4,R,q_seed)[0]
        q_5=robot.inv(p5,R,q_seed)[0]

        p_all.extend([p1,p2,p3,p4,p5])
        q_all.extend([q_1,q_2,q_3,q_4,q_5])
        v_all.extend([1,4,4,4,4])
        primitives.extend(['movej','movel','movel','movel','movel'])
        cond_all.extend([0,feedrate,feedrate,feedrate,feedrate])

p_all = p_all[:-1]
primitives = primitives[:-1]
q_all=q_all[:-1]
v_all=v_all[:-1]
cond_all=cond_all[:-1]
print('p_all',p_all)
print('primitives',primitives)
print('q_all',q_all)
print('v_all',v_all)
print('cond_all',cond_all)
ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True)