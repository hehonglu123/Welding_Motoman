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
# p_end=np.array([1680,-705,-255]) # 789
# p_start=np.array([1680,-920,-255]) # 789
client=MotionProgramExecClient()
ws=WeldSend(client)

feedrate = int(470)
base_layer_height=6.5
layer_height=8
weld_v = 5
z_int = -260
q_all=[]
v_all=[]
cond_all=[]
primitives=[]
p_all=[]
# quit
x_int = 1686#1630#
x_end = 1754#1690#
y_int = -924.5
y_end = -702.5
for i in range(2,3):
    if i%2==0:
        p_start=np.array([x_int,y_int,z_int])
        p_end=np.array([x_end,y_int,z_int])
    else:
        p_start=np.array([x_int,y_end,z_int])
        p_end=np.array([x_end,y_end,z_int])
    for n in range(10,11):#19
        if i%2==0:
            p_start=np.array([x_int,y_int+(12*(n)),z_int])
            p_end=np.array([x_end,y_int+(12*(n)),z_int])
            p1=p_start+np.array([0,0,i*layer_height])
            p2=p_end+np.array([0,0,i*layer_height])
            p3=p_end+np.array([0,6,i*layer_height])
            p4=p_start+np.array([0,6,i*layer_height])
            p5=p_start+np.array([0,12,i*layer_height])
            print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
            p_start=np.array([x_int,y_int+(12*(n+1)),z_int])
            p_end=np.array([x_end,y_int+(12*(n+1)),z_int])
        else:
            p_start=np.array([x_int,y_end+(-12*(n)),z_int])
            p_end=np.array([x_end,y_end+(-12*(n)),z_int])
            p1=p_start+np.array([0,0,i*layer_height])
            p2=p_end+np.array([0,0,i*layer_height])
            p3=p_end+np.array([0,-6,i*layer_height])
            p4=p_start+np.array([0,-6,i*layer_height])
            p5=p_start+np.array([0,-12,i*layer_height])
            print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
            p_start=np.array([x_int,y_end+(-12*(n+1)),z_int])
            p_end=np.array([x_end,y_end+(-12*(n+1)),z_int])
        q_1=robot.inv(p1,R,q_seed)[0]
        q_2=robot.inv(p2,R,q_seed)[0]
        q_3=robot.inv(p3,R,q_seed)[0]
        q_4=robot.inv(p4,R,q_seed)[0]
        q_5=robot.inv(p5,R,q_seed)[0]

        p_all.extend([p1,p2,p3,p4,p5])
        q_all.extend([q_1,q_2,q_3,q_4,q_5])
        v_all.extend([1,weld_v,weld_v,weld_v,weld_v])
        primitives.extend(['movej','movel','movel','movel','movel'])
        cond_all.extend([0,feedrate,feedrate,feedrate,feedrate])
# x_int = 1635#1630#
# x_end = 1695#1690#
# y_int = -925
# y_end = -705
# for i in range(1,2): #layers
#     if i%2==0:
#         p_start=np.array([x_int,y_int,z_int])
#         p_end=np.array([x_int,y_end,z_int])
#     else:
#         p_start=np.array([x_end,y_end,z_int])
#         p_end=np.array([x_end,y_int,z_int])
#     for n in range(0,6): # segments max: 6
#         if i%2==0:
#             p_start=np.array([x_int+(10*(n)),y_int,z_int])
#             p_end=np.array([x_int+(10*(n)),y_end,z_int])
#             p1=p_start+np.array([0,0,i*layer_height])
#             p2=p_end+np.array([0,0,i*layer_height])
#             p3=p_end+np.array([5,0,i*layer_height])
#             p4=p_start+np.array([5,0,i*layer_height])
#             p5=p_start+np.array([10,0,i*layer_height])
#             print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
#             p_start=np.array([x_int+(10*(n+1)),y_int,z_int])
#             p_end=np.array([x_int+(10*(n+1)),y_end,z_int])
#         else:
#             p_start=np.array([x_end+(-10*(n)),y_end,z_int])
#             p_end=np.array([x_end+(-10*(n)),y_int,z_int])
#             p1=p_start+np.array([0,0,i*layer_height])
#             p2=p_end+np.array([0,0,i*layer_height])
#             p3=p_end+np.array([-5,0,i*layer_height])
#             p4=p_start+np.array([-5,0,i*layer_height])
#             p5=p_start+np.array([-10,0,i*layer_height])
#             print('p1,p2,p3,p4,p5:',p1,p2,p3,p4,p5)
#             p_start=np.array([x_end+(-10*(n+1)),y_end,z_int])
#             p_end=np.array([x_end+(-10*(n+1)),y_int,z_int])
#         q_1=robot.inv(p1,R,q_seed)[0]
#         q_2=robot.inv(p2,R,q_seed)[0]
#         q_3=robot.inv(p3,R,q_seed)[0]
#         q_4=robot.inv(p4,R,q_seed)[0]
#         q_5=robot.inv(p5,R,q_seed)[0]

#         p_all.extend([p1,p2,p3,p4,p5])
#         q_all.extend([q_1,q_2,q_3,q_4,q_5])
#         v_all.extend([1,weld_v,weld_v,weld_v,weld_v])
#         primitives.extend(['movej','movel','movel','movel','movel'])
#         cond_all.extend([0,feedrate,feedrate,feedrate,feedrate])
        
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