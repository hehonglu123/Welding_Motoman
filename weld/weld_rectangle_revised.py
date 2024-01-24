import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sys.path.append('../toolbox/')
from robot_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])

p_start=np.array([1620,-880,-260])
p_end=np.array([1620,-750,-260])

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)

feedrate = 200
base_layer_height=1.5
layer_height=1.0
q_all=[]
v_all=[]
cond_all=[]
primitives=[]
p_all=[]
# 定义板的尺寸和焊道的宽度
length = 130
width = 60
weld_width = 5

# 定义左下角的初始坐标
start_point = np.array([1620, -880, -260])

# 路径点列表初始化
p_all = []

# 计算需要焊接的总数
num_passes = int(length / weld_width)

# 为每个焊道生成路径点
for i in range(num_passes):
    # 如果是偶数焊道，从下到上焊接
    if i % 2 == 0:
        p1 = start_point + np.array([i * weld_width, 0, 0])
        p2 = p1 + np.array([0, width, 0])
    # 如果是奇数焊道，从上到下焊接
    else:
        p1 = start_point + np.array([i * weld_width, width, 0])
        p2 = p1 + np.array([0, -width, 0])
    
    p_all.extend([p1, p2])

# # quit()
# for i in range(0,4):
#     p_start=np.array([1620,-880,-260])
#     for n in range(0,7):
#         if i%2==0:
#             p1=p_start+np.array([0,0,i*layer_height])
#             p2=p_end+np.array([0,0,i*layer_height])
#             p3=p_end+np.array([5,0,i*layer_height])
#             p4=p_start+np.array([5,0,i*layer_height])
#             p5=p_start+np.array([10,0,i*layer_height])
#         else:
#             p1=p_end+np.array([0,0,i*layer_height])
#             p2=p_start+np.array([0,0,i*layer_height])
#             p3=p_start+np.array([5,0,i*layer_height])
#             p4=p_end+np.array([5,0,i*layer_height])
#             p5=p_end+np.array([10,0,i*layer_height])
#         q_1=robot.inv(p1,R,q_seed)[0]
#         q_2=robot.inv(p2,R,q_seed)[0]
#         q_3=robot.inv(p3,R,q_seed)[0]
#         q_4=robot.inv(p4,R,q_seed)[0]
#         q_5=robot.inv(p5,R,q_seed)[0]

#         p_all.extend([p1,p2,p3,p4,p5])
#         q_all.extend([q_1,q_2,q_3,q_4,q_5])
#         v_all.extend([1,30,30,30,30])
#         primitives.extend(['movej','movel','movel','movel','movel'])
#         cond_all.extend([0,int(feedrate/10)+200,int(feedrate/10)+200,int(feedrate/10)+200,int(feedrate/10)+200])
#         p_start=np.array([1620+(10*(n+1)),-880,-260])
#         p_end=np.array([1620+(10*(n+1)),-750,-260])
# p_all = p_all[:-1]
# primitives = primitives[:-1]
# q_all=q_all[:-1]
# v_all=v_all[:-1]
# cond_all=cond_all[:-1]
print('p_all',p_all)
print('primitives',primitives)
print('q_all',q_all)
print('v_all',v_all)
print('cond_all',cond_all)
# 准备绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 将 p_all 路径点提取为x、y、z坐标列表
x_coords = [point[0] for point in p_all]
y_coords = [point[1] for point in p_all]
z_coords = [point[2] for point in p_all]

# 绘制点
ax.scatter(x_coords, y_coords, z_coords, color='b', s=50)

# 为每一对点绘制箭头
for i in range(1, len(p_all)):
    ax.quiver(x_coords[i-1], y_coords[i-1], z_coords[i-1], 
              x_coords[i]-x_coords[i-1], y_coords[i]-y_coords[i-1], z_coords[i]-z_coords[i-1],
              color='r', length=np.linalg.norm([x_coords[i]-x_coords[i-1], y_coords[i]-y_coords[i-1], z_coords[i]-z_coords[i-1]]),
              normalize=True, head_length = 1, head_width = 0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Robot Path')

plt.show()
# ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=False)