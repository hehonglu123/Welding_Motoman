import sys, time, datetime
sys.path.append('../toolbox/')
from robot_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from weldRRSensor import *
from pathlib import Path

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_data_logging'+formatted_time+'/'

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
## Nominal parameters:
# Base layers
# p_start=np.array([1650,-880,-258])
# p_end=np.array([1650,-780,-258])
#######################################
start_layer = 13
#######################################
velocity_matrix = 5
feedrate_matrix = 223
velocity_fill = 5
feedrate_fill = 228
end_layer = start_layer + 1
base_layer_height= 4
initial_z = -220
if (start_layer + 4) % 2 == 0:
    x_start = 1630
    x_end = 1680
    y_start_matrix = -920
    y_start_fill = -914.322
    distance = 11.356
else:
    x_start = 1680
    x_end = 1630
    y_start_matrix = -704.236
    y_start_fill = -709.914
    distance =-11.356
for n_matrix in range(0,20):

    print('n_matrix',n_matrix)
    # Top layers
    p_start=np.array([x_start,y_start_matrix + distance*n_matrix,initial_z])
    p_end=np.array([x_end,y_start_matrix + distance*n_matrix,initial_z])

    q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

    client=MotionProgramExecClient()
    ws=WeldSend(client)

    # weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
    # rr_sensors = WeldRRSensor(weld_service=weld_ser)

    feedrate = int(460)
    q_all=[]
    v_all=[]
    cond_all=[]
    primitives=[]

    # Base layer
    for i in range(start_layer,end_layer):
        p1=p_start+np.array([0,0,i*base_layer_height])
        p2=p_end+np.array([0,0,i*base_layer_height])

        q_init=robot.inv(p1,R,q_seed)[0]
        q_end=robot.inv(p2,R,q_seed)[0]
        
        q_all.extend([q_init,q_end])
        v_all.extend([1,velocity_matrix])
        primitives.extend(['movej','movel'])
        cond_all.extend([0,feedrate_matrix])
        print(cond_all)
    print('p1',p1)
    print('p2',p2)
    # print('primitives',primitives)
    # print('q_all',q_all)
    print('v_all',v_all)
    print('cond_all',cond_all)

    ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True)
    # time.sleep(5)
time.sleep(60)

for n_fill in range(0,19):
    print('n_fill',n_fill)
    # Top layers
    p_start=np.array([x_end,y_start_fill + distance * n_fill,initial_z])
    p_end=np.array([x_start,y_start_fill + distance * n_fill,initial_z])

    q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

    client=MotionProgramExecClient()
    ws=WeldSend(client)

    # weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
    # rr_sensors = WeldRRSensor(weld_service=weld_ser)

    feedrate = int(460)
    q_all=[]
    v_all=[]
    cond_all=[]
    primitives=[]

    # Base layer
    for i in range(start_layer,end_layer):
        p1=p_start+np.array([0,0,i*base_layer_height])
        p2=p_end+np.array([0,0,i*base_layer_height])

        q_init=robot.inv(p1,R,q_seed)[0]
        q_end=robot.inv(p2,R,q_seed)[0]
        
        q_all.extend([q_init,q_end])
        v_all.extend([1,velocity_fill])
        primitives.extend(['movej','movel'])
        cond_all.extend([0,feedrate_fill])
        print(cond_all)

    print('p1',p1)
    print('p2',p2)
    # print('primitives',primitives)
    # print('q_all',q_all)
    print('v_all',v_all)
    print('cond_all',cond_all)

    ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True)
    time.sleep(10)
    # input('press enter to continue')
print('Job finished, start layer:',start_layer)


