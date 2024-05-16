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

# Top layers
# p_start=np.array([1634,-846,-255]) # 847
# p_end=np.array([1660,-846,-255]) # 847

# p_end=np.array([1653,-755,-255]) # 789
# p_start=np.array([1653,-870,-255]) # 789

p_topleft = np.array([1625,-860,-260])
p_topright =np.array([1625,-800,-260])

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)

# weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
# rr_sensors = WeldRRSensor(weld_service=weld_ser)

feedrate = int(470)
base_layer_height=2.5
layer_height=1.5
q_all=[]
v_all=[]
cond_all=[]
primitives=[]

# Base layer
for i in range(2,3):
    if i%2==0:
        p1 = p_topleft+np.array([0,0,i*base_layer_height])
        p2 = p_topright+np.array([0,0,i*base_layer_height])
        p3 = p_topleft+np.array([60,0,i*base_layer_height])
        p4 = p_topright+np.array([60,0,i*base_layer_height])

    else:
        p1 = p_topright+np.array([60,0,i*base_layer_height])
        p2 = p_topleft+np.array([60,0,i*base_layer_height])
        p3 = p_topright+np.array([0,0,i*base_layer_height])
        p4 = p_topleft+np.array([0,0,i*base_layer_height])


    q_1=robot.inv(p1,R,q_seed)[0]
    q_2=robot.inv(p2,R,q_seed)[0]
    q_3=robot.inv(p3,R,q_seed)[0]
    q_4=robot.inv(p4,R,q_seed)[0]

	# p_mid1=p1+5*(p2-p1)/np.linalg.norm(p2-p1)
	# p_mid2=p2-5*(p2-p1)/np.linalg.norm(p2-p1)
	# q_mid1=robot.inv(p_mid1,R,q_seed)[0]
	# q_mid2=robot.inv(p_mid2,R,q_seed)[0]

    q_all.extend([q_1,q_2,q_3,q_4])
    v_all.extend([5,10,10,10])
    primitives.extend(['movej','movel','movel','movel'])
    cond_all.extend([0,feedrate,feedrate,feedrate])
    print(cond_all)

# Top Layers
	
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

print("p1",p1)
print('p2',p2)
print('p3',p3)
print('p4',p4)

print('primitives',primitives)
print('q_all',q_all)
print('v_all',v_all)
print('cond_all',cond_all)

# rr_sensors.start_all_sensors()
ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=True)
# rr_sensors.stop_all_sensors()

# Path(data_dir).mkdir(exist_ok=True)
# try:
# 	rr_sensors.save_all_sensors(data_dir)
# except:
# 	traceback.print_exc()
# input('press enter to continue')