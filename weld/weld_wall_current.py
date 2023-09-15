import sys, glob, wave, pickle
from multiprocessing import Process
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from pathlib import Path
sys.path.append('../../toolbox/')
from utils import *
from robot_def import *
from lambda_calc import *
from multi_robot import *
from flir_toolbox import *
from traj_manipulation import *
from dx200_motion_program_exec_client import *
from StreamingSend import *
sys.path.append('../')
from weldRRSensor import *
import datetime

def my_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

def save_data(recorded_dir,current_data,welding_data,slice_num):
	###MAKING DIR
	layer_data_dir=recorded_dir+'layer_'+str(slice_num)+'/'
	Path(layer_data_dir).mkdir(exist_ok=True)


	####CURRENT SAVING
	np.savetxt(layer_data_dir + 'current.csv',current_data, delimiter=',',header='timestamp,current', comments='')

	####FRONIUS SAVING
	np.savetxt(layer_data_dir + 'welding.csv',welding_data, delimiter=',',header='timestamp,voltage,current,feedrate,energy', comments='')

	return

current_time = datetime.datetime.now()
formatted_time = current_time.strftime('%Y_%m_%d_%H_%M_%S.%f')[:-7]
data_dir='../data/wall_weld_test/weld_scan_'+formatted_time+'/'



robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
# p_start=np.array([1650,-840,-260])
# p_end=np.array([1650,-780,-260])
p_start=np.array([1650,-850,-260])
p_end=np.array([1650,-770,-260])
# p_start=np.array([1650,-860,-260])
# p_end=np.array([1650,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)

########################################################RR FRONIUS########################################################
weld_ser = RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
########################################################RR CURRENT########################################################
current_ser=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')
## RR sensor objects
rr_sensors = WeldRRSensor(weld_service=weld_ser,current_service=current_ser)

feedrate = 200
base_layer_height=1.5
layer_height=1.0
q_all=[]
v_all=[]
cond_all=[]
primitives=[]

for i in range(15,20):
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
	v_all.extend([1,25])
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


ws.weld_segment_single(primitives,robot,q_all,v_all,cond_all,arc=False,wait=0.)