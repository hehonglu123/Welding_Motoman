import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from RobotRaconteur.Client import *


dataset='bell/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')

rob2_js=np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(int(2*slicing_meta['num_layers']/3))+'_0.csv',delimiter=',').reshape((-1,6))

client=MotionProgramExecClient()
ws=WeldSend(client)
ws.jog_single(robot2,rob2_js[0])