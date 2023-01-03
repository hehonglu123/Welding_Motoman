import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

data_dir='../data/wall/'
solution_dir='baseline/'

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

ms=MotionSend()
primitives_all=[]
p_bp_all=[]
q_bp_all=[]
# for file in glob.glob(data_dir+solution_dir+'1L/*.csv'):
# 	breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(file)
# 	ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,[1,20],0)

ms.exec_motion_from_dir(robot,data_dir+solution_dir+'1L/')