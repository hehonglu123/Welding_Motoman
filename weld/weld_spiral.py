import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

data_dir='../data/spiral_cylinder/'
solution_dir='baseline/'

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')

ms=MotionSend()
primitives_all=[]
p_bp_all=[]
q_bp_all=[]

cmd_dir=data_dir+solution_dir+'1C_reverse/'

# num_command=len(fnmatch.filter(cmd_dir, '*.csv'))
# num_command=10
# for i in range(num_command):
# 	breakpoints,primitives, p_bp,q_bp=ms.extract_data_from_cmd(cmd_dir+'command'+str(i)+'.csv')
# 	ms.exec_motions(robot,primitives,breakpoints,p_bp,q_bp,[1,20],0)

ms.exec_motion_from_dir(robot,cmd_dir,arc=False)