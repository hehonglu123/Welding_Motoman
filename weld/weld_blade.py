import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *


dataset='blade0.1/'
sliced_alg='NX_slice/'
data_dir='../data/'+dataset+sliced_alg
cmd_dir=data_dir+'cmd/50J/'

num_layers=50
curve_sliced_relative=[]
for i in range(num_layers):
	curve_sliced_relative.append(np.readtxt(data_dir+'curve_sliced_relative/'+str(i)+'.csv',delimiter=','))

robot=robot_obj('MA_2010_A0',def_path='../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA_2010_A0_pulse2deg.csv')
positioner=robot_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file=)

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

ms.exec_motion_from_dir(robot,cmd_dir,arc=True)