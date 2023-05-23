import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from WeldSend import *

data_dir='../data/cylinder/'
solution_dir='baseline/'

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)


###########################################layer welding############################################
client=MotionProgramExecClient()

mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg, tool_num = 12)

num_layer_start=48
num_layer_end=49
for layer in range(num_layer_start,num_layer_end,1):

	####################DETERMINE CURVE ORDER##############################################
	curve_sliced_js=np.loadtxt(data_dir+solution_dir+'curve_sliced_js/MA2010_js'+str(layer)+'.csv',delimiter=',')

	vd=7

	num_points_layer=3

	###find which end to start
	if layer % 2 ==0:
		breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
	else:
		breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

		
	mp.MoveJ(np.degrees(curve_sliced_js[breakpoints[0]]), 1)

	mp.setArc(True,cond_num=300)
	mp.MoveC(np.degrees(curve_sliced_js[breakpoints[0]]),np.degrees(curve_sliced_js[breakpoints[1]]),np.degrees(curve_sliced_js[breakpoints[2]]), vd,zone=1)
	mp.setArc(False)

client.execute_motion_program(mp)