import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from WeldSend import *

data_dir='../data/spiral_cylinder/'
solution_dir='baseline/'

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)


###########################################layer welding############################################
client=MotionProgramExecClient()

mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg, tool_num = 12)

num_layer_start=49
num_layer_end=50
for layer in range(num_layer_start,num_layer_end,1):

	####################DETERMINE CURVE ORDER##############################################
	curve_sliced_js=np.loadtxt(data_dir+solution_dir+'curve_sliced_js/MA2010_js'+str(layer)+'.csv',delimiter=',')

	vd=6

	num_points_layer=5 ####minimum of 5 (2 semi-circle) required for circular spiral, otherwise 3 points not work

	###find which end to start
	breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
		
	mp.MoveJ(np.degrees(curve_sliced_js[breakpoints[0]]), 1)

	mp.setArc(True,cond_num=300)
	mp.MoveC(np.degrees(curve_sliced_js[breakpoints[0]]),np.degrees(curve_sliced_js[breakpoints[1]]),np.degrees(curve_sliced_js[breakpoints[2]]), vd,zone=None)
	mp.MoveC(np.degrees(curve_sliced_js[breakpoints[2]]),np.degrees(curve_sliced_js[breakpoints[3]]),np.degrees(curve_sliced_js[breakpoints[4]]), vd,zone=None)

mp.setArc(False)

client.execute_motion_program(mp)