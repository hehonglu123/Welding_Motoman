import sys
sys.path.append('../toolbox/')
from robot_def import *
from tesseract_env import *


dataset='blade0.1/'
sliced_alg='NX_slice2/'
data_dir='../data/'+dataset+sliced_alg
cmd_dir=data_dir+'cmd/50J/'

num_points_layer=50
num_layers=4
curve_sliced_relative=[]
curve_sliced_js=[]
positioner_js=[]

for i in range(num_layers):
	curve_sliced_relative.append(np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(i)+'.csv',delimiter=','))
	curve_sliced_js.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'.csv',delimiter=','))
	positioner_js.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'.csv',delimiter=','))

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=20)
positioner=robot_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')

t=Tess_Env('../config/urdf/combined')				#create obj
# for i in range(num_layers):
# 	t.viewer_trajectory_dual(robot.robot_name,positioner.robot_name,curve_sliced_js[i][::20],positioner_js[i][::20])
# 	time.sleep(10)
t.viewer_trajectory_dual(robot.robot_name,positioner.robot_name,curve_sliced_js[2][::20],positioner_js[2][::20])
input("Press enter to quit")