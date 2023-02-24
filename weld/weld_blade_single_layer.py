import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *


dataset='blade0.1/'
sliced_alg='NX_slice2/'
data_dir='../data/'+dataset+sliced_alg
cmd_dir=data_dir+'cmd/50J/'

num_points_layer=50
curve_sliced_js=[]
positioner_js=[]

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=20)
positioner=robot_obj('D500B',def_path='../config/D500B_robot_default_config.yml',pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')

client=MotionProgramExecClient(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg)

base_layer=0
curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'.csv',delimiter=',')
positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'.csv',delimiter=',')

if base_layer % 2==1:
    breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
else:
    breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

for j in range(len(breakpoints)):
    target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
    client.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), 10,target2=target2)


# layer=0
# curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'.csv',delimiter=',')
# positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'.csv',delimiter=',')

# if layer % 2==1:
#     breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
# else:
#     breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

# for j in range(len(breakpoints)):
#     target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
#     client.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), 10,target2=target2)

    
client.ProgEnd()
client.execute_motion_program("AAA.JBI") 