import sys
sys.path.append('../toolbox/')
from robot_def import *
from multi_robot import *
from path_calc import *
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
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg.csv',d=15)
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
    pulse2deg_file_path='../config/D500B_pulse2deg.csv',base_transformation_file='../config/D500B_pose.csv')

client=MotionProgramExecClient(ROBOT_CHOICE='RB1',ROBOT_CHOICE2='ST1',pulse2deg=robot.pulse2deg,pulse2deg_2=positioner.pulse2deg)

###########################################base layer welding############################################
# base_layer=0
# curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_base_js'+str(base_layer)+'.csv',delimiter=',')
# positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_base_js'+str(base_layer)+'.csv',delimiter=',')
# curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/baselayer'+str(base_layer)+'.csv',delimiter=',')

# if base_layer % 2==1:
#     breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
# else:
#     breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

# vd_relative=8
# lam1=calc_lam_js(curve_sliced_js,robot)
# lam2=calc_lam_js(positioner_js,positioner)
# lam_relative=calc_lam_cs(curve_sliced_relative)
# s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)


# target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),10]
# client.MoveL(np.degrees(curve_sliced_js[breakpoints[0]]), s1_all[0],target2=target2)
# client.SetArc(True,cond_num=250)
# for j in range(1,len(breakpoints)):
#     target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
#     client.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), s1_all[j],target2=target2)
# client.SetArc(False)

###########################################layer welding############################################
layer=8
curve_sliced_js=np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(layer)+'.csv',delimiter=',')
positioner_js=np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(layer)+'.csv',delimiter=',')
curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(layer)+'.csv',delimiter=',')

if layer % 2==1:
    breakpoints=np.linspace(0,len(curve_sliced_js)-1,num=num_points_layer).astype(int)
else:
    breakpoints=np.linspace(len(curve_sliced_js)-1,0,num=num_points_layer).astype(int)

vd_relative=15
lam1=calc_lam_js(curve_sliced_js,robot)
lam2=calc_lam_js(positioner_js,positioner)
lam_relative=calc_lam_cs(curve_sliced_relative)
s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,breakpoints)
# print(s1_all)
target2=['MOVJ',np.degrees(positioner_js[breakpoints[0]]),10]
client.MoveL(np.degrees(curve_sliced_js[breakpoints[0]]), s1_all[0],target2=target2)

client.SetArc(True,cond_num=200)
for j in range(1,len(breakpoints)):
    target2=['MOVJ',np.degrees(positioner_js[breakpoints[j]]),10]
    client.MoveL(np.degrees(curve_sliced_js[breakpoints[j]]), s1_all[j],target2=target2)
client.SetArc(False)

client.execute_motion_program("AAA.JBI") 