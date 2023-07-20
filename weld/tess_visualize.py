import sys, glob
sys.path.append('../toolbox/')
from robot_def import *
from tesseract_env import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/flir.csv',\
	pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv',base_transformation_file='../config/MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path='../config/D500B_robot_default_config.yml',tool_file_path='../config/positioner_tcp.csv',\
	pulse2deg_file_path='../config/D500B_pulse2deg_real.csv',base_transformation_file='../config/D500B_pose.csv')

dataset='blade0.1/'
sliced_alg='auto_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

rob1_js=[]
rob2_js=[]
positioner_js=[]


for i in range(slicing_meta['num_layers']):
	num_sections=len(glob.glob(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_*.csv'))
	rob1_js_ith_layer=[]
	rob2_js_ith_layer=[]
	positioner_js_ith_layer=[]
	for x in range(num_sections):
		rob1_js_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		rob2_js_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6)))
		positioner_js_ith_layer.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,2)))
	rob1_js.append(rob1_js_ith_layer)
	rob2_js.append(rob2_js_ith_layer)
	positioner_js.append(positioner_js_ith_layer)


#link and joint names in urdf
MA2010_link_names=["MA2010_base_link","MA2010_link_1_s","MA2010_link_2_l","MA2010_link_3_u","MA2010_link_4_r","MA2010_link_5_b","MA2010_link_6_t"]
MA2010_joint_names=["MA2010_joint_1_s","MA2010_joint_2_l","MA2010_joint_3_u","MA2010_joint_4_r","MA2010_joint_5_b","MA2010_joint_6_t"]

MA1440_link_names=["MA1440_base_link","MA1440_link_1_s","MA1440_link_2_l","MA1440_link_3_u","MA1440_link_4_r","MA1440_link_5_b","MA1440_link_6_t"]
MA1440_joint_names=["MA1440_joint_1_s","MA1440_joint_2_l","MA1440_joint_3_u","MA1440_joint_4_r","MA1440_joint_5_b","MA1440_joint_6_t"]

D500B_joint_names=["D500B_joint_1","D500B_joint_2"]
D500B_link_names=["D500B_base_link","D500B_link_1","D500B_link_2"]

#Robot dictionaries, all reference by name
robot_linkname={'MA2010_A0':MA2010_link_names,'MA1440_A0':MA1440_link_names,'D500B':D500B_link_names}
robot_jointname={'MA2010_A0':MA2010_joint_names,'MA1440_A0':MA1440_joint_names,'D500B':D500B_joint_names}

t=Tess_Env('../config/urdf/motoman_cell',robot_linkname,robot_jointname)
H=copy.deepcopy(positioner.base_H)
H[:3,3]/=1000.
H[2,3]+=0.762+0.5
t.update_pose('D500B',H)
H=copy.deepcopy(robot2.base_H)
H[:3,3]/=1000.
H[2,3]+=0.762
t.update_pose('MA1440',H)

# 	t.viewer_trajectory_dual(robot.robot_name,positioner.robot_name,curve_sliced_js[i][::20],positioner_js[i][::20])
# 	time.sleep(10)
t.viewer_trajectory([robot.robot_name,robot2.robot_name,positioner.robot_name],np.hstack((rob1_js[400][0][::20],rob2_js[400][0][::20],positioner_js[400][0][::20])))
# t.viewer_trajectory_dual(robot.robot_name,positioner.robot_name,curve_sliced_base_js[1][::20],positioner_base_js[1][::20])
input("Press enter to quit")