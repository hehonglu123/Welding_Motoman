import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
from general_robotics_toolbox import *
from matplotlib import pyplot as plt
import sys
sys.path.append('../../toolbox/')
from robot_def import *

R1_ph_dataset_date='0801'
R2_ph_dataset_date='0725'
S1_ph_dataset_date='0725'
config_dir='../../config/'
R1_marker_dir=config_dir+'MA2010_marker_config/'
weldgun_marker_dir=config_dir+'weldgun_marker_config/'
R2_marker_dir=config_dir+'MA1440_marker_config/'
mti_marker_dir=config_dir+'mti_marker_config/'
S1_marker_dir=config_dir+'D500B_marker_config/'
S1_tcp_marker_dir=config_dir+'positioner_tcp_marker_config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=R1_marker_dir+'MA2010_'+R1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=weldgun_marker_dir+'weldgun_'+R1_ph_dataset_date+'_marker_config.yaml')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=R2_marker_dir+'MA1440_marker_config.yaml',tool_marker_config_file=mti_marker_dir+'mti_'+R2_ph_dataset_date+'_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=S1_marker_dir+'D500B_marker_config.yaml',tool_marker_config_file=S1_tcp_marker_dir+'positioner_tcp_marker_config.yaml')

robot_scan_positioner_base = robot_scan.T_base_basemarker.inv()*positioner.T_base_basemarker*Transform(np.eye(3),[0,0,-380])

zero_config=np.zeros(6)

algo_name = 'static'
x_start=-70
x_end=70
sample_dx = 0.1 # mm
amplitude=35
layer_dh = 0.1 # mm
total_layers = 500
torch_lambda = 0
scanner_distance=110

## start generating
algo_dir=algo_name+'/'
Path(algo_dir).mkdir(exist_ok=True)
path_dir=algo_dir+'curve_sliced_relative/'
Path(path_dir).mkdir(exist_ok=True)
js_dir=algo_dir+'curve_sliced_js/'
Path(js_dir).mkdir(exist_ok=True)

## generate path and js
# total_layers=1
for l in range(total_layers):
    curve_relative = []
    
    posx_range=np.arange(x_start,x_end+sample_dx,sample_dx)
    for posx in posx_range:
        t=posx-x_start
        fr = np.pi / 180.0 * (t/32) ** (1.)
        posy = amplitude * np.sin(np.multiply(fr,t))
        
        position=np.array([posx,posy,l*layer_dh])
        pos_normal=np.append(position,[0,0,-1])
        curve_relative.append(pos_normal)
        
    curve_relative=np.array(curve_relative)
    
    # plt.plot(curve_relative[:,0],curve_relative[:,1],'-o')
    # plt.axis("equal")
    # plt.show()
    
    np.savetxt(path_dir+'slice'+str(l)+'_0.csv',curve_relative,delimiter=',')

slicing_meta={}
slicing_meta['num_layer']=total_layers
slicing_meta['num_baselayers']=0
slicing_meta['line_resolution']=0.1
slicing_meta['baselayer_thickness']=0
slicing_meta['point_distance']=0.5
with open(algo_dir+'slicing.yml','w') as file:
    yaml.safe_dump(slicing_meta,file)