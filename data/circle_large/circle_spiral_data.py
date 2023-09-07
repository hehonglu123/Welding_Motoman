import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
from general_robotics_toolbox import *
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
    base_marker_config_file=R2_marker_dir+'MA1440_'+R2_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=mti_marker_dir+'mti_'+R2_ph_dataset_date+'_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=S1_marker_dir+'D500B_'+S1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=S1_tcp_marker_dir+'positioner_tcp_marker_config.yaml')

robot_scan_positioner_base = robot_scan.T_base_basemarker.inv()*positioner.T_base_basemarker*Transform(np.eye(3),[0,0,-380])

zero_config=np.zeros(6)
# print(robot_weld.fwd(zero_config))
# print(robot_scan.fwd(zero_config))
## generate circle path and positioner js
algo_name = 'static'
circle_radius=35
circle_offset=0
path_dlambda = 0.1 # mm
layer_dh = 0.1 # mm
total_layers = 500
torch_lambda = 55
scanner_distance=110

## start generating
algo_dir=algo_name+'/'
Path(algo_dir).mkdir(exist_ok=True)
path_dir=algo_dir+'curve_sliced_relative/'
Path(path_dir).mkdir(exist_ok=True)
js_dir=algo_dir+'curve_sliced_js/'
Path(js_dir).mkdir(exist_ok=True)

## generate path and js
for l in range(total_layers):
    path_dangle=path_dlambda/circle_radius
    curve_relative = []
    positioner_js = []
    angle_range=np.append(np.arange(circle_offset+2*np.pi,circle_offset+0,-1*path_dangle),circle_offset+0)
    for theta in angle_range:
        position=np.array([circle_radius*np.cos(theta),circle_radius*np.sin(theta),l*layer_dh])
        pos_normal=np.append(position,[0,0,-1])
        curve_relative.append(pos_normal)
        positioner_js.append([np.radians(-15),theta])
        
    curve_relative=np.array(curve_relative)
    positioner_js=np.array(positioner_js)
    
    torch_bp = int(torch_lambda/path_dlambda)
    # print(torch_bp)
    # print(len(angle_range))
    # exit()
    torch_p_S1TCP = Transform(np.array([[1,0,0],[0,-1,0],curve_relative[torch_bp][3:]]).T,curve_relative[torch_bp][:3])
    torch_p = positioner.fwd(positioner_js[0],world=True)*torch_p_S1TCP
    torch_Rx=[-1,0,0]
    torch_Ry=np.cross(torch_p.R[:,-1],torch_Rx)
    torch_Ry=torch_Ry/np.linalg.norm(torch_Ry)
    torch_R=np.array([torch_Rx,torch_Ry,curve_relative[torch_bp][3:]]).T
    torch_p.R=torch_R
    r1_js = robot_weld.inv(torch_p.p,torch_p.R,last_joints=zero_config)
    r1_js = np.tile(r1_js,(len(curve_relative),1))
    
    scanner_bp = int((torch_lambda-scanner_distance)/path_dlambda)
    scanner_p_S1TCP=curve_relative[scanner_bp][:3]+np.array([0,0,95])
    # print(scanner_p_S1TCP)
    # exit()
    Ry=np.array(curve_relative[scanner_bp+1][:3]-scanner_p_S1TCP)*(-1)
    Ry=Ry/np.linalg.norm(Ry)
    Rx=np.cross(Ry,curve_relative[scanner_bp+1][3:])
    Rx=Rx/np.linalg.norm(Rx)
    scanner_p_S1TCP = Transform(np.array([Rx,Ry,curve_relative[scanner_bp][3:]]).T,scanner_p_S1TCP)
    scanner_p = robot_scan_positioner_base*positioner.fwd(positioner_js[0])*scanner_p_S1TCP
    r2_js = robot_scan.inv(scanner_p.p,scanner_p.R,last_joints=zero_config)
    r2_js = np.tile(r2_js,(len(curve_relative),1))
    # print(np.degrees(r1_js[0]))
    # print(np.degrees(r2_js[0]))
    # exit()    
    # print(len(curve_relative))
    # print(len(positioner_js))
    # print(len(r1_js))
    # print(len(r2_js))
    
    np.savetxt(path_dir+'slice'+str(l)+'_0.csv',curve_relative,delimiter=',')
    np.savetxt(js_dir+'MA2010_js'+str(l)+'_0.csv',r1_js,delimiter=',')
    np.savetxt(js_dir+'MA1440_js'+str(l)+'_0.csv',r2_js,delimiter=',')
    np.savetxt(js_dir+'D500B_js'+str(l)+'_0.csv',positioner_js,delimiter=',')

slicing_meta={}
slicing_meta['num_layer']=total_layers
slicing_meta['num_baselayers']=0
slicing_meta['line_resolution']=0.1
slicing_meta['baselayer_thickness']=0
slicing_meta['point_distance']=0.5
with open(algo_dir+'slicing.yml','w') as file:
    yaml.safe_dump(slicing_meta,file)