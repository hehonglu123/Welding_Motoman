import numpy as np
import yaml
from pathlib import Path
from motoman_def import *

baselayer_length = 120
layer_length = 110
baselayernum = 2
baselayer_resolution = 2.75
layer_resolution = 0.1
layer_num = 440
path_dl = 0.025

positioner_joints = np.radians([-15,180])

##### save meta data #####
with open('sliced_meta.yml', 'w') as file:
    meta = {
        'baselayer_length': baselayer_length,
        'layer_length': layer_length,
        'baselayer_num': baselayernum,
        'baselayer_resolution': baselayer_resolution,
        'layer_resolution': layer_resolution,
        'layer_num': layer_num,
        'path_dl': path_dl
    }
    yaml.dump(meta, file)

##### generate wall in positioner tcp frame #####
y_position = 45

Path('curve_sliced_relative').mkdir(parents=True, exist_ok=True)
data_dir = 'curve_sliced_relative/'
## baselayers
baselayers = []
for n in range(baselayernum):
    layer = np.zeros((int(baselayer_length/path_dl+1), 6))
    layer[:, 0] = np.linspace(0, baselayer_length, int(baselayer_length/path_dl+1)) - baselayer_length/2
    layer[:, 1] = y_position
    layer[:, 2] = n*baselayer_resolution
    layer[:, 3:] = np.array([0, 0, -1])
    np.savetxt(data_dir+f"baselayer{n}_0.csv", layer, delimiter=",")
    baselayers.append(layer)

## layers
curve_layers = []
for n in range(layer_num):
    layer = np.zeros((int(layer_length/path_dl+1), 6))
    layer[:, 0] = np.linspace(0, layer_length, int(layer_length/path_dl+1)) - layer_length/2
    layer[:, 1] = y_position
    layer[:, 2] = baselayernum*baselayer_resolution + n*layer_resolution
    layer[:, 3:] = np.array([0, 0, -1])
    np.savetxt(data_dir+f"slice{n}_0.csv", layer, delimiter=",")
    curve_layers.append(layer)

exit()

##### generate robot js ######
Path('curve_sliced_js').mkdir(parents=True, exist_ok=True)
data_dir = 'curve_sliced_js/'

## define the robot
config_dir='../../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')

# get fujicam standoff distance
# Move the fujicam frame along the z-axis with the distance of the standoff distance
# will locate the frame onto 
# the plane perpendicular to the weldgun axis and passing through the weldgun TCP
T_weldgun = robot_weld.fwd(np.zeros(6))
T_scanner = robot_scan.fwd(np.zeros(6))
fujicam_standoff_d = np.dot((T_weldgun.p-T_scanner.p),T_weldgun.R[:3,2])/np.dot(T_scanner.R[:3,2],T_weldgun.R[:3,2])

robot_scan_motion=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=fujicam_standoff_d,tool_file_path=config_dir+'fujicam.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

r_weld_z = robot_weld.fwd(np.zeros(6))
r_scan_z = robot_scan_motion.fwd(np.zeros(6))
T_weld_scan = r_weld_z.inv()*r_scan_z
weld_scan_vec = T_weld_scan.p/np.linalg.norm(T_weld_scan.p)
rotate_y_direction = subproblem1(weld_scan_vec, np.array([0,1,0]), np.array([0,0,1]))
dist_weld_scan = np.linalg.norm(T_weld_scan.p)
dist_weld_scan_index = np.round(dist_weld_scan/path_dl).astype(int)

## baselayers
for n, layer in enumerate(baselayers):
    ##### find the scanner direction (lead or lag)
    positioner_tcp = positioner.fwd(positioner_joints, world=True)
    R_z = positioner_tcp.R@layer[0, 3:] # Rz
    # get Ry
    layer_weld_scan_vec = layer[dist_weld_scan_index,:3]-layer[0,:3]
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
    layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    R_y = rot(R_z, rotate_y_direction)@layer_weld_scan_vec
    R_x = np.cross(R_y, R_z) # Rx

    choices = []
    for i in range(2):
        weldgun_R = np.array([R_x, ((-1)**i)*R_y, R_z]).T
        weldgun_p = positioner_tcp.R@layer[0, :3] + positioner_tcp.p
        angles = robot_weld.inv(weldgun_p, weldgun_R)
        if len(angles)==0:
            choices.append(999999999999999999)
        else:
            choices.append(np.min(np.linalg.norm(angles-np.zeros(6), axis=1)))
    if np.argmin(choices) == 0:
        lead_lag = 0 # scanner leading
    else:
        lead_lag = 1 # scanner lagging

    print("layer", n, "lead_lag", lead_lag)

    ##### solve ik
    curve_js = [np.zeros(6)]
    for i in range(layer.shape[0]):
        positioner_tcp = positioner.fwd(positioner_joints, world=True)
        try:
            R_z = positioner_tcp.R@layer[i, 3:] # Rz
            # get Ry
            layer_weld_scan_vec = layer[dist_weld_scan_index+i,:3]-layer[i,:3]
            layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
            layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
            layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
            layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
            R_y = (-1)**(lead_lag)*rot(R_z, rotate_y_direction)@layer_weld_scan_vec
            R_x = np.cross(R_y, R_z) # Rx
            weldgun_R = np.array([R_x, R_y, R_z]).T
        except IndexError:
            pass
        weldgun_p = positioner_tcp.R@layer[i, :3] + positioner_tcp.p
        curve_js.append(robot_weld.inv(weldgun_p, weldgun_R, curve_js[-1])[0])
    curve_js = np.array(curve_js[1:])
    np.savetxt(data_dir+f"MA2010_base_js{n}_0.csv", curve_js, delimiter=",")

    ##### solve ik for the extra scanning motion
    positioner_tcp = positioner.fwd(positioner_joints, world=True)
    positioner_tcp_inv = positioner_tcp.inv()
    R_z = positioner_tcp.R@layer[i, 3:] # Rz
    # get Ry
    if lead_lag == 0:
        layer_weld_scan_vec = layer[dist_weld_scan_index,:3]-layer[0,:3]
    else:
        layer_weld_scan_vec = layer[-1-dist_weld_scan_index,:3]-layer[-1,:3]
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
    layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    R_y = rot(R_z, rotate_y_direction)@layer_weld_scan_vec
    R_x = np.cross(R_y, R_z) # Rx
    weldgun_R = np.array([R_x, R_y, R_z]).T
    starting_id = dist_weld_scan_index if lead_lag == 0 else -dist_weld_scan_index-1
    ending_id = -1 if lead_lag == 0 else 0
    step_direction = -1 if lead_lag == 0 else 1
    curve_js = [curve_js[0]] if lead_lag == 0 else [curve_js[-1]]
    baselayer_scan = []
    for i in range(starting_id, ending_id, step_direction):
        weld_scan_vec_base = weldgun_R@T_weld_scan.p
        weldgun_p = positioner_tcp.R@layer[i, :3] + positioner_tcp.p
        weldgun_p = weldgun_p - weld_scan_vec_base
        curve_js.append(robot_weld.inv(weldgun_p, weldgun_R, curve_js[-1])[0])
        weldgun_p_relative = positioner_tcp_inv.R@weldgun_p + positioner_tcp_inv.p
        weldgun_R_relative = positioner_tcp_inv.R@weldgun_R
        baselayer_scan.append(np.append(weldgun_p_relative, weldgun_R_relative[:,-1]))
    curve_js = np.array(curve_js[1:])
    np.savetxt(data_dir+f"MA2010_base_js{n}_scanOnly.csv", curve_js, delimiter=",")
    np.savetxt('curve_sliced_relative/'+f"baselayer{n}_scanOnly.csv", np.array(baselayer_scan), delimiter=",")

## layers
for n, layer in enumerate(curve_layers):
    ##### find the scanner direction (lead or lag)
    positioner_tcp = positioner.fwd(positioner_joints, world=True)
    R_z = positioner_tcp.R@layer[0, 3:] # Rz
    # get Ry
    layer_weld_scan_vec = layer[dist_weld_scan_index,:3]-layer[0,:3]
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
    layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    R_y = rot(R_z, rotate_y_direction)@layer_weld_scan_vec
    R_x = np.cross(R_y, R_z) # Rx

    choices = []
    for i in range(2):
        weldgun_R = np.array([R_x, ((-1)**i)*R_y, R_z]).T
        weldgun_p = positioner_tcp.R@layer[0, :3] + positioner_tcp.p
        angles = robot_weld.inv(weldgun_p, weldgun_R)
        if len(angles)==0:
            choices.append(999999999999999999)
        else:
            choices.append(np.min(np.linalg.norm(angles-np.zeros(6), axis=1)))
    if np.argmin(choices) == 0:
        lead_lag = 0 # scanner leading
    else:
        lead_lag = 1 # scanner lagging

    print("layer", n, "lead_lag", lead_lag)

    ##### solve ik
    curve_js = [np.zeros(6)]
    for i in range(layer.shape[0]):
        positioner_tcp = positioner.fwd(positioner_joints, world=True)
        try:
            R_z = positioner_tcp.R@layer[i, 3:] # Rz
            # get Ry
            layer_weld_scan_vec = layer[dist_weld_scan_index+i,:3]-layer[i,:3]
            layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
            layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
            layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
            layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
            R_y = (-1)**(lead_lag)*rot(R_z, rotate_y_direction)@layer_weld_scan_vec
            R_x = np.cross(R_y, R_z) # Rx
            weldgun_R = np.array([R_x, R_y, R_z]).T
        except IndexError:
            pass
        weldgun_p = positioner_tcp.R@layer[i, :3] + positioner_tcp.p
        curve_js.append(robot_weld.inv(weldgun_p, weldgun_R, curve_js[-1])[0])
    curve_js = np.array(curve_js[1:])
    np.savetxt(data_dir+f"MA2010_js{n}_0.csv", curve_js, delimiter=",")

    ##### solve ik for the extra scanning motion
    positioner_tcp = positioner.fwd(positioner_joints, world=True)
    positioner_tcp_inv = positioner_tcp.inv()
    R_z = positioner_tcp.R@layer[i, 3:] # Rz
    # get Ry
    if lead_lag == 0:
        layer_weld_scan_vec = layer[dist_weld_scan_index,:3]-layer[0,:3]
    else:
        layer_weld_scan_vec = layer[-1-dist_weld_scan_index,:3]-layer[-1,:3]
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    layer_weld_scan_vec = positioner_tcp.R@layer_weld_scan_vec
    layer_weld_scan_vec = layer_weld_scan_vec-np.dot(layer_weld_scan_vec, R_z)*R_z
    layer_weld_scan_vec /= np.linalg.norm(layer_weld_scan_vec)
    R_y = rot(R_z, rotate_y_direction)@layer_weld_scan_vec
    R_x = np.cross(R_y, R_z) # Rx
    weldgun_R = np.array([R_x, R_y, R_z]).T
    starting_id = dist_weld_scan_index if lead_lag == 0 else -dist_weld_scan_index-1
    ending_id = -1 if lead_lag == 0 else 0
    step_direction = -1 if lead_lag == 0 else 1
    curve_js = [curve_js[0]] if lead_lag == 0 else [curve_js[-1]]
    layer_scan = []
    for i in range(starting_id, ending_id, step_direction):
        weld_scan_vec_base = weldgun_R@T_weld_scan.p
        weldgun_p = positioner_tcp.R@layer[i, :3] + positioner_tcp.p
        weldgun_p = weldgun_p - weld_scan_vec_base
        curve_js.append(robot_weld.inv(weldgun_p, weldgun_R, curve_js[-1])[0])
        weldgun_p_relative = positioner_tcp_inv.R@weldgun_p + positioner_tcp_inv.p
        weldgun_R_relative = positioner_tcp_inv.R@weldgun_R
        layer_scan.append(np.append(weldgun_p_relative, weldgun_R_relative[:,-1]))
    curve_js = np.array(curve_js[1:])
    np.savetxt(data_dir+f"MA2010_js{n}_scanOnly.csv", curve_js, delimiter=",")
    np.savetxt('curve_sliced_relative/'+f'slice{n}_scanOnly.csv', np.array(layer_scan), delimiter=",")
