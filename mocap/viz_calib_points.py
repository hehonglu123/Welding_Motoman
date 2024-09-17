import numpy as np 
from motoman_def import *
from general_robotics_toolbox import *
from matplotlib import pyplot as plt

# points = [m6,m7,m8,m9,m10,m11]

points=np.array([
[  55.09985325, -177.07850729,  343.63096401],
[  64.17453863, -141.18725139,  322.83182291],
[  30.03792868, -136.04187084,  408.77403559],
[ -33.7741809,  -138.3807089,   284.43658125],
[ -43.22753248, -175.0488713,   304.43420322],
[ -68.6888113,  -134.3290627,   368.96438777]])

y_axis_vec_1 = points[5] - points[2] # m11-m8
y_axis_vec_2 = points[4] - points[0] # m10-m6
y_axis_vec_3 = points[3] - points[1] # m9-m7
z_axis_vec_1 = points[4] - points[5] # m10-m11
z_axis_vec_2 = points[0] - points[2] # m6-m8
origin = (points[2]+points[5])/2 # (m8+m11)/2

y_axis = np.mean([y_axis_vec_1, y_axis_vec_2,y_axis_vec_3], axis=0)
y_axis = y_axis/np.linalg.norm(y_axis)
z_axis = np.mean([z_axis_vec_1, z_axis_vec_2], axis=0)
z_axis = z_axis-np.dot(z_axis, y_axis)*y_axis
z_axis = z_axis/np.linalg.norm(z_axis)
x_axis = np.cross(y_axis, z_axis)
x_axis = x_axis/np.linalg.norm(x_axis)
T_camera_flange = Transform(np.array([x_axis, y_axis, z_axis]).T, origin)
T_camera_flange = H_from_RT(T_camera_flange.R, T_camera_flange.p)

print(T_camera_flange)

T_frame_camera = H_from_RT(Ry(np.radians(15.3)), np.array([-60.69, 0, -221.848]))

T_frame_flange = np.dot(T_camera_flange,T_frame_camera)

print(T_frame_flange)

print('p',T_frame_flange[:3,3])
print('R',T_frame_flange[:3,:3])

exit()
dataset_date='09162024'

config_dir='../config/'

robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+dataset_date+'_marker_config.yaml')

T_fwd = robot.fwd(np.zeros(6))
T = H_from_RT(T_fwd.R, T_fwd.p)
points = np.array([np.dot(T, np.array([p[0], p[1], p[2], 1]))[:3] for p in points])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2])
set_axes_equal(ax)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()