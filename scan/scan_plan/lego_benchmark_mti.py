from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
sys.path.append('../scan_plan/')
sys.path.append('../scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import matplotlib

zero_config=np.zeros(6)
config_dir='../../config/'
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config.yaml')


#### change base H to calibrated ones ####
robot_scan.base_H = H_from_RT(robot_scan.T_base_basemarker.R,robot_scan.T_base_basemarker.p)
positioner.base_H = H_from_RT(positioner.T_base_basemarker.R,positioner.T_base_basemarker.p)
T_to_base = Transform(np.eye(3),[0,0,-380])
positioner.base_H = np.matmul(positioner.base_H,H_from_RT(T_to_base.R,T_to_base.p))

data_dir='../../data/lego_brick/mti/'
Path(data_dir).mkdir(exist_ok=True)
out_scan_dir = data_dir+'scans/'
Path(out_scan_dir).mkdir(exist_ok=True)

data_collect=False

print(robot_scan.fwd(zero_config))

if data_collect:
    # MTI connect to RR
    robot_client=MotionProgramExecClient()
    mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
    mti_client.setExposureTime("25")

    ## path
    q_bp1=np.radians([[[18.7118,46.2071,-14.7867,1.7969,-30.7251,74.5963]],[[26.1357,50.3446,-6.9595,1.2165,-34.5150,67.7113]]])

    to_start_speed=5
    scan_speed=10

    ## move to start
    q3=[-15,180]
    mp=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=positioner.pulse2deg)
    mp.MoveJ(q3,10,0)
    robot_client.execute_motion_program(mp)

    mp = MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    mp.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0)
    robot_client.execute_motion_program(mp)

    ## motion start
    mp = MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    # # routine motion
    # for path_i in range(1,len(q_bp1)-1):
    #     mp.MoveL(np.degrees(q_bp1[path_i][0]), scan_speed)
    mp.MoveL(np.degrees(q_bp1[-1][0]), scan_speed, 0)

    robot_client.execute_motion_program_nonblocking(mp)
    ###streaming
    robot_client.StartStreaming()
    start_time=time.time()
    state_flag=0
    joint_recording=[]
    robot_stamps=[]
    mti_recording=[]
    r_pulse2deg = robot_scan.pulse2deg
    while True:
        if state_flag & 0x08 == 0 and time.time()-start_time>1.:
            break
        res, data = robot_client.receive_from_robot(0.01)
        if res:
            joint_angle=np.radians(np.divide(np.array(data[26:32]),r_pulse2deg))
            state_flag=data[16]
            joint_recording.append(joint_angle)
            timestamp=data[0]+data[1]*1e-9
            robot_stamps.append(timestamp)
            ###MTI scans YZ point from tool frame
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
    robot_client.servoMH(False)

    scan_to_home_st = time.time()
    mti_recording=np.array(mti_recording)
    q_out_exe=joint_recording

    # input("Press Enter to Move Home")
    # move robot to home
    q2=np.zeros(6)
    q2[0]=90
    mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    mp.MoveJ(q2,to_start_speed,0)
    robot_client.execute_motion_program(mp)
    #####################
    # exit()

    print("Total exe len:",len(q_out_exe))
    ## save traj
    # save poses
    np.savetxt(out_scan_dir + 'scan_js_exe.csv',q_out_exe,delimiter=',')
    np.savetxt(out_scan_dir + 'scan_robot_stamps.csv',robot_stamps,delimiter=',')
    with open(out_scan_dir + 'mti_scans.pickle', 'wb') as file:
        pickle.dump(mti_recording, file)
    print('Total scans:',len(mti_recording))

q_out_exe=np.loadtxt(out_scan_dir + 'scan_js_exe.csv',delimiter=',')
robot_stamps=np.loadtxt(out_scan_dir +'scan_robot_stamps.csv',delimiter=',')
with open(out_scan_dir+ 'mti_scans.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

scan_process = ScanProcess(robot_scan,positioner)
q_init_table=np.radians([-15,180])
pcd = scan_process.pcd_register_mti(mti_recording,q_out_exe,robot_stamps,static_positioner_q=q_init_table)
# visualize_pcd([pcd])
o3d.io.write_point_cloud(out_scan_dir+'raw_pcd.pcd',pcd)
pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                    min_bound=(-1e5,-1e5,10),max_bound=(1e5,1e5,30),cluster_based_outlier_remove=True,cluster_neighbor=0.75,min_points=150)
visualize_pcd([pcd])
o3d.io.write_point_cloud(out_scan_dir+'processed_pcd.pcd',pcd)

pcd_combined=pcd
######## compare with scan mesh
scan_mesh=o3d.io.read_triangle_mesh(data_dir+'../cad/scan_mesh.stl')
scan_mesh.compute_vertex_normals()
# visualize_pcd([scan_mesh])
scan_mesh_points = o3d.io.read_point_cloud(data_dir+'../cad/scan_mesh_points.pcd')
## register points (dont need this after the motion trackers (?))
min_bound = (-1,-1,9)
max_bound = (143.1+5,15.8+1,30.6+1)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
scan_mesh_points=scan_mesh_points.crop(bbox)

trans_guess=np.eye(4)
trans_guess[0,3]=70
trans_guess[1,3]=10
pcd_combined.transform(trans_guess)
visualize_pcd([pcd_combined,scan_mesh_points])

pcd_combined_reg = pcd_combined.voxel_down_sample(voxel_size=0.5)

trans_init=np.eye(4)
threshold = 20
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_combined_reg, scan_mesh_points, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
# print(reg_p2p)
print(reg_p2p.transformation)
# visualize_pcd([pcd_combined,scan_mesh])
pcd_combined.transform(reg_p2p.transformation)
# pcd_combined.paint_uniform_color([1, 0.706, 0])
# scan_mesh_points.paint_uniform_color([0, 0.651, 0.929])
visualize_pcd([pcd_combined,scan_mesh_points])
visualize_pcd([pcd_combined,scan_mesh])

##

## crop scan mesh surfaces
# normals = np.asarray(scan_mesh.triangle_normals)
# normal_z_id = np.squeeze(np.argwhere(np.abs(normals[:,2])==1))
scan_mesh_target=deepcopy(scan_mesh)
# scan_mesh_target.triangles = o3d.utility.Vector3iVector(
#     np.asarray(scan_mesh.triangles)[normal_z_id, :])
# scan_mesh_target.triangle_normals = o3d.utility.Vector3dVector(
#     np.asarray(scan_mesh.triangle_normals)[normal_z_id, :])
# visualize_pcd([scan_mesh_target])
# visualize_pcd([scan_mesh_target,pcd_combined])
## calculate distance to mesh
scan_mesh_target_t = o3d.t.geometry.TriangleMesh.from_legacy(scan_mesh_target)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(scan_mesh_target_t)
query_points = o3d.core.Tensor(np.asarray(pcd_combined.points), dtype=o3d.core.Dtype.Float32)
unsigned_distance = scene.compute_distance(query_points)
unsigned_distance = unsigned_distance.numpy()
min_dist=np.min(unsigned_distance)
max_dist=np.max(unsigned_distance)
print("unsigned distance", unsigned_distance)
print("unsigned distance min", min_dist)
print("unsigned distance max", max_dist)
print("Unsigned distance mean", np.mean(unsigned_distance))
print("Unsigned distance std", np.std(unsigned_distance))
low_bound=0
high_bound=max_dist+0.1
color_dist = plt.get_cmap("rainbow")((unsigned_distance-low_bound)/(high_bound-low_bound))
pcd_combined_dist_scan=deepcopy(pcd_combined)
pcd_combined_dist_scan.colors = o3d.utility.Vector3dVector(color_dist[:, :3])

visualize_pcd([pcd_combined_dist_scan,scan_mesh])

#### histogram
N_points = len(unsigned_distance)
n_bins = 100
fig = plt.figure()
ax = fig.add_axes([0.05, 0.2, 0.9, 0.7])
ax1 = fig.add_axes([0.05, 0.05, 0.9, 0.1])

# We can set the number of bins with the *bins* keyword argument.
N, bins, patches = ax.hist(unsigned_distance, bins=n_bins)
# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch, thisbin in zip(fracs, patches,bins):
    color = plt.cm.rainbow((thisbin-low_bound)/(high_bound-low_bound))
    thispatch.set_facecolor(color)
cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=plt.cm.rainbow,
                                norm= matplotlib.colors.Normalize(vmin=low_bound, vmax=high_bound),
                                orientation='horizontal')
plt.show()