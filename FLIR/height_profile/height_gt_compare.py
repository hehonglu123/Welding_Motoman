import numpy as np
import matplotlib.pyplot as plt
import os, yaml, pickle, inspect
import matplotlib.pyplot as plt
from flir_toolbox import *
from motoman_def import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from ultralytics import YOLO


config_dir='../../config/'

robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',d=15)
robot2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
	pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

flir_intrinsic=yaml.load(open(config_dir+'FLIR_A320.yaml'), Loader=yaml.FullLoader)


data_dir='../../../recorded_data/wall_weld_test/4043_150ipm_2024_06_18_11_16_32/'
#load model
torch_model = YOLO(os.path.dirname(inspect.getfile(flir_toolbox))+"/torch.pt")

#get the number of folders
num_layers=len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])

flame_3d_all_layers=[]
for i in range(2,num_layers):
    folder='layer_'+str(i)
    flame_3d_layer=[]
    with open(data_dir+folder+'/ir_recording.pickle', 'rb') as file:
        ir_recording = pickle.load(file)
    ir_ts=np.loadtxt(data_dir+folder+'/ir_stamps.csv', delimiter=',')
    # Load the IR recording data from the pickle file
    joint_angle=np.loadtxt(data_dir+folder+'/weld_js_exe.csv', delimiter=',')

    pixel_coord_layer=[]    #find all pixel regions to record from flame detection
    #find all pixel regions to record from flame detection
    for i in range(len(ir_recording)):
        
        ir_image = ir_recording[i]

        centroid, bbox = flame_detection_aluminum(ir_image,threshold=1.0e4,area_threshold=10)
        if centroid is not None:
            #find spatial vector ray from camera sensor
            vector=np.array([(centroid[0]-flir_intrinsic['c0'])/flir_intrinsic['fsx'],(centroid[1]-flir_intrinsic['r0'])/flir_intrinsic['fsy'],1])
            vector=vector/np.linalg.norm(vector)
            #find index closest in time of joint_angle
            joint_idx=np.argmin(np.abs(ir_ts[i]-joint_angle[:,0]))
            robot2_pose_world=robot2.fwd(joint_angle[joint_idx][8:-2],world=True)
            p2=robot2_pose_world.p
            v2=robot2_pose_world.R@vector
            robot1_pose=robot.fwd(joint_angle[joint_idx][2:8])
            p1=robot1_pose.p
            v1=robot1_pose.R[:,2]
            #find intersection point
            intersection=line_intersection(p1,v1,p2,v2)
            flame_3d_layer.append(intersection)

            ##########################################################DEBUGGING & VISUALIZATION: plot out p1,v1,p2,v2,intersection##########################################################
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(p1[0],p1[1],p1[2],c='r',label='robot1')
            # ax.quiver(p1[0],p1[1],p1[2],v1[0],v1[1],v1[2],color='r',label='robot1_ray',length=100)
            # ax.scatter(p2[0],p2[1],p2[2],c='b',label='robot2')
            # ax.quiver(p2[0],p2[1],p2[2],v2[0],v2[1],v2[2],color='b',label='robot2_ray',length=100)
            # ax.quiver(p2[0],p2[1],p2[2],robot2_pose_world.R[0,2],robot2_pose_world.R[1,2],robot2_pose_world.R[2,2],color='g',label='optical_axis',length=100)
            # ax.scatter(intersection[0],intersection[1],intersection[2],c='g',label='intersection')

            # ax.legend()
            # ax.set_xlabel('X')
            # ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            # plt.show()
    flame_3d_layer=np.array(flame_3d_layer)

    ##################################################################Layer Height Plot############################################################################################################
    # #plot the flame 3d
    # print(flame_3d_layer.shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(flame_3d_layer[:,0],flame_3d_layer[:,1],flame_3d_layer[:,2])
    # #set equal aspect ratio
    # ax.set_box_aspect([np.ptp(flame_3d_layer[:,0]),np.ptp(flame_3d_layer[:,1]),np.ptp(flame_3d_layer[:,2])])
    # plt.show()

    ####################################################################################################################################
    flame_3d_all_layers.append(flame_3d_layer)

flame_3d_all_layers_concat = np.concatenate(flame_3d_all_layers, axis=0)
##########################################################plot the flame 3d####################################################################################################################################
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(flame_3d_all_layers_concat[:,0],flame_3d_all_layers_concat[:,1],flame_3d_all_layers_concat[:,2])
# #set equal aspect ratio
# ax.set_box_aspect([np.ptp(flame_3d_all_layers_concat[:,0]),np.ptp(flame_3d_all_layers_concat[:,1]),np.ptp(flame_3d_all_layers_concat[:,2])])
# plt.show()


#fit all points onto a plane
normal, _ = fit_plane(flame_3d_all_layers_concat)
#project all points onto the plane
flame2d_all_layers_projected = project_onto_plane(flame_3d_all_layers_concat, normal)

plt.scatter(flame2d_all_layers_projected[:,0],flame2d_all_layers_projected[:,1])
plt.show()


##############################################GROUND TRUTH FROM MTI SCANNING################################################
height_profile_all_layers=[]
for i in range(2,num_layers):
    folder='layer_'+str(i)    
    # ###load ground truth height profile
    height_profile_all_layers.append(np.load(data_dir+folder+'/scans/height_profile.npy'))
height_profile_all_layers_concat = np.concatenate(height_profile_all_layers, axis=0) 
height_profile_centroid=np.mean(height_profile_all_layers_concat,axis=0)
height_profile_all_layers_concat=height_profile_all_layers_concat-height_profile_centroid


##############################################use ICP to align two groups##############################################
flame2d_centroid=np.mean(flame2d_all_layers_projected,axis=0)
flame2d_all_layers_projected=flame2d_all_layers_projected-flame2d_centroid
source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(np.hstack((flame2d_all_layers_projected, np.zeros((flame2d_all_layers_projected.shape[0], 1)))))
print(np.asarray(source.points))
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(np.hstack((height_profile_all_layers_concat, np.zeros((height_profile_all_layers_concat.shape[0], 1)))))
threshold=9999
H_guess=np.eye(4)
H_guess[:-1,:-1]=Rz(np.pi/2)
reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, H_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=999999999))
H = reg_p2p.transformation
print("Residual error after ICP: ", reg_p2p.inlier_rmse)
#visualize the result
source.transform(H)
source.paint_uniform_color([1, 0, 0])  # Red color for source
target.paint_uniform_color([0, 0, 1])  # Green color for target
o3d.visualization.draw_geometries([target,source])



# plt.plot(np.asarray(source.points)[:,0],np.asarray(source.points)[:,1],c='r',label='IR Flame Detection')
# plt.plot(np.asarray(target.points)[:,0],np.asarray(target.points)[:,1],c='b',label='MTI Scanning')
# plt.title('Height Profile Comparison')
# plt.xlabel('X')
# plt.ylabel('Z')
# plt.legend()
# plt.show()


###################################WORST CASE ERROR CALCULATION##############################################
error_all=[]
for point in np.asarray(source.points):
    distances=np.linalg.norm(np.asarray(target.points)-point,axis=1)
    error_all.append(np.min(distances))
error_all=np.array(error_all)
print('Worst case error: ',np.max(error_all))
