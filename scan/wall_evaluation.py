import copy, yaml
import open3d as o3d
from matplotlib import cm
import numpy as np
from robotics_utils import *
from result_analysis import *
from scipy.spatial import ConvexHull, Delaunay
from scan.scan_tools.animation_3d import animation_mesh
    
dataset='wall/'
sliced_alg='dense_slice/'
data_dir='../geometry_data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
height_threshold=np.loadtxt(data_dir+'curve_sliced/slice%i_0.csv'%(slicing_meta['num_layers']-1),delimiter=',')[0,2]+0.

###read target points
target_points_pc=[]
target_points_pc_temp=[]
for i in range(10,slicing_meta['num_layers']-70,5):
# for i in range(0,slicing_meta['num_layers']):
    target_points_pc_temp.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',')[:,:3]-np.array([32.5,0,0]))
target_points_pc=copy.deepcopy(target_points_pc_temp)

target_points_pc_temp=np.concatenate(target_points_pc_temp,axis=0)
target_points_pc=np.concatenate(target_points_pc,axis=0)

target_points=o3d.geometry.PointCloud()
target_points.points=o3d.utility.Vector3dVector(target_points_pc)

# scanned_mesh = o3d.io.read_triangle_mesh('../data/blade0.1/blade.stl')
# scanned_mesh.compute_vertex_normals()
# scanned_mesh_pcd = scanned_mesh.sample_points_uniformly(number_of_points=30000)
# scanned_mesh_pcd.paint_uniform_color([0.3, 0.3, 0.3])
# o3d.visualization.draw_geometries([scanned_mesh_pcd])

# scanned_dir='../data/wall_weld_test/moveL_100_baseline_weld_scan_2023_07_07_15_20_56/' # with baseline
scanned_dir='../data/wall_weld_test/moveL_100_weld_scan_2023_07_24_11_19_58/' # with correction
print(scanned_dir)
######## read the scanned stl
scanned_points = o3d.io.read_point_cloud(scanned_dir+'pcd_wall.pcd')

try:
    print("read mesh")
    scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'wall.stl')
    scanned_mesh.compute_vertex_normals()
    scanned_mesh_points = np.array(scanned_mesh.vertices)
    scanned_mesh_points_pcd = o3d.geometry.PointCloud()
    scanned_mesh_points_pcd.points = o3d.utility.Vector3dVector(scanned_mesh_points)
    R_guess,p_guess=global_alignment(scanned_mesh_points,target_points_pc_temp)
    print(R_guess,p_guess)

    threshold=5
    max_iteration=1000
    reg_p2p = o3d.pipelines.registration.registration_icp(
                scanned_mesh_points_pcd, target_points, threshold, H_from_RT(R_guess,p_guess),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    H = reg_p2p.transformation
    scanned_mesh.transform(H)
    scanned_mesh_pcd = scanned_mesh.sample_points_uniformly(number_of_points=10000)

    target_points_transform=np.array(target_points.points)
    scanned_points_tranform=np.array(scanned_mesh_pcd.points)
    left_indices,right_indices=separate_by_y(scanned_points_tranform,target_points_transform)
    print(len(left_indices),len(right_indices))

    # o3d.visualization.draw_geometries([scanned_mesh])
    # animation_mesh(scanned_mesh,rotation_angle=np.radians(1),steps=7200,sleep_time=0.01)
    # exit()

    left_pc = o3d.geometry.PointCloud()
    left_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[left_indices])
    left_pc.paint_uniform_color([0.0, 0.8, 0.0])

    right_pc = o3d.geometry.PointCloud()
    right_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[right_indices])
    right_pc.paint_uniform_color([0.7, 0.7, 0.0])

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([target_points,left_pc,right_pc])
    
    target_points_transform_pcd = o3d.geometry.PointCloud()
    target_points_transform_pcd.points = o3d.utility.Vector3dVector(target_points_pc_temp)

    target_points.paint_uniform_color([0.2, 0.2, 1])
    o3d.visualization.draw_geometries([target_points,left_pc,right_pc])
    # visualize_pcd([target_points,left_pc,right_pc])
    # o3d.visualization.draw_geometries([scanned_mesh,target_points_transform_pcd])
    # exit()
except Exception as e:
    print(e)

# remove all the points below z=threshold
cut_threshold=8.7
min_bound= (-np.inf, -10, cut_threshold)
max_bound= (np.inf, 10, np.inf)
bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
scanned_points=scanned_points.crop(bbox)
visualize_pcd([scanned_points])

## sample as pointclouds
scanned_points_temp = deepcopy(scanned_points)

## global tranformation
R_guess,p_guess=global_alignment(scanned_points_temp.points,target_points_pc_temp)
R_guess=np.eye(3)
p_guess=np.array([0,0,0])
print(R_guess,p_guess)

## sample as sparser pointclouds
scanned_points = scanned_points.voxel_down_sample(voxel_size=1)
print(len(scanned_points.points))
visualize_pcd([scanned_points])

target_points = target_points.paint_uniform_color([0, 0.0, 0.8])
scanned_points = scanned_points.paint_uniform_color([0.8, 0, 0.0])

threshold=5
max_iteration=1000
reg_p2p = o3d.pipelines.registration.registration_icp(
            scanned_points, target_points, threshold, H_from_RT(R_guess,p_guess),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
H = reg_p2p.transformation

#############MANUAL H ADJUSTMENT FOR OL TRIANGULAR WALL#############
# H_new=np.eye(4)
# H_new[:3,:3]=Ry(-np.pi/35)@H[:3,:3]
# H_new[:3,3]=H[:3,3]+np.array([2,0,-5])
##################################################################


scanned_points=scanned_points.transform(H)
target_points_transform=np.array(target_points.points)
scanned_points_tranform=np.array(scanned_points.points)

# print(np.max(scanned_points_tranform[:,2]),np.max(target_points_transform[:,2]))

###thresholding top layers
scanned_points_tranform=scanned_points_tranform[scanned_points_tranform[:,2]<height_threshold]

left_indices,right_indices=separate_by_y(scanned_points_tranform,target_points_transform)
print(len(left_indices),len(right_indices))

left_pc = o3d.geometry.PointCloud()
left_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[left_indices])
left_pc.paint_uniform_color([0.0, 0.8, 0.0])

right_pc = o3d.geometry.PointCloud()
right_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[right_indices])
right_pc.paint_uniform_color([0.7, 0.7, 0.0])

# Visualize the point cloud
# o3d.visualization.draw_geometries([target_points,left_pc,right_pc])
visualize_pcd([target_points,left_pc,right_pc])

width,collapsed_surface=collapse(np.array(left_pc.points),np.array(right_pc.points),target_points_transform)
collapsed_surface_pc=o3d.geometry.PointCloud()
collapsed_surface_pc.points=o3d.utility.Vector3dVector(collapsed_surface)
collapsed_surface_pc.paint_uniform_color([0.7, 0.7, 0.0])

print('\sigma(w): ',np.std(width),'\mu(w): ',np.average(width))

# error=calc_error_projected(target_points_transform,collapsed_surface)

# collapsed_surface_xy = np.array(collapsed_surface)[:,[0,2]]
# # Function to check if a point is inside the convex hull
# def is_point_in_hull(point, hull):
#     # Create a Delaunay triangulation of the hull vertices
#     delaunay = Delaunay(hull.points[hull.vertices])
#     # Check if the point is within the Delaunay triangulation
#     return delaunay.find_simplex(point) >= 0

# # build convex hull
# print("building convex hull")
# hull = ConvexHull(collapsed_surface_xy)
# # Plot the points and the convex hull
# plt.plot(collapsed_surface_xy[:, 0], collapsed_surface_xy[:, 1], 'o')
# for simplex in hull.simplices:
#     plt.plot(collapsed_surface_xy[simplex, 0], collapsed_surface_xy[simplex, 1], 'k-')
# plt.show()
# exit()

def closest_point_error(target_points_transform,collapsed_surface):
    error=[]
    for point in target_points_transform:
        # if point[2]<5:
        #      error.append(0.1)
        #      continue
        possible_error_1 = np.linalg.norm(collapsed_surface-point,axis=1)
        possible_error_2 = np.linalg.norm(collapsed_surface-(point+np.array([0,0,1])),axis=1)
        possible_error = np.append(possible_error_1,possible_error_2)
        error.append(np.min(possible_error))
    # np.array(error)
    # np.nan_to_num(error, copy=False, nan=np.nanmin(error))
    return np.array(error)
closest_error=closest_point_error(target_points_transform,collapsed_surface)
error=closest_error


# highlight_pc=o3d.geometry.PointCloud()
# highlight_pc.points=o3d.utility.Vector3dVector([collapsed_surface[error.argmax()]])
# highlight_pc.paint_uniform_color([0.0, 1.0, 0.0])
# o3d.visualization.draw_geometries([target_points,collapsed_surface_pc,highlight_pc])

print('error max: ',error.max(),'error avg: ',np.mean(error[::2]))

# error_display_max=2
error_display_max=3.8
print(error)
error_normalized=error/error_display_max
#convert normalized error map to color heat map
error_color=cm.plasma(error_normalized)[:,:3]
# collapsed_surface_pc.colors=o3d.utility.Vector3dVector(error_color)
target_points_transform_pcd = o3d.geometry.PointCloud()
target_points_transform_pcd.points = o3d.utility.Vector3dVector(target_points_transform)
target_points_transform_pcd.colors=o3d.utility.Vector3dVector(error_color)


z_rng = np.arange(error.max(), error.min(), (error.min()-error.max())/100)
ax = plt.subplot()
im = ax.imshow(np.vstack((z_rng, z_rng, z_rng, z_rng)).T, extent=(0,  error_display_max/20, 0,error_display_max), cmap='plasma')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('error [mm]')
plt.show()
o3d.visualization.draw_geometries([collapsed_surface_pc])
scanned_points.paint_uniform_color([0, 1, 0])
# o3d.visualization.draw_geometries([target_points_transform_pcd,scanned_points])
o3d.visualization.draw_geometries([target_points_transform_pcd])