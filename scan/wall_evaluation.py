import copy, yaml
import open3d as o3d
from matplotlib import cm
import numpy as np
from robotics_utils import *
from result_analysis import *


    
dataset='right_triangle/'
sliced_alg='dense_slice/'
data_dir='../../geometry_data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)
height_threshold=np.loadtxt(data_dir+'curve_sliced/slice%i_0.csv'%(slicing_meta['num_layers']-1),delimiter=',')[0,2]+0.

###read target points
target_points_pc=[]
target_points_pc_temp=[]
for i in range(1,slicing_meta['num_layers'],5):
    target_points_pc_temp.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',')[:,:3])
target_points_pc=copy.deepcopy(target_points_pc_temp)

target_points_pc_temp=np.concatenate(target_points_pc_temp,axis=0)
target_points_pc=np.concatenate(target_points_pc,axis=0)

target_points=o3d.geometry.PointCloud()
target_points.points=o3d.utility.Vector3dVector(target_points_pc)


# scanned_dir='../../recorded_data/ER316L/streaming/'+dataset+'/bf_ol_v10_f100/'
scanned_dir='../../recorded_data/ER316L/streaming/'+dataset+'/bf_T25000/'
######## read the scanned stl
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'scan.stl')
scanned_mesh.compute_vertex_normals()
scanned_mesh_temp = o3d.io.read_triangle_mesh(scanned_dir+'scan.stl')



## sample as pointclouds
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=111000)
scanned_points_temp = scanned_mesh_temp.sample_points_uniformly(number_of_points=111000)

## global tranformation
R_guess,p_guess=global_alignment(scanned_points_temp.points,target_points_pc_temp)

## sample as sparser pointclouds
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=10000)
target_points = target_points.paint_uniform_color([0, 0.0, 0.8])
scanned_points = scanned_points.paint_uniform_color([0.8, 0, 0.0])

threshold=5
max_iteration=1000
reg_p2p = o3d.pipelines.registration.registration_icp(
            scanned_points, target_points, threshold, H_from_RT(R_guess,p_guess),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
H = reg_p2p.transformation
scanned_points=scanned_points.transform(H)


target_points_transform=np.array(target_points.points)
scanned_points_tranform=np.array(scanned_points.points)
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
o3d.visualization.draw_geometries([target_points,left_pc,right_pc])

width,collapsed_surface=collapse(np.array(left_pc.points),np.array(right_pc.points),target_points_transform)
collapsed_surface_pc=o3d.geometry.PointCloud()
collapsed_surface_pc.points=o3d.utility.Vector3dVector(collapsed_surface)
collapsed_surface_pc.paint_uniform_color([0.7, 0.7, 0.0])

print('\sigma(w): ',np.std(width),'\mu(w): ',np.average(width))

error=calc_error_projected(target_points_transform,collapsed_surface)

highlight_pc=o3d.geometry.PointCloud()
highlight_pc.points=o3d.utility.Vector3dVector([collapsed_surface[error.argmax()]])
highlight_pc.paint_uniform_color([0.0, 1.0, 0.0])
o3d.visualization.draw_geometries([target_points,collapsed_surface_pc,highlight_pc])

print('error max: ',error.max(),'error avg: ',np.mean(error))

error_display_max=2
print(error)
error_normalized=error/error_display_max
#convert normalized error map to color heat map
error_color=cm.inferno(error_normalized)[:,:3]
collapsed_surface_pc.colors=o3d.utility.Vector3dVector(error_color)


z_rng = np.arange(error.max(), error.min(), (error.min()-error.max())/100)
ax = plt.subplot()
im = ax.imshow(np.vstack((z_rng, z_rng, z_rng, z_rng)).T, extent=(0,  error_display_max/20, 0,error_display_max), cmap='inferno')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('error [mm]')
plt.show()
o3d.visualization.draw_geometries([collapsed_surface_pc])