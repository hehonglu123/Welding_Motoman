import sys,time, copy, yaml
import open3d as o3d
from matplotlib import cm

sys.path.append('../toolbox/')
from utils import *
from pointcloud_toolbox import *


def calc_error(target_points,collapsed_points):
    error_off=[]
    error_miss=[]
    for p in collapsed_points:
        error_off.append(np.linalg.norm(target_points-p,axis=1).min())
    
    for p in target_points:
        error_miss.append(np.linalg.norm(collapsed_points-p,axis=1).min())
    
    return np.array(error_off), np.array(error_miss)

def separate(scanned_points,target_points):
    num_points=10
    left_indices=[]
    right_indices=[]

    for i in range(len(scanned_points)):
        indices=np.argsort(np.linalg.norm(target_points-scanned_points[i],axis=1))[:num_points]
        normal, centroid=fit_plane(target_points[indices])
        v=vector_to_plane(scanned_points[i], centroid, normal)
        if np.dot(v,np.array([0,0,20]))<0:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices,right_indices

def collapse(left_pc,right_pc,target_points):
    num_points=10
    collapsed_surface=[]
    width=[]
    for i in range(len(left_pc)):
        indices=np.argsort(np.linalg.norm(target_points-left_pc[i],axis=1))[:num_points]
        normal, centroid=fit_plane(target_points[indices])
        v1=-vector_to_plane(left_pc[i], centroid, normal)     ###vector from surface to left
        indices_right=np.argsort(np.linalg.norm(right_pc-(left_pc[i]-2*v1),axis=1))[:num_points]
        normal_right, centroid_right=fit_plane(right_pc[indices_right])
        v2=vector_to_plane(left_pc[i]-v1, centroid_right, normal_right)
        width.append([np.linalg.norm(v1),np.linalg.norm(v2)])
        collapsed_surface.append(left_pc[i]-v1+(v1+v2)/2)
    
    return np.sum(width,axis=1), collapsed_surface

def error_map_gen(collapsed_surface,target_points):
    error_map=np.zeros(len(collapsed_surface))
    for i in range(len(collapsed_surface)):
        error_map[i]=np.linalg.norm(target_points-collapsed_surface[i],axis=1).min()
    return error_map

def visualize_pcd(show_pcd_list,point_show_normal=False):

    show_pcd_list_legacy=[]
    for obj in show_pcd_list:
        if type(obj) is o3d.cpu.pybind.t.geometry.PointCloud or type(obj) is o3d.cpu.pybind.t.geometry.TriangleMesh:
            show_pcd_list_legacy.append(obj.to_legacy())
        else:
            show_pcd_list_legacy.append(obj)

    points_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20,origin=[0,0,0])
    show_pcd_list_legacy.append(points_frame)
    o3d.visualization.draw_geometries(show_pcd_list_legacy,width=960,height=540,point_show_normal=point_show_normal)


    
dataset='cup/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

###read target points
target_points_pc=[]
target_points_pc_temp=[]
for i in range(1,slicing_meta['num_layers'],5):
    target_points_pc_temp.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',')[:,:3])
target_points_pc=copy.deepcopy(target_points_pc_temp)
for i in range(0,128,5):
    target_points_pc.append(np.loadtxt(data_dir+'curve_sliced/slice0_'+str(i)+'.csv',delimiter=',')[:,:3])
target_points_pc_temp=np.concatenate(target_points_pc_temp,axis=0)
target_points_pc=np.concatenate(target_points_pc,axis=0)

target_points=o3d.geometry.PointCloud()
target_points.points=o3d.utility.Vector3dVector(target_points_pc)


scanned_dir='../../evaluation/Cup_ER316L/'
######## read the scanned stl
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'cup.stl')
scanned_mesh.compute_vertex_normals()
scanned_mesh_temp = o3d.io.read_triangle_mesh(scanned_dir+'no_base_layer.stl')



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

left_indices,right_indices=separate(scanned_points_tranform,target_points_transform)
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

error_off,error_miss=calc_error(target_points_transform,collapsed_surface)

highlight_pc=o3d.geometry.PointCloud()
highlight_pc.points=o3d.utility.Vector3dVector([collapsed_surface[error_off.argmax()]])
highlight_pc.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([target_points,collapsed_surface_pc,highlight_pc])

print('error max: ',error_off.max(),'error avg: ',np.mean(error_off))
print(error_miss.max(),np.mean(error_miss))

error_map=error_map_gen(collapsed_surface,target_points_transform)
print(error_map)
error_map_normalized=error_map/np.max(error_map)
#convert normalized error map to color heat map
error_map_color=cm.inferno(error_map_normalized)[:,:3]
collapsed_surface_pc.colors=o3d.utility.Vector3dVector(error_map_color)
o3d.visualization.draw_geometries([collapsed_surface_pc])