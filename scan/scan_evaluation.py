import sys,time
import open3d as o3d

sys.path.append('../toolbox/')
from utils import *

def separate(scanned_points,target_points):
    num_points=10
    left_indices=[]
    right_indices=[]

    for i in range(len(scanned_points)):
        indices=np.argsort(np.linalg.norm(target_points-scanned_points[i],axis=1))[:num_points]
        normal, centroid=fit_plane(target_points[indices])
        v=vector_to_plane(scanned_points[i], centroid, normal)
        if v[1]<0:
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
    
    return width, collapsed_surface

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

    
data_dir='../data/blade0.1/'
scanned_dir='../../bladescan/'
######## read the scanned stl
target_mesh = o3d.io.read_triangle_mesh(data_dir+'surface.stl')
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'no_base_layer.stl')
target_mesh.compute_vertex_normals()
scanned_mesh.compute_vertex_normals()
target_mesh.scale(25.4, center=(0, 0, 0))

H=np.array([[ 2.26583472e-01,  9.16800291e-01, -3.28842146e-01,  1.20460914e+02],
            [ 9.73877061e-01, -2.18434742e-01,  6.20462179e-02,  3.92577679e+00],
            [-1.49465585e-02, -3.34310470e-01, -9.42344475e-01, -1.71426153e+01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

## sample as pointclouds
target_points = target_mesh.sample_points_uniformly(number_of_points=11000)
target_points.paint_uniform_color([0.0, 0.0, 0.8])
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=11000)
# target_points = np.array(target_mesh.vertices.points)
# scanned_points = scanned_mesh.vertices

target_points_transform=np.array(target_points.points)
scanned_points_tranform=np.array(scanned_points.transform(H).points)

left_indices,right_indices=separate(scanned_points_tranform,target_points_transform)
print(len(left_indices),len(right_indices))

left_pc = o3d.geometry.PointCloud()
left_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[left_indices])
left_pc.paint_uniform_color([0.0, 0.8, 0.0])

right_pc = o3d.geometry.PointCloud()
right_pc.points = o3d.utility.Vector3dVector(scanned_points_tranform[right_indices])
right_pc.paint_uniform_color([0.7, 0.7, 0.0])

# Visualize the point cloud
# o3d.visualization.draw_geometries([target_points,left_pc,right_pc])

width,collapsed_surface=collapse(np.array(left_pc.points),np.array(right_pc.points),target_points_transform)
collapsed_surface_pc=o3d.geometry.PointCloud()
collapsed_surface_pc.points=o3d.utility.Vector3dVector(collapsed_surface)
collapsed_surface_pc.paint_uniform_color([0.7, 0.7, 0.0])
o3d.visualization.draw_geometries([target_points,collapsed_surface_pc])