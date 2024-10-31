import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from general_robotics_toolbox import *

from slicing2 import slicing_uniform,cut_mesh_z_axis,visualize_objects

data_dir = "../data/face_mesh_tanja_straight/"

mesh = o3d.io.read_triangle_mesh(data_dir+"mesh_final.stl")
# Compute the vertex normals of the mesh
mesh.compute_vertex_normals()

# using the mesh vertices to create a point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices

visualize_objects([mesh])

print("Define parameters used for hidden_point_removal")

pt_map_all=[]
for x in np.linspace(-300,300,10):
    for z in np.linspace(-100,300,10):
        camera = [x, -50, z]
        radius = 13500

        _, pt_map = pcd.hidden_point_removal(camera, radius)
        pt_map_all.extend(pt_map)

print("Visualize result")
pcd_surface = pcd.select_by_index(pt_map_all)

visualize_objects([mesh,pcd_surface])

# Extract the vertices of the original mesh and points from the pointcloud
mesh_vertices = np.asarray(mesh.vertices)
pointcloud_points = np.asarray(pcd_surface.points)

# Use a set to identify vertices that are in the pointcloud
point_set = set(map(tuple, pointcloud_points))

# Identify the indices of vertices that are in the pointcloud
keep_indices = [i for i, v in enumerate(mesh_vertices) if tuple(v) in point_set]

# Create a mapping from old vertex indices to new vertex indices
index_map = -1 * np.ones(len(mesh_vertices), dtype=int)
index_map[keep_indices] = np.arange(len(keep_indices))

# Filter out the vertices and triangles based on the keep indices
new_vertices = mesh_vertices[keep_indices]
new_triangles = []
for tri in np.asarray(mesh.triangles):
    if all(v in keep_indices for v in tri):
        new_triangles.append(index_map[tri])

# Create a new mesh with filtered vertices and triangles
mesh_surface = o3d.geometry.TriangleMesh()
mesh_surface.vertices = o3d.utility.Vector3dVector(new_vertices)
mesh_surface.triangles = o3d.utility.Vector3iVector(np.array(new_triangles))
mesh_surface.compute_vertex_normals()

visualize_objects([mesh_surface,pcd_surface])

o3d.io.write_triangle_mesh(data_dir+"mesh_surface.stl", mesh_surface)

exit()




mesh_cut = cut_mesh_z_axis(mesh, 150, 120)
mesh_cut.compute_vertex_normals()

pcd = mesh_cut.sample_points_uniformly(50000)

visualize_objects([mesh_cut,pcd])

print("Define parameters used for hidden_point_removal")
# diameter=
camera = [0, -10, 135]
radius = 3000

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd_surface = pcd.select_by_index(pt_map)

visualize_objects([mesh_cut,pcd_surface])