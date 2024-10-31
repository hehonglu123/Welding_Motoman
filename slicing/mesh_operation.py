import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from general_robotics_toolbox import *

from slicing2 import slicing_uniform,cut_mesh_z_axis

def visualize_meshes(mesh):
    # Create a coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # visualize the mesh
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=900)
    # Visualize the mesh along with the coordinate frame
    if type(mesh) == list:
        for m in mesh:
            vis.add_geometry(m)
    else:
        vis.add_geometry(mesh)
    vis.add_geometry(coordinate_frame)
    vis.run()
    vis.destroy_window()

# data_dir = "../data/eric_mesh/"
# data_dir = "../data/face_mesh_tanja/"
data_dir = "../data/face_mesh_tanja_straight/"

# Read the STL file
# mesh = o3d.io.read_triangle_mesh(data_dir+"eric_mesh.stl")
# mesh = o3d.io.read_triangle_mesh(data_dir+"mesh_cut.stl")
mesh = o3d.io.read_triangle_mesh(data_dir+"mesh.stl")
# Compute the vertex normals of the mesh
mesh.compute_vertex_normals()



hand_tune_location = False
if hand_tune_location:
    # Define a translation vector
    translation = [0, 0, 0]
    # Define a rotation matrix (example: 45 degrees around the Z axis)
    rotation_xyz = [0, 0, 0]
    while True:

        print(f"Translation: {translation}")
        print(f"Rotation matrix:\n{rotation_xyz}")

        # Apply the translation and rotation to the mesh
        mesh.translate(translation)
        mesh.rotate(mesh.get_rotation_matrix_from_xyz(rotation_xyz) , center=(0, 0, 0))

        visualize_meshes(mesh)

        # get new translation and rotation from user input
        try:
            translation = [float(x) for x in input("Enter translation vector (x y z): ").split()]
            rotation_xyz = [np.radians(float(x)) for x in input("Enter rotation matrix (x y z): ").split()]
        except ValueError:
            print("Invalid input. Breaking...")
            break

    # save the mesh
    o3d.io.write_triangle_mesh(data_dir+"mesh_transformed.stl", mesh)

mesh = o3d.io.read_triangle_mesh(data_dir+"mesh_transformed.stl")
# Compute the vertex normals of the mesh
mesh.compute_vertex_normals()

# move the lowset vertex to the origin
vertices = np.asarray(mesh.vertices)
lowest_vertex = np.argmin(vertices[:,2])
translation = -vertices[lowest_vertex]
mesh.translate(translation)

visualize_meshes(mesh)

# rotate the mesh around the x axis
# mesh.rotate(mesh.get_rotation_matrix_from_xyz([np.radians(30), 0, np.radians(10)]), center=(0, 0, 0))

# # move in z direction
translation = [0, -5, -15]
mesh.translate(translation)

mesh.rotate(mesh.get_rotation_matrix_from_xyz([np.radians(-10),0,np.radians(90)]), center=(0, 0, 0))

visualize_meshes(mesh)

# draw a plane and visualize it in open3d
# Define the plane parameters
# plane_center = [0, 0, 0]
# plane_normal = [0, 0, 1]
# plane_size = 200
# # Create a plane mesh
# plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.1)
# plane.translate([-plane_size / 2, -plane_size / 2, -0.05])
# plane.paint_uniform_color([1,0,0])

mesh = mesh.subdivide_midpoint(number_of_iterations=1)
# Find triangles with vertices on both sides of the z-axis
triangles = np.asarray(mesh.triangles)
vertices = np.asarray(mesh.vertices)

# Get the mask of vertices that have z <= threshold
mask =  np.logical_and(vertices[:, 2] <= 5,vertices[:, 2] >= -5)

# Filter vertices based on the mask
new_vertices = vertices[mask]

# Create a mapping from old vertex index to new vertex index
old_to_new_indices = -1 * np.ones(len(vertices), dtype=int)
old_to_new_indices[mask] = np.arange(len(new_vertices))

# Filter out triangles that have all vertices retained
new_triangles = []
for triangle in triangles:
    if mask[triangle[0]] and mask[triangle[1]] and mask[triangle[2]]:
        new_triangles.append([old_to_new_indices[i] for i in triangle])
new_triangles = np.asarray(new_triangles)

# Create a new mesh with the filtered vertices and triangles
mesh_removed = o3d.geometry.TriangleMesh(
    vertices=o3d.utility.Vector3dVector(new_vertices),
    triangles=o3d.utility.Vector3iVector(new_triangles),
)

pcd = mesh_removed.sample_points_uniformly(number_of_points=50000)
# pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
pcd_arr = np.asarray(pcd.points)

bottom_edge = slicing_uniform(pcd_arr,0,threshold=0.01)

# find the sequence of the bottom edge
y_sort = np.argsort(bottom_edge[:,1])[::-1]
for i in y_sort:
    if bottom_edge[i,0]<0:
        bottom_edge_start = np.copy(bottom_edge[i])
        break

bottom_edge_sort = [bottom_edge_start]
dist_sort_arg = np.argsort(np.linalg.norm(bottom_edge - bottom_edge_sort[-1], axis=1))
bottom_edge = np.delete(bottom_edge, dist_sort_arg[0], axis=0)
# iterative find the next closest point
for i in range(len(bottom_edge)-1):
    dist_sort_arg = np.argsort(np.linalg.norm(bottom_edge - bottom_edge_sort[-1], axis=1))
    next_point = bottom_edge[dist_sort_arg[0]]
    bottom_edge_sort.append(next_point)
    # remove the point from the list
    bottom_edge = np.delete(bottom_edge, dist_sort_arg[0], axis=0)
bottom_edge = np.array(bottom_edge_sort)

# Plot the bottom edge with color mapping
fig = plt.figure()
ax = fig.add_subplot(111)
# Create a color map
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=0, vmax=len(bottom_edge))
# Plot each point with a color corresponding to its index
for i in range(len(bottom_edge)):
    ax.scatter(bottom_edge[i, 0], bottom_edge[i, 1], color=cmap(norm(i)), s=10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('Bottom Edge with Color Mapping')
plt.show()

# for triangle in triangles:
#     z_values = vertices[triangle, 2]
#     if (z_values <= 0).any() and (z_values > 0).any():
#         # find the vertices of the triangle <= 0
#         z_values_0 = z_values <= 0
#         triangle_index = triangle[z_values_0]
#         vertices[triangle_index, 2] = 0
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# # Remove triangles where all 3 vertices have z < 0
# triangles_to_remove = []
# for i, triangle in enumerate(triangles):
#     z_values = vertices[triangle, 2]
#     if (z_values < 0).all():
#         triangles_to_remove.append(i)
# # Create a new mesh without the unwanted triangles
# mesh.remove_triangles_by_index(triangles_to_remove)

# Add the plane to the visualization
visualize_meshes([mesh,pcd])


# for z in range(0,100,5):
#     mesh_cut = cut_mesh_z_axis(mesh, z+5,z-5)
#     pcd = mesh_cut.sample_points_uniformly(number_of_points=30000)
#     visualize_meshes([mesh,pcd])


# save mesh
o3d.io.write_triangle_mesh(data_dir+"mesh_final.stl", mesh)

# save the bottom edge path
np.savetxt(data_dir+"bottom_edge_raw.csv", bottom_edge, delimiter=",")