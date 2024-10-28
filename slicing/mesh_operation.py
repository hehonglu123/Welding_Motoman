import numpy as np
import open3d as o3d
from general_robotics_toolbox import *

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

data_dir = "../data/eric_mesh/"

# Read the STL file
# mesh = o3d.io.read_triangle_mesh(data_dir+"eric_mesh.stl")
mesh = o3d.io.read_triangle_mesh(data_dir+"mesh_cut.stl")
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

# move the lowset vertex to the origin
vertices = np.asarray(mesh.vertices)
lowest_vertex = np.argmin(vertices[:,2])
translation = -vertices[lowest_vertex]
mesh.translate(translation)

# rotate the mesh around the x axis
mesh.rotate(mesh.get_rotation_matrix_from_xyz([np.radians(30), 0, np.radians(10)]), center=(0, 0, 0))

# # move in z direction
translation = [25, 0, -25]
mesh.translate(translation)

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

for triangle in triangles:
    z_values = vertices[triangle, 2]
    if (z_values <= 0).any() and (z_values >= 0).any():
        # find the vertices of the triangle <= 0
        z_values_0 = z_values <= 0
        triangle_index = triangle[z_values_0]
        vertices[triangle_index, 2] = 0
mesh.vertices = o3d.utility.Vector3dVector(vertices)

# Remove triangles where all 3 vertices have z < 0
triangles_to_remove = []
for i, triangle in enumerate(triangles):
    z_values = vertices[triangle, 2]
    if (z_values < 0).all():
        triangles_to_remove.append(i)

# Create a new mesh without the unwanted triangles
mesh.remove_triangles_by_index(triangles_to_remove)

# Add the plane to the visualization
visualize_meshes([mesh])
# save mesh
o3d.io.write_triangle_mesh(data_dir+"mesh_transformed.stl", mesh)