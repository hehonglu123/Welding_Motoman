from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_bottom_edge(stl_file):
    # Load the STL mesh
    model_mesh = mesh.Mesh.from_file(stl_file)

    # Find the minimum z-coordinate
    max_z = np.max(model_mesh.vectors[:,:,2])

    # Set a threshold for identifying the bottom edge
    threshold = 1e-6

    # Extract the bottom triangles
    bottom_triangles = model_mesh.vectors[np.abs(model_mesh.vectors[:,:,2] - max_z) < threshold]

    # Extract the bottom edge vertices
    bottom_edge_vertices = np.unique(bottom_triangles.reshape(-1, 3), axis=0)

    return bottom_edge_vertices


# Load the STL file
filename = '../data/blade0.1/surface.stl'
your_mesh = mesh.Mesh.from_file(filename)
# Get the number of facets in the STL file
num_facets = len(your_mesh)

# Extract all vertices
vertices = np.zeros((num_facets, 3, 3))
for i, facet in enumerate(your_mesh.vectors):
    vertices[i] = facet
# Flatten the vertices array and remove duplicates
unique_vertices = np.unique(vertices.reshape(-1, 3), axis=0)

bottom_edge = extract_bottom_edge(filename)

# Plot the original points and the fitted curved plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(bottom_edge[:,0], bottom_edge[:,1], bottom_edge[:,2], c='r', marker='o', label='Original points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('bottom edge extraction')
plt.show()