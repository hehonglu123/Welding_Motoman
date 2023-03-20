from stl import mesh
import numpy as np
import matplotlib.pyplot as plt

# Load the STL file
filename = '../data/blade0.1/blade_modified.stl'
your_mesh = mesh.Mesh.from_file(filename)

volume, cog, inertia = your_mesh.get_mass_properties()
print("Volume                                  = {0}".format(volume))
print("Position of the center of gravity (COG) = {0}".format(cog))
print("Inertia matrix at expressed at the COG  = {0}".format(inertia[0,:]))
print("                                          {0}".format(inertia[1,:]))
print("                                          {0}".format(inertia[2,:]))

# # Access the vertices of the STL file
# for i, facet in enumerate(your_mesh):
#     for j, vertex in enumerate(facet):
#         print(f'Facet {i + 1}, Vertex {j + 1}: {vertex}')

# # If you want to access the normal vector of each facet:
# for i, normal in enumerate(your_mesh.normals):
#     print(f'Facet {i + 1}, Normal: {normal}')


# Get the number of facets in the STL file
num_facets = len(your_mesh)

# Extract all vertices
vertices = np.zeros((num_facets, 3, 3))
for i, facet in enumerate(your_mesh.vectors):
    vertices[i] = facet
print(vertices[0])
# Flatten the vertices array and remove duplicates
unique_vertices = np.unique(vertices.reshape(-1, 3), axis=0)


ax = plt.figure().add_subplot(projection='3d')
ax.scatter(unique_vertices[:,0],unique_vertices[:,1],unique_vertices[:,2])

plt.show()