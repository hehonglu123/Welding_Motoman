from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, copy
sys.path.append('../toolbox')
from path_calc import *
from error_check import *

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

def calculate_surface_normal(p1,p2,p3):
    vector1 = p2 - p1
    vector2 = p3 - p1
    surface_normal = np.cross(vector1, vector2)
    surface_normal /= np.linalg.norm(surface_normal)  # Normalize the normal vector
    return surface_normal

def fit_plane(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    # Calculate the SVD of the centered points
    u, s, vh = np.linalg.svd(centered_points)

    # The normal vector of the plane is the last column of vh
    normal = vh[-1]

    return normal, centroid

def project_point_onto_plane(point, normal, centroid):
    # Compute the vector from the centroid to the point
    v = point - centroid

    # Compute the dot product of v and the normal vector
    dot_product = np.dot(v, normal)

    # Compute the projection of the point onto the plane
    projected_point = point - dot_product * normal

    return projected_point


def slicing_uniform(stl_pc,z,threshold = 1e-6):

    

    bottom_edge_vertices = stl_pc[np.where(np.abs(stl_pc[:,2] - z) <threshold)[0]]

    return bottom_edge_vertices

def get_curve_normal(curve,stl_pc,direction):
    ###provide the curve and complete stl point cloud, a rough normal direction
    curve_normal=[] 
    for i in range(len(curve)):
        tangent=curve[min(len(curve)-1,i+1)]-curve[max(0,i-1)]
        indices=np.argsort(np.linalg.norm(stl_pc-curve[i],axis=1))[:10]
        surf_norm, _=fit_plane(stl_pc[indices])
        true_norm=np.cross(surf_norm,tangent)
        if true_norm@direction<0:
            true_norm=-true_norm
        curve_normal.append(true_norm/np.linalg.norm(true_norm))
    
    return np.array(curve_normal)

def slice_next_layer(curve,stl_pc,direction,slice_height):

    curve_normal=get_curve_normal(curve,stl_pc,direction)
    slice_next=[]
    for i in range(len(curve)):
        p_plus=curve[i]+slice_height*curve_normal[i]

        indices=np.argsort(np.linalg.norm(stl_pc-p_plus,axis=1))[:10]
        normal,centroid=fit_plane(stl_pc[indices])
        p_plus=project_point_onto_plane(p_plus,normal,centroid)

        slice_next.append(p_plus)

    return np.array(slice_next)


def slice(bottom_curve,stl_pc,direction,slice_height):
    direction=np.array([0,0,-1])
    bottom_curve_normal=get_curve_normal(bottom_curve,stl_pc,direction)
    slice_all=[bottom_curve]
    for i in range(50):
        slice_all.append(slice_next_layer(slice_all[-1],stl_pc,direction,slice_height) )

    return slice_all

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
stl_pc = np.unique(vertices.reshape(-1, 3), axis=0)
stl_pc *= 25.4      ##convert to mm

bottom_edge = slicing_uniform(stl_pc,z = np.max(stl_pc[:,2]))
curve_normal=get_curve_normal(bottom_edge,stl_pc,np.array([0,0,-1]))

slice_all=slice(bottom_edge,stl_pc,np.array([0,0,-1]),slice_height=0.8)

# Plot the original points and the fitted curved plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=5

# ax.plot3D(bottom_edge[::vis_step,0],bottom_edge[::vis_step,1],bottom_edge[::vis_step,2],'r.-')
# ax.quiver(bottom_edge[::vis_step,0],bottom_edge[::vis_step,1],bottom_edge[::vis_step,2],curve_normal[::vis_step,0],curve_normal[::vis_step,1],curve_normal[::vis_step,2],length=0.1, normalize=True)
# ax.scatter(stl_pc[:,0], stl_pc[:,1], stl_pc[:,2], c='b', marker='o', label='Original points')

for i in range(len(slice_all)):
    ax.plot3D(slice_all[i][::vis_step,0],slice_all[i][::vis_step,1],slice_all[i][::vis_step,2],'r.-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('STL first X Layer Slicing')
plt.show()