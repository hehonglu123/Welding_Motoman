from slicing import *

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

slice_136 = np.loadtxt('slicing_result/slice_136_0.csv',delimiter=',')
slice_137 = np.loadtxt('slicing_result/slice_137_0.csv',delimiter=',')
# curve_normal=get_curve_normal(slice_137,stl_pc,np.array([0,0,-1]))
curve_normal=get_curve_normal_from_curves(slice_137,slice_136)

slice_138=slice_next_layer(slice_137,stl_pc,curve_normal,slice_height=0.1)
slice_138=fit_to_length(slice_138,stl_pc)

###split the curve based on projection error
sub_curves_next=split_slices(slice_138,stl_pc)
print(len(sub_curves_next))
# for j in range(len(sub_curves_next)):
#     sub_curves_next[j]=smooth_curve(sub_curves_next[j])

# slice_ith_layer.extend(sub_curves_next)

# Plot the original points and the fitted curved plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=5
ax.plot3D(slice_136[::vis_step,0],slice_136[::vis_step,1],slice_136[::vis_step,2],'r.-')
ax.plot3D(slice_137[::vis_step,0],slice_137[::vis_step,1],slice_137[::vis_step,2],'g.-')
ax.plot3D(slice_138[::vis_step,0],slice_138[::vis_step,1],slice_138[::vis_step,2],'b.-')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('STL first X Layer Slicing')
plt.show()