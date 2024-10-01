import glob
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

class TreeNode:
    def __init__(self, value):
        self.value = value  # Node value
        self.children = []  # List of child nodes
        self.parent = None  # Reference to parent node

    # Add a child to the node and set its parent
    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child
        self.children.append(child_node)

    # Get the parent of the node
    def get_parent(self):
        return self.parent

    # Display the tree structure
    def display(self, level=0):
        indent = " " * level * 4  # Indent each level of the tree
        print(f"{indent}{self.value}")  # Print the current node value
        for child in self.children:  # Recursively display children
            child.display(level + 1)

data_dir = '../data/blade0.1/auto_slice/curve_sliced_relative/'

baselayers = glob.glob(data_dir+'base*.csv')
slicelayers = glob.glob(data_dir+'slice*_0.csv')

print('Number of base layers:',len(baselayers))
print('Number of slice layers:',len(slicelayers))

# Plot the original points and the fitted curved plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=1

height = 1

cmap = plt.get_cmap("jet")

end_plot = len(slicelayers)
end_plot = 10

nodes = {}
dh_status = []
for i in range(end_plot):
    print('Processing slice %d'%i)
    if i%height!=0:
        continue
    slice_region = glob.glob(data_dir+'slice%d_*.csv'%i)
    print(slice_region)
    for x in range(len(slice_region)):
        this_slice = np.loadtxt(data_dir+'slice%d_%d.csv'%(i,x),delimiter=',')
        
        if i==0:
            this_slice_color = cmap(np.linspace(0,1,len(this_slice))) # color by index
            for p,p_color in zip(this_slice,this_slice_color):
                ax.plot3D(p[0],p[1],p[2],'.',color=tuple(p_color))
                # add to nodes
                nodes[tuple(p)] = TreeNode(p)
        else:
            this_slice_color = []
            for p in this_slice:
                last_slice_dense_closest = last_slice_dense[np.argmin(np.linalg.norm(last_slice_dense-p,axis=1))]
                this_color = last_slice_color[np.argmin(np.linalg.norm(last_slice-last_slice_dense_closest,axis=1))]
                dh_status.append(np.linalg.norm(last_slice_dense_closest-p))
                this_slice_color.append(this_color)
                # this_color = cmap(np.min(np.linalg.norm(last_slice_dense-p,axis=1)))
                ax.plot3D(p[0],p[1],p[2],'.',color=tuple(this_color))
                # add to nodes
                last_slice_closest = last_slice[np.argmin(np.linalg.norm(last_slice-last_slice_dense_closest,axis=1))]
                nodes[tuple(p)] = TreeNode(p)
                nodes[tuple(last_slice_closest)].add_child(nodes[tuple(p)])
            
        # this_slice_color = [tuple(row[:3]) for row in this_slice_color]
        # ax.plot3D(this_slice[::vis_step,0],this_slice[::vis_step,1],this_slice[::vis_step,2],'.',color=this_slice_color)

    last_slice = deepcopy(this_slice)
    last_slice_dense = []
    for i in range(1,len(this_slice)-1):
        last_slice_dense.extend(np.linspace(this_slice[i-1],this_slice[i],int(np.linalg.norm(this_slice[i-1]-this_slice[i])/0.1)+1)[:-1])
    last_slice_dense.append(this_slice[-1])
    last_slice_dense = np.array(last_slice_dense)
    last_slice_color = deepcopy(this_slice_color)

print('mean dh:',np.mean(dh_status))
print('Min 5 dh:',np.sort(dh_status)[:5])
print('Max 5 dh:',np.sort(dh_status)[-5:])

# # Function to set equal scaling for all axes
# def set_axes_equal(ax):
#     """Set 3D plot axes to equal scale."""
#     limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
#     span = limits[:, 1] - limits[:, 0]
#     center = np.mean(limits, axis=1)
#     radius = 0.5 * np.max(span)
    
#     ax.set_xlim(center[0] - radius, center[0] + radius)
#     ax.set_ylim(center[1] - radius, center[1] + radius)
#     ax.set_zlim(center[2] - radius, center[2] + radius)

# # Apply the function to set equal scaling
# set_axes_equal(ax)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('STL Slicing')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for key in nodes.keys():
    ax.plot3D(key[0],key[1],key[2],'r.')
    for child in nodes[key].children:
        ax.plot3D([key[0],child.value[0]],[key[1],child.value[1]],[key[2],child.value[2]],'b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Node tree')
plt.show()

plt.plot(dh_status,'.')
plt.show()