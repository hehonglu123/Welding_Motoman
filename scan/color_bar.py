import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import open3d as o3d
from matplotlib import cm

def Rx(theta):
	return np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
def Ry(theta):
	return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
def Rz(theta):
	return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

'''
create the colorbar as desired in matplotlib
'''
rng = [-2, 120]
z_rng = np.arange(rng[1], rng[0], (rng[0]-rng[1])/100)
ax = plt.subplot()

# Use the 'inferno' colormap
im = ax.imshow(np.vstack((z_rng, z_rng, z_rng, z_rng)).T, extent=(0, (rng[1]-rng[0])/20, rng[0], rng[1]), cmap='inferno')

plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.ylabel('z [m]')

plt.show()
# '''
# write it to an IO buffer in memory
# '''
# buf = io.BytesIO()
# plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, transparent=True)

# '''
# read the buffer as a cv2 image (maybe also works directly into open3d?)
# '''
# buf.seek(0)
# array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
# im = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)

# '''
# convert image to a open3d material (texture) and map it on a thin box
# '''
# im_o3d = o3d.geometry.Image(cv2.cvtColor(im, cv2.COLOR_RGBA2BGRA))
# box = o3d.geometry.TriangleMesh.create_box(im.shape[1]/10,im.shape[0]/1, 0.1, create_uv_map=True, map_texture_to_each_face=True)
# box.rotate(Rx(-np.pi/2),center=(0,0,0))
# box.compute_triangle_normals()

# mat = o3d.visualization.rendering.MaterialRecord()
# mat.shader = 'defaultLitTransparency' # 'defaultLit' would show background of box white
# # mat.base_color = [1,1,1, 0.75]
# mat.albedo_img = im_o3d

# pcd = o3d.geometry.PointCloud()
# num_points = 1000
# points = np.random.rand(num_points, 3) * 5
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.paint_uniform_color([0, 0, 1])  # Color the point cloud blue for contrast

# '''
# create open3d scene to show the result
# '''
# o3d.visualization.gui.Application.instance.initialize()
# w = o3d.visualization.O3DVisualizer('title', 1024, 768)
# w.add_geometry('colorbar', box, mat)
# w.add_geometry('pcd', pcd)

# w.reset_camera_to_default()
# o3d.visualization.gui.Application.instance.add_window(w)
# o3d.visualization.gui.Application.instance.run()