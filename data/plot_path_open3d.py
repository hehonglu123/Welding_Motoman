import numpy as np
import yaml
from matplotlib import pyplot as plt
import open3d as o3d

def visualize_objects(mesh,coordinate_size=100):
    # Create a coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coordinate_size, origin=[0, 0, 0])

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

data_dir = 'face_mesh_tanja_straight/'
algo_dir = 'slicing_result_10/'

mesh = o3d.io.read_triangle_mesh(data_dir+'mesh_final.stl')

with open(data_dir+algo_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

vis_step=1

pcd_list = []
cmap = plt.get_cmap('tab10')
for i in range(0,slicing_meta['num_layers'],vis_step):
    curve_sliced_relative = np.loadtxt(data_dir+algo_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(curve_sliced_relative[:,:3])
    color = list(cmap(i/slicing_meta['num_layers'])[:3])
    pcd.paint_uniform_color(color)
    pcd_list.append(pcd)
# pcd_list.append(mesh)
visualize_objects(pcd_list)