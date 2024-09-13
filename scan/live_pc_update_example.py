import open3d as o3d
import numpy as np
import time

dataset='blade0.1/'
sliced_alg='dense_slice/'
data_dir='../../geometry_data/'+dataset+sliced_alg
slice_increment=10
first_layer=np.loadtxt(data_dir+'curve_sliced_relative/slice0_0.csv',delimiter=',')[:,:3]


# Create a visualizer window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Create an initial point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(first_layer)


vis.add_geometry(pcd)


counts=1
# Main loop
while True:
    # Generate new random points (replace with your real-time data update logic)
    new_points=np.loadtxt(data_dir+'curve_sliced_relative/slice'+str(counts*slice_increment)+'_0.csv',delimiter=',')[:,:3]
    
    # Append new points to the existing points
    all_points = np.vstack((np.asarray(pcd.points), new_points)) if counts > 0 else new_points
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Update the visualizer
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.1)  # Sleep for a short duration to simulate real-time update
    counts+=1