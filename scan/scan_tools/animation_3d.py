import open3d as o3d
import numpy as np
import time
from copy import deepcopy

def animation_mesh(mesh,rotation_axis=np.array([0, 0, 1]),rotation_angle=np.radians(2),steps=180,sleep_time=0.01):

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if type(mesh) is list:
        points_list = []
        if type(mesh[0]) is o3d.geometry.TriangleMesh:
            viz_obj = o3d.geometry.TriangleMesh()
            for m in mesh:
                viz_obj += m
                points_list.append(np.asarray(m.vertices))
        elif type(mesh[0]) is o3d.geometry.PointCloud:
            viz_obj = o3d.geometry.PointCloud()
            for m in mesh:
                viz_obj += m
                points_list.append(np.asarray(m.points))
    else:
        viz_obj = deepcopy.copy(mesh)
        if type(mesh) is o3d.geometry.TriangleMesh:
            points_list = [np.asarray(mesh.vertices)]
        elif type(mesh) is o3d.geometry.PointCloud:
            points_list = [np.asarray(mesh.points)]
    
    # move everything to xy center of viz obj
    points = np.array(points_list).reshape(-1,3)
    center = np.mean(points, axis=0)
    center_xy = np.append(center[:2],0)
    viz_obj.translate(-center_xy, relative=False)
    
    vis.add_geometry(viz_obj)

    # Set rotation parameters
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    try:
        for _ in range(steps):  # 180 steps of 2 degrees each for a full 360-degree rotation
            # Rotate the mesh
            viz_obj.rotate(rotation_matrix, center=(0, 0, 0))

            # Update the geometry in the visualizer
            vis.update_geometry(viz_obj)
            vis.poll_events()
            vis.update_renderer()

            # Add a small delay to control the speed of the rotation
            time.sleep(sleep_time)
    finally:
        vis.destroy_window()