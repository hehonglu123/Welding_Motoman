import open3d as o3d
import numpy as np
import time

def animation_mesh(mesh,rotation_axis=np.array([0, 0, 1]),rotation_angle=np.radians(2),steps=180,sleep_time=0.01):

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    # Set rotation parameters
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)

    try:
        for _ in range(steps):  # 180 steps of 2 degrees each for a full 360-degree rotation
            # Rotate the mesh
            mesh.rotate(rotation_matrix, center=(0, 0, 0))

            # Update the geometry in the visualizer
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            # Add a small delay to control the speed of the rotation
            time.sleep(sleep_time)
    finally:
        vis.destroy_window()