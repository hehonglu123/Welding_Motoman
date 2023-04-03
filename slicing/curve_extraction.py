import numpy as np
from stl import mesh
import itertools

def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1))

def find_adjacent_faces(mesh):
    adjacent_faces = {}
    for i, face in enumerate(mesh.vectors):
        for edge in (face[[0, 1]], face[[1, 2]], face[[2, 0]]):
            edge_tuple = tuple(sorted(edge.tobytes() for edge in edge))
            adjacent_faces.setdefault(edge_tuple, []).append(i)
    return adjacent_faces

def extract_curve(stl_file, angle_tolerance_degrees):
    # Read the STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Calculate the normals of each face
    face_normals = stl_mesh.normals

    # Find adjacent faces for each edge
    adjacent_faces = find_adjacent_faces(stl_mesh)

    # Collect edges that form the curve based on the angle tolerance
    curve_edges = []
    angle_tolerance_radians = np.radians(angle_tolerance_degrees)
    for edge, face_indices in adjacent_faces.items():
        if len(face_indices) == 2:
            angle = angle_between_vectors(face_normals[face_indices[0]], face_normals[face_indices[1]])
            if angle <= angle_tolerance_radians:
                edge_vertices = np.array([np.frombuffer(v, dtype=np.float32) for v in edge])
                curve_edges.append(edge_vertices)

    return curve_edges

# Example usage
stl_file = '../data/blade0.1/surface.stl'
angle_tolerance_degrees = 15  # Angle tolerance in degrees
curve_edges = extract_curve(stl_file, angle_tolerance_degrees)

# Print the curve edges
for edge in curve_edges:
    print(edge)
