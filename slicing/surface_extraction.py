import numpy as np
from stl import mesh

def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1))

def extract_curved_surface(stl_file, output_file, reference_normal, angle_tolerance_degrees):
    # Read the STL file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # Calculate the normals of each face
    face_normals = stl_mesh.normals

    # Find faces with normals within the angle tolerance from the reference normal
    target_faces = []
    angle_tolerance_radians = np.radians(angle_tolerance_degrees)
    for i, normal in enumerate(face_normals):
        angle = angle_between_vectors(normal, reference_normal)
        if angle <= angle_tolerance_radians:
            target_faces.append(stl_mesh.vectors[i])

    # Create a new mesh for the target surface
    target_surface = mesh.Mesh(np.zeros(len(target_faces), dtype=mesh.Mesh.dtype))
    target_surface.vectors = target_faces

    # Save the target surface as a new STL file
    target_surface.save(output_file)

# Example usage
stl_file = '../data/blade0.1/blade_modified.stl'
output_file = 'output_file.stl'
reference_normal = np.array([0, 1, -1])  # Reference normal vector
reference_normal = reference_normal/np.linalg.norm(reference_normal)
angle_tolerance_degrees = 15  # Angle tolerance in degrees
extract_curved_surface(stl_file, output_file, reference_normal, angle_tolerance_degrees)
