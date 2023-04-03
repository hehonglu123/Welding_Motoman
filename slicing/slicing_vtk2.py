import numpy as np
import vtk
from scipy.spatial import KDTree

def read_stl_as_polydata(stl_file):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    return reader.GetOutput()

def vtk_to_numpy(vtk_array):
    return np.array(vtk_array)

def get_surface_normals(polydata):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()

def resample_points(points, normals, num_points):
    kdtree = KDTree(points)
    resampled_points = [points[np.random.randint(len(points))]]
    print(resample_points[-1])
    resampled_normals = [normals[resampled_points[-1]]]
    for _ in range(num_points - 1):
        dist, idx = kdtree.query(resampled_points[-1], k=2)
        new_point = points[idx[1]]
        resampled_points.append(new_point)
        resampled_normals.append(normals[idx[1]])
    return np.array(resampled_points), np.array(resampled_normals)

def bisector_plane(point1, normal1, point2, normal2):
    bisector_origin = (point1 + point2) / 2
    bisector_normal = normal1 + normal2
    bisector_normal /= np.linalg.norm(bisector_normal)
    return bisector_origin, bisector_normal

def slice_stl_to_numpy_arrays(input_stl, num_slices):
    input_polydata = read_stl_as_polydata(input_stl)
    surface_normals = get_surface_normals(input_polydata)

    points_vtk = input_polydata.GetPoints().GetData()
    points_numpy = vtk_to_numpy(points_vtk)

    normals_vtk = surface_normals.GetPointData().GetNormals()
    normals_numpy = vtk_to_numpy(normals_vtk)

    resampled_points, resampled_normals = resample_points(points_numpy, normals_numpy, num_slices - 1)

    slices_points = []
    for i in range(num_slices - 1):
        origin, normal = bisector_plane(resampled_points[i], resampled_normals[i], resampled_points[i + 1], resampled_normals[i + 1])

        plane = vtk.vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(input_polydata)
        cutter.Update()

        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputConnection(cutter.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()

        slice_polydata = connectivity_filter.GetOutput()

        if slice_polydata.GetNumberOfPoints() > 0:
            points_vtk = slice_polydata.GetPoints().GetData()
            points_numpy = vtk_to_numpy(points_vtk)
            slices_points.append(points_numpy)

    return slices_points

# Example usage
input_stl = '../data/blade0.1/surface.stl'
num_slices = 10

slices_points = slice_stl_to_numpy_arrays(input_stl, num_slices)

# Print the points of each slice
for i, points in enumerate(slices_points):
    print(f'Slice {i}:')
    print
