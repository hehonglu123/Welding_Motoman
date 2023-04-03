import numpy as np
import vtk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_stl_as_polydata(stl_file):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    return reader.GetOutput()

def vtk_to_numpy(vtk_array):
    return np.array(vtk_array)


def plot_slices_points_3d(slices_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, points in enumerate(slices_points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        ax.scatter(x, y, z, label=f'Slice {i}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sliced Points')
    ax.legend()

    plt.show()

def slice_stl_to_numpy_arrays_num(input_stl, num_slices):
    # Read the input STL file as polydata
    input_polydata = read_stl_as_polydata(input_stl)

    # Get bounds of the input STL file
    bounds = input_polydata.GetBounds()
    z_min, z_max = bounds[4], bounds[5]

    # Calculate the slicing height based on the number of slices
    slicing_height = (z_max - z_min) / (num_slices - 1)

    # Extract points of each slice as NumPy arrays
    slices_points = []
    for i in range(num_slices):
        z_position = z_min + i * slicing_height

        # Create a plane to slice the STL data
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, z_position)
        plane.SetNormal(0, 0, 1)

        # Create a cutter to slice the input STL data along the plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(input_polydata)
        cutter.Update()

        # Extract the slice as an individual polydata object
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputConnection(cutter.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()

        slice_polydata = connectivity_filter.GetOutput()
        
        # Check if the slice contains points
        if slice_polydata.GetNumberOfPoints() > 0:
            points_vtk = slice_polydata.GetPoints().GetData()
            points_numpy = vtk_to_numpy(points_vtk)
            slices_points.append(points_numpy)

    return slices_points


def slice_stl_to_numpy_arrays(input_stl, slicing_height):
    # Read the input STL file as polydata
    input_polydata = read_stl_as_polydata(input_stl)

    # Get bounds of the input STL file
    bounds = input_polydata.GetBounds()
    z_min, z_max = bounds[4], bounds[5]

    # Calculate the number of slices based on the slicing height
    num_slices = int((z_max - z_min) / slicing_height) + 1

    # Extract points of each slice as NumPy arrays
    slices_points = []
    for i in range(num_slices):
        z_position = z_min + i * slicing_height

        # Create a plane to slice the STL data
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, z_position)
        plane.SetNormal(0, 0, 1)

        # Create a cutter to slice the input STL data along the plane
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(input_polydata)
        cutter.Update()

        # Extract the slice as an individual polydata object
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputConnection(cutter.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()

        slice_polydata = connectivity_filter.GetOutput()
        
        # Check if the slice contains points
        if slice_polydata.GetNumberOfPoints() > 0:
            points_vtk = slice_polydata.GetPoints().GetData()
            points_numpy = vtk_to_numpy(points_vtk)
            slices_points.append(points_numpy)

    return slices_points

# Example usage
input_stl = '../data/blade0.1/surface.stl'
slicing_height = 0.1

slices_points = slice_stl_to_numpy_arrays_num(input_stl, 20)

plot_slices_points_3d(slices_points)
