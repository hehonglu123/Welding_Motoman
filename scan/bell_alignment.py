
import sys,time, copy, yaml
import open3d as o3d


sys.path.append('../toolbox/')
from utils import *
from pointcloud_toolbox import *
sys.path.append('../slicing/')
from slicing import check_boundary
dataset='bell/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
	slicing_meta = yaml.safe_load(file)

###read target points
target_points_pc=[]
# for i in range(0,51,5):
#     target_points_pc.append(np.loadtxt(data_dir+'curve_sliced/slice0_'+str(i)+'.csv',delimiter=',')[:,:3])
for i in range(1,slicing_meta['num_layers'],5):
    target_points_pc.append(np.loadtxt(data_dir+'curve_sliced/slice'+str(i)+'_0.csv',delimiter=',')[:,:3])
target_points_pc=np.concatenate(target_points_pc,axis=0)
target_points=o3d.geometry.PointCloud()
target_points.points=o3d.utility.Vector3dVector(target_points_pc)


scanned_dir='../../evaluation/Bell_ER316L/'
######## read the scanned stl
scanned_mesh = o3d.io.read_triangle_mesh(scanned_dir+'bell_316L_baseline.stl')
scanned_mesh.compute_vertex_normals()
## sample as pointclouds
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=111000)
## global tranformation
R_guess,p_guess=global_alignment(scanned_points.points,target_points.points)

## sample as sparser pointclouds
scanned_points = scanned_mesh.sample_points_uniformly(number_of_points=100000)
target_points = target_points.paint_uniform_color([0, 0.8, 0.0])
scanned_points = scanned_points.paint_uniform_color([0.8, 0, 0.0])
H = H_from_RT(R_guess,p_guess)

scanned_points=scanned_points.transform(H)

pca1 = PCA()
pca1.fit(np.array(target_points.points))
R1 = pca1.components_.T
pca2 = PCA()
pca2.fit(np.array(scanned_points.points ))
R2 = pca2.components_.T
if np.cross(R1[:,0],R1[:,1])@R1[:,2]<0:
    R1[:,2]=-R1[:,2]   
    print('here1')
if np.cross(R2[:,0],R2[:,1])@R2[:,2]<0:
    R2[:,2]=-R2[:,2]
    print('here2')

print(R1)
print(R2)

frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=np.mean(np.array(target_points.points),axis=0))
frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=np.mean(np.array(scanned_points.points),axis=0))

# Apply the rotation matrix to the coordinate frame
frame1.rotate(R1, center=np.mean(np.array(target_points.points),axis=0))

frame2.rotate(R2, center=np.mean(np.array(scanned_points.points),axis=0))


# Step 2: Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Step 3: Add the mesh to the visualization window
vis.add_geometry(target_points)
vis.add_geometry(scanned_points)

vis.add_geometry(frame1)
vis.add_geometry(frame2)
# Step 4: Run the visualization loop (this will block and display the visualization window until it is closed by the user)
vis.run()

# Step 5: Destroy the visualization window to clean up resources
vis.destroy_window()
