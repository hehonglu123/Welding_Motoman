import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


cylinder_radius=30
total_height=100
line_resolution=0.1
points_distance=0.5

num_slices=int(total_height/line_resolution)
 

points_per_slice=int(np.pi*2*cylinder_radius/points_distance)
curve_dense=np.zeros((num_slices*points_per_slice,6))

for slice in range(num_slices):
	angles=np.linspace(0,np.pi*2,num=points_per_slice,endpoint=False)
	curve_dense[slice*points_per_slice:(slice+1)*points_per_slice,0]=np.cos(angles)*cylinder_radius
	curve_dense[slice*points_per_slice:(slice+1)*points_per_slice,1]=np.sin(angles)*cylinder_radius


	curve_dense[slice*points_per_slice:(slice+1)*points_per_slice,2]=slice*line_resolution

curve_dense[:,-1]=-np.ones(len(curve_dense))


vis_step=10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],'r.-')
ax.quiver(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],curve_dense[::vis_step,3],curve_dense[::vis_step,4],curve_dense[::vis_step,5],length=1, normalize=True)
plt.show()

for slice in range(num_slices):
	np.savetxt('dense_slice/curve_sliced/slice'+str(slice)+'_0.csv',curve_dense[slice*points_per_slice:(slice+1)*points_per_slice],delimiter=',')