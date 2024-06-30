import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

wall_length_base=100
total_height=50*np.sqrt(3)
line_resolution=0.1
points_distance=0.5
num_layers=int(total_height/line_resolution)

curve_dense=[]
vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

for layer in range(num_layers):
	wall_length=(1-layer/num_layers)*wall_length_base
	points_per_layer=int(wall_length/points_distance)
	layer_x=np.linspace((wall_length_base-wall_length)/2,(wall_length_base-wall_length)/2+wall_length,points_per_layer)
	curve_dense.append(np.array([layer_x,np.zeros(len(layer_x)),np.ones(len(layer_x))*layer*line_resolution,np.zeros(len(layer_x)),np.zeros(len(layer_x)),-np.ones(len(layer_x))]).T)
	if len(curve_dense[layer])>1:
		np.savetxt('dense_slice/curve_sliced/slice'+str(layer)+'_0.csv',curve_dense[layer],delimiter=',')

	if layer % vis_step==0:
		ax.plot3D(curve_dense[layer][::vis_step,0],curve_dense[layer][::vis_step,1],curve_dense[layer][::vis_step,2])
		ax.quiver(curve_dense[layer][::vis_step,0],curve_dense[layer][::vis_step,1],curve_dense[layer][::vis_step,2],curve_dense[layer][::vis_step,3],curve_dense[layer][::vis_step,4],curve_dense[layer][::vis_step,5],length=0.5)
	

plt.show()
