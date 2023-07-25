import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

total_height=100
wall_width=100
line_resolution=0.1
points_distance=0.5
num_layers=int(total_height/line_resolution)

points_per_layer=int(wall_width/points_distance)
curve_dense=np.zeros((num_layers*points_per_layer,6))

for layer in range(num_layers):
	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,0]=np.linspace(0,wall_width,points_per_layer)
	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,2]=layer*line_resolution

curve_dense[:,-1]=-np.ones(len(curve_dense))


vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],'r.-')
ax.quiver(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],curve_dense[::vis_step,3],curve_dense[::vis_step,4],curve_dense[::vis_step,5],length=1, normalize=True)
plt.show()

for layer in range(num_layers):
	np.savetxt('dense_slice/curve_sliced/slice'+str(layer)+'_0.csv',curve_dense[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')

