import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


wall_width=10
layer_height=1
num_layers=50

points_distance=0.1

points_per_layer=int(wall_width/points_distance)
curve_dense=np.zeros((num_layers*points_per_layer,6))

for layer in range(num_layers):
	if layer % 2 ==0:
		curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,0]=np.linspace(0,wall_width,points_per_layer)
	else:
		curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,0]=np.linspace(wall_width,0,points_per_layer)

	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,2]=layer*layer_height

curve_dense[:,-1]=np.ones(len(curve_dense))


vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],'r.-')
ax.quiver(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],curve_dense[::vis_step,3],curve_dense[::vis_step,4],curve_dense[::vis_step,5],length=1, normalize=True)
plt.show()

for layer in range(num_layers):
	np.savetxt('curve_sliced/'+str(layer)+'.csv',curve_dense[layer*points_per_layer:(layer+1)*points_per_layer],delimiter=',')

