import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


wall_radius=10
layer_height=1
num_layers=50

radius_distance=np.pi/180

points_per_layer=int(np.pi*2/radius_distance)
curve_dense=np.zeros((num_layers*points_per_layer,6))

for layer in range(num_layers):
	angles=np.linspace(0,np.pi*2,num=points_per_layer,endpoint=False)
	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,0]=np.cos(angles)
	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,1]=np.sin(angles)


	curve_dense[layer*points_per_layer:(layer+1)*points_per_layer,2]=layer*layer_height

curve_dense[:,-1]=np.ones(len(curve_dense))


vis_step=10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],'r.-')
ax.quiver(curve_dense[::vis_step,0],curve_dense[::vis_step,1],curve_dense[::vis_step,2],curve_dense[::vis_step,3],curve_dense[::vis_step,4],curve_dense[::vis_step,5],length=1, normalize=True)
plt.show()

np.savetxt('Curve_dense.csv',curve_dense,delimiter=',')
