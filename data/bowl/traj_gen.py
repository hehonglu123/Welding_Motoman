import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import normalize

total_height=100
z_offset=2
line_resolution=1
point_distance=0.5

vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


curve_dense=[]
###BASE LAYER
radius_outter = 10*z_offset**(1/4)
num_circle=int(np.ceil(radius_outter/line_resolution))
radii=np.linspace(line_resolution/2,radius_outter,num_circle)
slice0=[]
section_num=0
for r in radii:
	diameter=2*np.pi*r
	num_points=int(np.floor(diameter/point_distance))
	theta = np.linspace(0, 2 * np.pi, num_points)

	section=np.vstack((r * np.cos(theta),r * np.sin(theta),np.zeros(num_points),np.zeros(num_points),np.zeros(num_points),-np.ones(num_points))).T

	slice0.append(section)
	if section_num % 10==0:
		ax.plot3D(section[::vis_step,0],section[::vis_step,1],section[::vis_step,2],'r.-')
		ax.quiver(section[::vis_step,0],section[::vis_step,1],section[::vis_step,2],section[::vis_step,3],section[::vis_step,4],section[::vis_step,5],length=1, normalize=True)

	section_num+=1
curve_dense.append(slice0)
layer_prev=section

###BOWL LAYERS
z=0
layer=1
while z<total_height:
	drdz=2.5 * (z + z_offset) ** (-0.75)
	dd=np.linalg.norm([1,drdz])
	dz=line_resolution/dd
	z+=dz
	r=10*(z+z_offset)**(1/4)

	diameter=2*np.pi*r
	num_points=int(np.floor(diameter/point_distance))
	theta = np.linspace(0, 2 * np.pi, num_points)

	points=np.vstack((r * np.cos(theta),r * np.sin(theta),z*np.ones(num_points))).T
	normal=-np.vstack((np.cos(theta)*drdz,np.sin(theta)*drdz,np.ones(num_points))).T
	normal=normalize(normal, axis=1)

	layer+=1
	if layer % 20==0:
		ax.plot3D(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],'r.-')
		ax.quiver(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],normal[::vis_step,0],normal[::vis_step,1],normal[::vis_step,2],length=5, normalize=True)

	curve_dense.append([np.hstack((points,normal))])
ax.set_box_aspect([1, 1, 1])  # Adjust the values as needed


for i in range(len(curve_dense)):
	for x in range(len(curve_dense[i])):
		np.savetxt('circular_slice/curve_sliced/slice%i_%i.csv'%(i,x),curve_dense[i][x],delimiter=',')
plt.show()
