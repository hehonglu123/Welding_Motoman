import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import normalize

cylinder_height=40
cylinder_radius=30
z_offset=3
r_circle=150
bowl_height=r_circle
radii_coeff=10
line_resolution=1.3
point_distance=0.5

vis_step=2
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


curve_dense=[]

###BASE CYLINDER
num_circle=int(np.ceil(cylinder_height/line_resolution))
num_points=int(np.floor(np.pi*2*cylinder_radius/point_distance))
num_layers_cylinder=int(cylinder_height/line_resolution)
for layer in range(num_layers_cylinder):
	angles=np.linspace(0,np.pi*2,num=num_points,endpoint=False)

	x=np.cos(angles)*cylinder_radius
	y=np.sin(angles)*cylinder_radius
	z=layer*line_resolution+np.linspace(0,line_resolution,num_points)

	slice_ith_layer=np.vstack((x,y,z,np.zeros((2,num_points)),-np.ones(num_points))).T
	curve_dense.append([slice_ith_layer])
	if layer % 30==0:
		ax.plot3D(slice_ith_layer[::vis_step,0],slice_ith_layer[::vis_step,1],slice_ith_layer[::vis_step,2],'r.-')


###BOWL LAYERS
#r=sqrt(r_circle^2-(z-r_circle-z_offset)^2)
z=cylinder_height
z_offset=-(-np.sqrt(r_circle**2-cylinder_radius**2)+r_circle-z)
layer=1
magnitude = 0.3*2	###offset start/end position along the circle
frequency = line_resolution/10		###every 10 mm comes back to same start/end point
osc_center=-0.3

while z<bowl_height+cylinder_height:
	drdz=-(-r_circle+z-z_offset)/np.sqrt((z-z_offset)*(2*r_circle-z+z_offset))

	dd=np.linalg.norm([1,drdz])
	dz=line_resolution/dd
	z+=dz
	r=np.sqrt(r_circle**2-(z-r_circle-z_offset)**2) ###RADII DEFINITION

	diameter=2*np.pi*r
	num_points=int(np.floor(diameter/point_distance))

	offset=osc_center+magnitude * (2 * np.abs(frequency * layer % 1 - 0.5) - 0.5)

	theta = np.linspace(offset, offset + 2 * np.pi, num_points)

	points=np.vstack((r * np.cos(theta),r * np.sin(theta), z*np.ones(num_points))).T
	normal=-np.vstack((np.cos(theta)*drdz,np.sin(theta)*drdz,np.ones(num_points))).T
	normal=normalize(normal, axis=1)

	layer+=1
	if layer % 20==0:
		ax.plot3D(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],'r.-')
		ax.quiver(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],normal[::vis_step,0],normal[::vis_step,1],normal[::vis_step,2],length=5, normalize=True)

	curve_dense.append([np.hstack((points,normal))])


for i in range(len(curve_dense)):
	for x in range(len(curve_dense[i])):
		np.savetxt('circular_slice_shifted/curve_sliced/slice%i_%i.csv'%(i,x),curve_dense[i][x],delimiter=',')
ax.set_xlim([0,1000])
ax.set_ylim([0,1000])
ax.set_zlim([0,1000])

plt.show()

