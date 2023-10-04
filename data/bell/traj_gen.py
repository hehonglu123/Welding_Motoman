import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import normalize

coeff1=100
coeff2=40
final_radius=40
total_height=60
def bell(z, A=final_radius, sigma=30):
    # Gaussian function defined for x >= 0
    if z >= 0:
        return A * np.exp(-(coeff1-z-coeff2)**2 / (2*sigma**2))
    else:
        return None
def dr_dz(z, A=final_radius, sigma=30):
    exponential_term = np.exp(-((coeff1 - z - coeff2) ** 2) / (2 * sigma ** 2))
    derivative_term = (coeff1 - z - coeff2) / (sigma ** 2)
    
    return A * exponential_term * derivative_term

line_resolution=0.1
point_distance=0.5

vis_step=2
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


curve_dense=[]
###BASE LAYER
radius_outter = bell(0)
num_circle=int(radius_outter/line_resolution)
radii=np.linspace(line_resolution,radius_outter,num_circle)
slice0=[]
section_num=0
for r in radii:
	diameter=2*np.pi*r
	num_points=int(np.floor(diameter/point_distance))
	if num_points<4:
		continue
	theta = np.linspace(0, 2 * np.pi, num_points)

	section=np.vstack((r * np.cos(theta),r * np.sin(theta),np.zeros(num_points),np.zeros(num_points),np.zeros(num_points),-np.ones(num_points))).T

	slice0.append(section)
	if section_num % 1==0:
		ax.plot3D(section[::vis_step,0],section[::vis_step,1],section[::vis_step,2],'r.-')
		ax.quiver(section[::vis_step,0],section[::vis_step,1],section[::vis_step,2],section[::vis_step,3],section[::vis_step,4],section[::vis_step,5],length=1, normalize=True)

	section_num+=1
curve_dense.append(slice0)
layer_prev=section

###CUP LAYERS
z=0
layer=1
magnitude = 0.3*2	###offset start/end position along the circle
frequency = 0.1		###every 10 layers comes back to same start/end point
osc_center=-0.3

while z<total_height:
	drdz=dr_dz(z)
	dd=np.linalg.norm([1,drdz])
	dz=line_resolution/dd
	z+=dz
	r=bell(z)	###RADII DEFINITION

	diameter=2*np.pi*r
	num_points=int(np.floor(diameter/point_distance))

	offset=0 # osc_center+magnitude * (2 * np.abs(frequency * layer % 1 - 0.5) - 0.5)

	theta = np.linspace(offset, offset + 2 * np.pi, num_points)

	points=np.vstack((r * np.cos(theta),r * np.sin(theta),z*np.ones(num_points))).T
	normal=-np.vstack((np.cos(theta)*drdz,np.sin(theta)*drdz,np.ones(num_points))).T
	normal=normalize(normal, axis=1)

	layer+=1
	if layer % 1==0:
		ax.plot3D(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],'r.-')
		# ax.quiver(points[::vis_step,0],points[::vis_step,1],points[::vis_step,2],normal[::vis_step,0],normal[::vis_step,1],normal[::vis_step,2],length=5, normalize=True)

	curve_dense.append([np.hstack((points,normal))])


for i in range(len(curve_dense)):
	for x in range(len(curve_dense[i])):
		np.savetxt('circular_slice/curve_sliced/slice%i_%i.csv'%(i,x),curve_dense[i][x],delimiter=',')
ax.set_box_aspect([1, 1, 1])  # Adjust the values as needed

plt.show()

