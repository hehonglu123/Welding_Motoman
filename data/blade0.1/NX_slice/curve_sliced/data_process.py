import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def find_norm(p1,p2,p3):
	#find normal vector from p1 pointing to line of p2p3
	p2p1=p2-p1
	p3p1=p3-p1
	p2p3=p2-p3
	vec=np.cross(np.cross(p2p1,p3p1),p2p3)
	return vec/np.linalg.norm(vec)
	
slice1=np.loadtxt('raw/slice1.csv',delimiter=',')
slice2=np.loadtxt('raw/slice2.csv',delimiter=',')
slice3=np.loadtxt('raw/slice3.csv',delimiter=',')
slice4=np.loadtxt('raw/slice4.csv',delimiter=',')

slice1_normal=[]
for i in range(len(slice1)):
	#find closest 2 points
	idx_1,idx_2=np.argsort(np.linalg.norm(slice2-slice1[i],axis=1))[:2]
	slice1_normal.append(find_norm(slice1[i],slice2[idx_1],slice2[idx_2]))
slice1_normal=-np.array(slice1_normal)
slice2_normal=[]	
for i in range(len(slice2)):
	#find closest 2 points
	idx_1,idx_2=np.argsort(np.linalg.norm(slice3-slice2[i],axis=1))[:2]
	slice2_normal.append(find_norm(slice2[i],slice3[idx_1],slice3[idx_2]))
slice2_normal=-np.array(slice2_normal)

slice3_normal=[]
for i in range(len(slice3)):
	#find closest 2 points
	idx_1,idx_2=np.argsort(np.linalg.norm(slice4-slice3[i],axis=1))[:2]
	slice3_normal.append(find_norm(slice3[i],slice4[idx_1],slice4[idx_2]))
slice3_normal=-np.array(slice3_normal)


vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(slice1[::vis_step,0],slice1[::vis_step,1],slice1[::vis_step,2],'r.-')
ax.quiver(slice1[::vis_step,0],slice1[::vis_step,1],slice1[::vis_step,2],slice1_normal[::vis_step,0],slice1_normal[::vis_step,1],slice1_normal[::vis_step,2],length=0.3, normalize=True)
ax.plot3D(slice2[::vis_step,0],slice2[::vis_step,1],slice2[::vis_step,2],'g.-')
ax.quiver(slice2[::vis_step,0],slice2[::vis_step,1],slice2[::vis_step,2],slice2_normal[::vis_step,0],slice2_normal[::vis_step,1],slice2_normal[::vis_step,2],length=0.3, normalize=True)
ax.plot3D(slice3[::vis_step,0],slice3[::vis_step,1],slice3[::vis_step,2],'b.-')
ax.quiver(slice3[::vis_step,0],slice3[::vis_step,1],slice3[::vis_step,2],slice3_normal[::vis_step,0],slice3_normal[::vis_step,1],slice3_normal[::vis_step,2],length=0.3, normalize=True)
plt.title('0.1 blade first 3 layers')
plt.show()

np.savetxt('slice0.csv',np.hstack((slice1,slice1_normal)),delimiter=',')
np.savetxt('slice1.csv',np.hstack((slice2,slice2_normal)),delimiter=',')
np.savetxt('slice2.csv',np.hstack((slice3,slice3_normal)),delimiter=',')