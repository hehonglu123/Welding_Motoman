import numpy as np
import matplotlib.pyplot as plt
import sys, glob

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
count=0
for file in glob.glob('curve_sliced/*.csv'):
	curve=np.loadtxt(file,delimiter=',')
	if count>0:
		curve=np.vstack((curve[2:],curve[:2]))
		# curve=curve[2:]
	ax.plot3D(curve[:,0],curve[:,1],curve[:,2])


	count+=1
	# if count>100:
	# 	break
ax.axes.set_xlim3d(left=44, right=555) 
ax.axes.set_ylim3d(bottom=44, top=555) 
ax.axes.set_zlim3d(bottom=44, top=555) 
plt.show()


# curve=np.loadtxt('curve_sliced/_z_0.9.csv',delimiter=',')
# curve=np.vstack((curve[2:],curve[:2]))
# vis_step=10
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot3D(curve[:,0],curve[:,1],curve[:,2],'r.-')
# plt.show()