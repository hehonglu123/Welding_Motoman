import numpy as np
import matplotlib.pyplot as plt


curve=np.loadtxt('thin_blade_new.csv',delimiter=',')
vis_step=10
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot3D(curve[:5000,0],curve[:5000,1],curve[:5000,2],'r.-')
plt.show()