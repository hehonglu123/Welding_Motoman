import numpy as np
import matplotlib.pyplot as plt

x_start=1630
x_end=1670
x_all=np.linspace(x_start,x_end,3)
recorded_dir='recorded_data/'
for x in x_all:
    #height profile
    profile_height=np.load(recorded_dir+'scans/x_%.1f'%x+'/height_profile.npy')
    plt.plot(profile_height[:,0],profile_height[:,1],label='x=%.1f'%x)

plt.title('single layer repeatability')
plt.xlabel('lambda (mm)')
plt.ylabel('height (mm)')
plt.legend()
plt.show()