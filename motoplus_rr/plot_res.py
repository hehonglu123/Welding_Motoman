import matplotlib.pyplot as plt
import numpy as np

dat = np.loadtxt('joint_recording_streaming.csv', delimiter=',')
print(dat[:,0])

plt.plot(dat[:,2])
plt.show()
