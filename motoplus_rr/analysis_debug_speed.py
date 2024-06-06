import numpy as np 
import matplotlib.pyplot as plt

recorded_data=np.loadtxt('debug_speed_q_all.csv', delimiter=',')
ts,q_all=recorded_data[:,0],recorded_data[:,1:]
#find time difference
dt=np.diff(ts)
plt.scatter(ts[1:],dt)
plt.xlabel('timestamp (s)')
plt.ylabel('time difference (s)')
plt.title('Time Difference')
plt.show()