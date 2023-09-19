import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

data=np.loadtxt('current.csv',delimiter=',',skiprows=1)
data=data[np.argsort(data[:, 0])]
ts=data[:,0]
current=data[:,1]

###count peaks in 1s
idx=np.where(ts>1)[0][0]
current_1s=current[:idx]
peaks, _ = find_peaks(current_1s, height=80)
print('Peaks in 1s: ',len(peaks))


n_avg=10
ts_avg=ts[::n_avg]
current_avg=np.average(current[:int(len(current)/n_avg)*n_avg].reshape(-1, n_avg), axis=1)
plt.plot(ts,current)
plt.plot(ts_avg[:-1],current_avg)
plt.scatter(ts[peaks],current[peaks],c='r')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.title('Current vs Time, Steel 100ipm')
plt.grid()
plt.legend(['Raw','Averaged'])

plt.show()