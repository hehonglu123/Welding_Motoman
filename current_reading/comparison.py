import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
from scipy.signal import correlate

sys.path.append('../toolbox/')
from utils import *

data_dir='../../recorded_data/Cup_ER70S6_spiral_perfect/layer_257/'
data=np.loadtxt(data_dir+'current.csv',delimiter=',',skiprows=1)
data=data[np.argsort(data[:, 0])]
ts_clamp=data[:,0]
current_clamp=data[:,1]

n_avg=100

current_clamp_moving_avg=moving_average(current_clamp,n=n_avg,padding=False)
ts_clamp_moving_avg=moving_average(ts_clamp,n=n_avg,padding=False)

ts_clamp_avg=ts_clamp[::n_avg]
current_clamp_avg=np.average(current_clamp[:int(len(current_clamp)/n_avg)*n_avg].reshape(-1, n_avg), axis=1)



data=np.loadtxt(data_dir+'welding.csv',delimiter=',',skiprows=1)
data=data[np.argsort(data[:, 0])]
ts_fronius=data[:,0]
ts_fronius=(ts_fronius-ts_fronius[1])/1e6
current_fronius=data[:,2]

# plt.plot(ts_clamp,current_clamp,label='clamp')
plt.plot(ts_fronius,current_fronius+3,label='fronius')
# plt.plot(ts_clamp_moving_avg,current_clamp_moving_avg,label='clamp moving avg')
plt.plot(ts_clamp_avg[1:],current_clamp_avg,label='clamp avg')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

plt.show()



###cross correlation
# Interpolate signals onto a common time base
common_time = np.linspace(max(ts_fronius.min(),ts_clamp_avg[1:].min()), min(ts_fronius.max(),ts_clamp_avg[1:].max()), max(len(ts_fronius),len(current_clamp_avg)),endpoint=False)
interp_signal1 = interp1d(ts_fronius, current_fronius, kind='linear')(common_time)
interp_signal2 = interp1d(ts_clamp_avg[1:], current_clamp_avg, kind='linear')(common_time)
offset=np.average(interp_signal1)-np.average(interp_signal2)
# Compute cross-correlation
cross_corr = correlate(interp_signal2[500:-500], interp_signal1[500:-500]+offset, mode='valid')

# Find the optimal time shift
shift_idx = np.argmax(cross_corr)
time_shift = common_time[shift_idx] - common_time[0]
print(shift_idx)
plt.plot(common_time, interp_signal1, label='fronius')
plt.plot(common_time-time_shift, interp_signal2+offset, label='clamp avg shifted')
plt.plot(ts_clamp_avg[1:],current_clamp_avg+offset,label='clamp avg original')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

plt.show()