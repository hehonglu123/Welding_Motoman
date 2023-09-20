import numpy as np 
import matplotlib.pyplot as plt
import sys
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

plt.plot(ts_clamp,current_clamp,label='clamp')
plt.plot(ts_fronius,current_fronius,label='fronius')
plt.plot(ts_clamp_moving_avg,current_clamp_moving_avg,label='clamp moving avg')
plt.plot(ts_clamp_avg[1:],current_clamp_avg,label='clamp avg')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')

plt.show()