import numpy as np
from matplotlib import pyplot as plt

#### data directory
dataset='circle_large/'

sliced_algs = ['static_stepwise_shift','static_stepwise_zero','static_stepwise_shift']
collected_data = ['weld_scan_baseline_2023_09_18_16_17_34','weld_scan_correction_2023_09_19_18_03_53','weld_scan_correction_2023_09_19_16_32_31']
legends=['Baseline','Correction Zero','Correction Shift']

total_datasets=len(collected_data)
for i in range(total_datasets):
    sliced_alg=sliced_algs[i]
    curve_data_dir = '../data/'+dataset+sliced_alg+'/'
    method=collected_data[i]
    data_dir=curve_data_dir+method+'/'
    height_std = np.load(data_dir+'height_std.npy')
    plt.plot(height_std,'-o',label=legends[i])
plt.legend()
plt.xlabel("Layer #")
plt.ylabel("STD (mm)")
plt.title('Height STD')
plt.show()