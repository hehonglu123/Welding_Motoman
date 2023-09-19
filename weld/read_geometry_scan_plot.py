import numpy as np
from matplotlib import pyplot as plt

#### data directory
dataset='circle_large/'
sliced_alg='static_stepwise_shift/'
curve_data_dir = '../data/'+dataset+sliced_alg

collected_data = ['weld_scan_baseline_2023_09_18_16_17_34','weld_scan_correction_2023_09_18_14_35_10']
legends=['Baseline','Correction']

total_datasets=len(collected_data)
for i in range(total_datasets):
    dataset=collected_data[i]
    data_dir=curve_data_dir+dataset+'/'
    height_std = np.load(data_dir+'height_std.npy')
    plt.plot(height_std,'-o',label=legends[i])
plt.legend()
plt.title('Height STD')
plt.show()