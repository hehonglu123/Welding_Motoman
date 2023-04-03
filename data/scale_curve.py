import numpy as np
import os

data_dir = 'blade0.1/NX_slice2/curve_sliced_relative/'
output_dir = 'blade0.8/NX_slice2/curve_sliced_relative/'

all_files = os.listdir(data_dir)

scale_ratio = 8

for filename in all_files:
    print(filename)
    if filename[-4:] == '.csv':
        curve = np.loadtxt(data_dir+filename,delimiter=',')
        curve[:,:3] = curve[:,:3]*scale_ratio
        np.savetxt(output_dir+filename,curve,delimiter=',')