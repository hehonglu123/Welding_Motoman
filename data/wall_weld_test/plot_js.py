import numpy as np
from matplotlib import pyplot as plt
import glob

data_dir = 'curve_sliced_js/'

curve_js_files = glob.glob(data_dir + 'MA2010_base_js*_0.csv')
for i in range(len(curve_js_files)):
    curve_js = np.loadtxt(data_dir+f'MA2010_base_js{i}_0.csv', delimiter=',')
    curve_js_scan = np.loadtxt(data_dir+f'MA2010_base_js{i}_scanOnly.csv', delimiter=',')

    print(len(curve_js), len(curve_js_scan))
    min_index = np.argmin(np.linalg.norm(curve_js - curve_js_scan[0], axis=1))
    if min_index == 0:
        curve_js = np.vstack((curve_js_scan[::-1],curve_js))
    elif min_index == len(curve_js)-1:
        curve_js = np.vstack((curve_js,curve_js_scan))
    else:
        assert False, 'No match found'
    
    plt.plot(curve_js)
    plt.show()