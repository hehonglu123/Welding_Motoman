import numpy as np
from matplotlib import pyplot as plt
import glob

data_dir = 'curve_sliced_js/'

curve_js_files = glob.glob(data_dir + 'MA2010_base_js*_0_forward.csv')
for i in range(len(curve_js_files)):
    curve_js = np.loadtxt(data_dir+f'MA2010_base_js{i}_0_forward.csv', delimiter=',')
    curve_js_scan = np.loadtxt(data_dir+f'MA2010_base_js{i}_0_scan_forward.csv', delimiter=',')

    print(len(curve_js), len(curve_js_scan))
    min_index = np.argmin(np.linalg.norm(curve_js - curve_js_scan[0], axis=1))
    if min_index == 0:
        curve_js = np.vstack((curve_js_scan[::-1],curve_js))
    elif min_index == len(curve_js)-1:
        curve_js = np.vstack((curve_js,curve_js_scan))
    else:
        assert False, 'No match found'
    
    plt.plot(curve_js, '-o')
    plt.legend(['j1','j2','j3','j4','j5','j6'])
    plt.show()