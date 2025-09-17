import numpy as np
from matplotlib import pyplot as plt
import glob

data_dir = 'curve_sliced_js/'

direction = 'backward' # 'forward' or 'backward'

# curve_js_files = glob.glob(data_dir + f'MA2010_base_js*_0_{direction}.csv')
# for i in range(len(curve_js_files)):

weld_parts = 'layer' # base or layer
layer_num = 459

if weld_parts == 'base':
    curve_js = np.loadtxt(data_dir+f'MA2010_base_js{layer_num}_0_{direction}.csv', delimiter=',')
    curve_js_scan = np.loadtxt(data_dir+f'MA2010_base_js{layer_num}_0_scan_{direction}.csv', delimiter=',')
    curve_js_positioner = np.loadtxt(data_dir+f'D500B_base_js{layer_num}_0_{direction}.csv', delimiter=',')
    curve_js_scan_positioner = np.loadtxt(data_dir+f'D500B_base_js{layer_num}_0_scan_{direction}.csv', delimiter=',')
else:
    curve_js = np.loadtxt(data_dir+f'MA2010_js{layer_num}_0_{direction}.csv', delimiter=',')
    curve_js_scan = np.loadtxt(data_dir+f'MA2010_js{layer_num}_0_scan_{direction}.csv', delimiter=',')
    curve_js_positioner = np.loadtxt(data_dir+f'D500B_js{layer_num}_0_{direction}.csv', delimiter=',')
    curve_js_scan_positioner = np.loadtxt(data_dir+f'D500B_js{layer_num}_0_scan_{direction}.csv', delimiter=',')

print(len(curve_js), len(curve_js_scan))
min_index = np.argmin(np.linalg.norm(curve_js - curve_js_scan[0], axis=1))
if min_index == 0:
    curve_js = np.vstack((curve_js_scan[::-1],curve_js))
    curve_js_positioner = np.vstack((curve_js_scan_positioner[::-1],curve_js_positioner))
elif min_index == len(curve_js)-1:
    curve_js = np.vstack((curve_js,curve_js_scan))
    curve_js_positioner = np.vstack((curve_js_positioner,curve_js_scan_positioner))
else:
    assert False, 'No match found'

plt.plot(np.degrees(curve_js), '-o')
plt.plot(np.degrees(curve_js_positioner), '-o')

plt.legend(['j1','j2','j3','j4','j5','j6','ext_j1','ext_j2'])
plt.show()