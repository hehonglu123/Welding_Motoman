import numpy as np
import yaml

# read a curve and make them to a desired point distance
path_dl = 0.025 # desired point distance

data_dir = 'casing_scaled/'
source_dir = 'curve_sliced_relative_origin/'
target_dir = 'curve_sliced_relative/'

with open(data_dir+'sliced_meta.yml', 'r') as f:
    meta_data = yaml.safe_load(f)

for layer_n in range(meta_data['layer_num']):
    curve = np.loadtxt(data_dir+source_dir+f'slice{layer_n}_0.csv',delimiter=',')
    if not np.all(curve[0]==curve[-1]):
        curve = np.vstack((curve,curve[0])) # make it closed if not closed
    curve_lambda = np.cumsum(np.linalg.norm(np.diff(curve[:,:3],axis=0),axis=1))
    curve_lambda = np.insert(curve_lambda,0,0)
    total_lambda = curve_lambda[-1]
    sample_lambda = np.arange(0,total_lambda,path_dl)

    curve_resampled = []
    for curve_i in range(6):
        curve_resampled.append(np.interp(sample_lambda, curve_lambda, curve[:,curve_i]))
    curve_resampled = np.array(curve_resampled).T

    # for curve_i in range(curve_resampled.shape[0]):
    #     curve_resampled[curve_i,3:] = curve_resampled[curve_i,3:]/np.linalg.norm(curve_resampled[curve_i,3:])
    
    np.savetxt(data_dir+target_dir+f'slice{layer_n}_0.csv', curve_resampled, delimiter=',')
    print(f'layer {layer_n} done, from {curve.shape[0]} to {curve_resampled.shape[0]} points, total length {total_lambda:.2f}mm')

meta_data['path_dl'] = path_dl
with open(data_dir+'meta_data.yaml', 'w') as f:
    yaml.dump(meta_data, f)