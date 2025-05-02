import glob, yaml, sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
sys.path.append('../')

ignore_start_end = 10
start_x = -55 + ignore_start_end
end_x = 55 - ignore_start_end

data_dir = '../../data/wall_weld_test/'

# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_17_39_17/']
# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/']
# logdata_dir_all = ['weld_fujicontrol_2025_03_12_18_27_33/']
# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/']
logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/','weld_fujiscan_2025_02_26_16_24_21/','weld_fujicontrol_2025_03_12_18_27_33/']

for logdata_dir_name in logdata_dir_all:
    logdata_dir = data_dir + logdata_dir_name

    with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    
    total_layers_name = glob.glob(logdata_dir+'layer*')
    layer_nums = []
    for layer_name in total_layers_name:
        this_layer = layer_name.split('\\')[-1]
        this_layer = this_layer.split('r')[-1]
        layer_nums.append(int(this_layer))
    layer_nums = np.sort(layer_nums)
    
    for lauer_n_id,layer_n in enumerate(layer_nums):
        if layer_n == layer_nums[-1] and 'control' in logdata_dir_name:
            continue
        print('layer',layer_n)
        this_layer_dir = logdata_dir+'layer'+str(layer_n)+'/'
        weld_data = pd.read_csv(this_layer_dir + 'profile_welding.csv', header=0)
        weld_data = weld_data.to_dict(orient='list')

        timestamps = np.array(weld_data['# time'])
        x_location = np.array(weld_data['x'])

        if x_location[-1]>x_location[0]:
            start_index = np.where(x_location > start_x)[0][0] 
            end_index = np.where(x_location < end_x)[0][-1]
        else:
            start_index = np.where(x_location < end_x)[0][0] 
            end_index = np.where(x_location > start_x)[0][-1]

        timestamps = timestamps[start_index:end_index]

        print("time diff mean:", np.mean(np.diff(timestamps)))
        print("time diff std:", np.std(np.diff(timestamps)))
        print("time diff max:", np.max(np.diff(timestamps)))

        # plt.plot(timestamps)
        # plt.plot(np.diff(timestamps), label='diff time')
        # plt.show()