import numpy as np
import yaml
import glob

logdata_dirs = ['weld_fujiscan_2025_06_11_16_27_41/','weld_fujiscan_2025_06_11_16_52_36/','weld_fujiscan_2025_06_11_17_16_48/',\
        'weld_fujiscan_2025_06_11_17_49_27/','weld_fujiscan_2025_06_11_18_14_56/','weld_fujiscan_2025_06_12_17_33_24/',\
        'weld_fujiscan_2025_06_12_16_59_09/','weld_fujiscan_2025_06_12_15_33_03/','weld_fujiscan_2025_06_12_15_03_27/',\
        'weld_fujiscan_2025_07_09_14_52_42/','weld_fujiscan_2025_07_09_15_21_35/','weld_fujiscan_2025_07_09_16_16_40/']

data_dir = 'wall_weld_test/'
max_layers = 0

control_parameters = {}
for logdata_dir in logdata_dirs:
    
    data_name = logdata_dir[14:-4] # remove 'weld_fujiscan_' and the last '_xx/'
    control_parameters[data_name] = {} # record control parameters for each log data

    this_data_dir = data_dir + logdata_dir

    for weld_part in ['baselayer','layer']:
        total_layers_name = glob.glob(this_data_dir+weld_part+'*')
        layer_nums = []
        for layer_name in total_layers_name:
            this_layer = layer_name.split('\\')[-1]
            this_layer = this_layer.split('r')[-1]
            layer_nums.append(int(this_layer))
        layer_nums = np.sort(layer_nums)
        for layer_i, layer_n in enumerate(layer_nums):
            weld_cmd = np.loadtxt(this_data_dir+f'{weld_part}{layer_n}/weld_cmd.csv', delimiter=',')
            mid_weld_cmd = weld_cmd[len(weld_cmd)//2]
            control_parameters[data_name][weld_part+f'{layer_i}'] = {}
            control_parameters[data_name][weld_part+f'{layer_i}']['speed'] = mid_weld_cmd[-2]
            control_parameters[data_name][weld_part+f'{layer_i}']['wire_feed'] = mid_weld_cmd[-1]
        
        if len(layer_nums) > max_layers:
            max_layers = len(layer_nums)
print(f'Max layers found: {max_layers}')
# print the control parameters in a table and save to a csv file
# the row is baselayer0, baselayer1, layer0, layer1, ...
# the column is the log data name
# the value is (speed, wire_feed), two decimal places
import os
if not os.path.exists(data_dir+'weld_control_parameters.csv'):
    os.makedirs(data_dir, exist_ok=True)

table_str = 'layer/logdata,'
for logdata_dir in logdata_dirs:
    data_name = logdata_dir[14:-4] # remove 'weld_fujiscan_' and the last '_xx/'
    table_str += f'{data_name},'
table_str += '\n'
for weld_part in ['baselayer','layer']:
    for layer_i in range(max_layers):
        if layer_i==2 and weld_part=='baselayer':
            break # only 2 baselayers
        layer_name = weld_part + f'{layer_i}'
        table_str += f'{layer_name},'
        for logdata_dir in logdata_dirs:
            data_name = logdata_dir[14:-4] # remove 'weld_fujiscan_' and the last '_xx/'
            if layer_name in control_parameters[data_name]:
                speed = control_parameters[data_name][layer_name]['speed']
                wire_feed = control_parameters[data_name][layer_name]['wire_feed']
                table_str += f'({speed:.2f}/{int(wire_feed)}),'
            else:
                table_str += 'N/A,'
        table_str += '\n'
with open(data_dir+'weld_control_parameters.csv', 'w') as f:
    f.write(table_str)
print(f'Control parameters saved to {data_dir}weld_control_parameters.csv')