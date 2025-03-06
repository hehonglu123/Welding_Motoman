import glob, yaml
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

inch2mm = 25.4
mm2inch = 1/inch2mm
cross_section = 1.2

ignore_start_end = 8

data_dir = '../../data/wall_weld_test/'

# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_17_39_17/']
logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/']

# input_signals = ['cmd_v','cmd_feedrate']
# input_signals = ['cmd_v','cmd_VPD','torch_height']
input_signals = ['v','feedrate','power']
# control_signals = ['v','feedrate','power']
control_signals = []
output_signals = ['dheight','width','thermal']

data_pairs = {}

# for control_signal in control_signals:
#     data_pairs[control_signal] = {}
#     for output_signal in output_signals:
#         data_pairs[control_signal][output_signal] = []

for logdata_dir_name in logdata_dir_all:
    logdata_dir = data_dir + logdata_dir_name

    with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
        meta_data = yaml.safe_load(f)
    data_pairs[meta_data['VPD']] = {}

    for input_signal in input_signals:
        data_pairs[meta_data['VPD']][input_signal] = {}
        for output_signal in output_signals:
            data_pairs[meta_data['VPD']][input_signal][output_signal] = {}

    total_layers_name = glob.glob(logdata_dir+'layer*')
    layer_nums = []
    for layer_name in total_layers_name:
        this_layer = layer_name.split('\\')[-1]
        this_layer = this_layer.split('r')[-1]
        layer_nums.append(int(this_layer))
    layer_nums = np.sort(layer_nums)
    
    for layer_n in layer_nums:
        print('layer',layer_n)
        this_layer_dir = logdata_dir+'layer'+str(layer_n)+'/'
        weld_data = pd.read_csv(this_layer_dir + 'profile_welding.csv', header=0)
        weld_data = weld_data.to_dict(orient='list')

        x = np.array(weld_data['x'])
        # ignore the first and last 5 mm
        if x[-1]>x[0]:
            start_index = np.where(x > x[0]+ignore_start_end)[0][0] 
            end_index = np.where(x < x[-1]-ignore_start_end)[0][-1]
        else:
            start_index = np.where(x < x[0]-ignore_start_end)[0][0] 
            end_index = np.where(x > x[-1]+ignore_start_end)[0][-1]
        for k in weld_data.keys():
            weld_data[k] = weld_data[k][start_index:end_index+1]
        weld_data['power'] = np.array(weld_data['voltage'])*np.array(weld_data['current'])
        weld_data['torch_height'] = np.array(weld_data['torch_height'])+np.array(weld_data['dheight'])+15
        weld_data['cmd_VPD'] = cross_section*inch2mm*np.array(weld_data['cmd_feedrate'])/np.array(weld_data['cmd_v'])

        for input_sig_key in data_pairs[meta_data['VPD']].keys():
            for output_sig_key in data_pairs[meta_data['VPD']][input_sig_key].keys():
                if input_sig_key in control_signals:
                    data_pairs[meta_data['VPD']][input_sig_key][output_sig_key].extend(np.vstack((weld_data[input_sig_key], weld_data[output_sig_key])).T)
                else:
                    for (data_input,data_output) in zip(weld_data[input_sig_key], weld_data[output_sig_key]):
                        if data_input in data_pairs[meta_data['VPD']][input_sig_key][output_sig_key]:
                            data_pairs[meta_data['VPD']][input_sig_key][output_sig_key][data_input].append(data_output)
                        else:
                            data_pairs[meta_data['VPD']][input_sig_key][output_sig_key][data_input] = [data_output]
    
fig, axs = plt.subplots(len(output_signals),len(input_signals))
for i,input_sig_key in enumerate(input_signals):
    for j,output_sig_key in enumerate(output_signals):
        for data_VPD in data_pairs.keys():
            x_values=[]
            y_values=[]
            for data_input in data_pairs[data_VPD][input_sig_key][output_sig_key].keys():
                if output_sig_key == 'width':
                    width_data = np.array(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
                    width_data = width_data[width_data>0.5]
                    if width_data.size > 0:
                        data_pairs[data_VPD][input_sig_key][output_sig_key][data_input] = np.mean(width_data)
                        x_values.append(data_input)
                        y_values.append(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
                    else:
                        data_pairs[data_VPD][input_sig_key][output_sig_key][data_input] = 0.5
                else:
                    data_pairs[data_VPD][input_sig_key][output_sig_key][data_input] = np.mean(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
                    x_values.append(data_input)
                    y_values.append(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
            # x_values = list(data_pairs[data_VPD][input_sig_key][output_sig_key].keys())
            # y_values = list(data_pairs[data_VPD][input_sig_key][output_sig_key].values())
            # fitna linear line
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            axs[j,i].scatter(x_values, y_values)
            axs[j,i].plot(x_values, p(x_values), 'r--')
            axs[j,i].set_title(input_sig_key + ' vs ' + output_sig_key)
plt.show()

for data_VPD in data_pairs.keys():
    feedrate_dh = data_pairs[data_VPD]['feedrate']['dheight']
    feedrate_dw = data_pairs[data_VPD]['feedrate']['width']
    dhdw_ratio = []
    for fdr in feedrate_dh.keys():
        if fdr in feedrate_dw:
            print(feedrate_dh[fdr])
            print(feedrate_dw[fdr])
            print("====")
            dhdw = feedrate_dh[fdr]*feedrate_dw[fdr]
            dhdw_ratio.append(dhdw)
    # plt.scatter([data_VPD]*len(dhdw_ratio), dhdw_ratio)
    plt.scatter(list(feedrate_dh.keys()), dhdw_ratio, label='VPD='+str(round(data_VPD)))
plt.legend()
plt.xlabel('feedrate')
plt.ylabel('dh*dw')
plt.title('feedrate vs dh*dw')
plt.show()