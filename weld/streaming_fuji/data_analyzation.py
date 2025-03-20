import glob, yaml, sys
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
sys.path.append('../')
from weld_dh2v import *

inch2mm = 25.4
mm2inch = 1/inch2mm
cross_section = 1.2

ignore_start_end = 5
start_x = -55 + ignore_start_end
end_x = 55 - ignore_start_end

data_dir = '../../data/wall_weld_test/'

# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_17_39_17/']
# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/']
logdata_dir_all = ['weld_fujicontrol_2025_03_12_18_27_33/']

input_signals = ['cmd_v','cmd_feedrate']
# input_signals = ['cmd_v','cmd_VPD','torch_height']
# input_signals = ['v','feedrate','power']
# control_signals = ['v','feedrate','power']
control_signals = []
# output_signals = ['dheight','width','thermal']
output_signals = ['dheight','width']

data_pairs = {}

# for control_signal in control_signals:
#     data_pairs[control_signal] = {}
#     for output_signal in output_signals:
#         data_pairs[control_signal][output_signal] = []

for logdata_dir_name in logdata_dir_all:
    logdata_dir = data_dir + logdata_dir_name

    ## mimicing welding model recursive update
    P_mat = np.eye(2)*0.1 # initial covariance
    lambda_fac = 0.99
    theta_param = deepcopy(material_param['ER_4043']['100ipm'])
    dataScatters = []
    modelcurveLines = []
    cmap = plt.get_cmap('tab20')
    v_plot_min = 1
    v_plot_max = 12
    fig, ax = plt.subplots()
    ax.set_ylim(0.1,1.9)
    plt.ion()  # Turn on interactive mode
    plt.show()
    z_model = np.polyfit(np.log([v_plot_min,v_plot_max]), np.log(v2dh_loglog([v_plot_min,v_plot_max],100)), 1)
    p_model = np.poly1d(z_model)
    line, = ax.plot(np.log([v_plot_min,v_plot_max]), p_model(np.log([v_plot_min,v_plot_max])), 'g--')
    modelcurveLines.append(line)
    ###

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
    
    for lauer_n_id,layer_n in enumerate(layer_nums):
        if layer_n == layer_nums[-1] and 'control' in logdata_dir_name:
            continue
        print('layer',layer_n)
        this_layer_dir = logdata_dir+'layer'+str(layer_n)+'/'
        weld_data = pd.read_csv(this_layer_dir + 'profile_welding.csv', header=0)
        weld_data = weld_data.to_dict(orient='list')

        x = np.array(weld_data['x'])
        # ignore the first and last x mm
        if x[-1]>x[0]:
            start_index = np.where(x > start_x)[0][0] 
            end_index = np.where(x < end_x)[0][-1]
        else:
            start_index = np.where(x < end_x)[0][0] 
            end_index = np.where(x > start_x)[0][-1]
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
        
        ## mimicing welding model update
        if lauer_n_id > 1:
            x_raw = np.array(weld_data['cmd_v'])
            y_raw = np.array(weld_data['dheight'])
            x_raw_mean = []
            y_raw_mean = []
            while len(x_raw)>0:
                x_raw_mean.append(x_raw[0])
                y_raw_mean.append(np.mean(y_raw[x_raw==x_raw[0]]))
                y_raw = y_raw[x_raw!=x_raw[0]]
                x_raw = x_raw[x_raw!=x_raw[0]]
            x_raw = np.array(x_raw_mean)
            y_raw = np.array(y_raw_mean)
            x_raw = x_raw[y_raw>0]
            y_raw = y_raw[y_raw>0]

            
            X_new_input = np.vstack((np.log(x_raw), np.ones_like(x_raw))).T
            Y_new_output = np.log(y_raw)
            K_gain = P_mat@X_new_input.T@np.linalg.inv(lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@P_mat@X_new_input.T)
            theta_param = theta_param + K_gain@(Y_new_output-X_new_input@theta_param)
            P_mat = (P_mat-K_gain@X_new_input@P_mat)/lambda_fac
            scatter = ax.scatter(X_new_input[:,0], Y_new_output,c=cmap(layer_n%10*2),s=5)
            if len(modelcurveLines)<=1:
                line_newcurve, = ax.plot(np.log([v_plot_min,v_plot_max]), theta_param@np.vstack((np.log([v_plot_min,v_plot_max]),[1,1])), 'r--')
                modelcurveLines.append(line_newcurve)
            else:
                line_newcurve.set_data(np.log([v_plot_min,v_plot_max]), theta_param@np.vstack((np.log([v_plot_min,v_plot_max]),[1,1])))
            # line.set_data()
            dataScatters.append(scatter)
            print("The velocity when dh=2.3 using new theta_param", np.exp((np.log(2.3)-theta_param[1])/theta_param[0]))
            # if lauer_n_id == 2:
            #     plt.pause(15)
            plt.pause(0.1)
            plt.draw()
            ###
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

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
                    # print("len of data_input",len(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input]))
                    data_pairs[data_VPD][input_sig_key][output_sig_key][data_input] = np.mean(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
                    x_values.append(data_input)
                    y_values.append(data_pairs[data_VPD][input_sig_key][output_sig_key][data_input])
            # x_values = list(data_pairs[data_VPD][input_sig_key][output_sig_key].keys())
            # y_values = list(data_pairs[data_VPD][input_sig_key][output_sig_key].values())
            # fitna linear line

            if input_sig_key == 'cmd_v' and output_sig_key in ['dheight','width']:
                print("Any y_values lower than 0?",np.any(np.array(y_values)<0))
                x_values = np.log(x_values)
                y_values = np.log(y_values)
                z_model = np.polyfit(np.log([3,4]), np.log(v2dh_loglog([3,4],100)), 1)
                p_model = np.poly1d(z_model)

            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            axs[j,i].scatter(x_values, y_values, s=5)
            axs[j,i].plot(np.log([v_plot_min,v_plot_max]), p(np.log([v_plot_min,v_plot_max])), 'r--')
            if input_sig_key == 'cmd_v' and output_sig_key in ['dheight']:
                axs[j,i].plot(np.log([v_plot_min,v_plot_max]), p_model(np.log([v_plot_min,v_plot_max])), 'g--')
                axs[i,j].set_ylim(0.1,1.9)
                axs[i,j].set_xlim(-0.1,2.6)
            axs[j,i].set_title(input_sig_key + ' vs ' + output_sig_key)
plt.show()

# for data_VPD in data_pairs.keys():
#     feedrate_dh = data_pairs[data_VPD]['feedrate']['dheight']
#     feedrate_dw = data_pairs[data_VPD]['feedrate']['width']
#     dhdw_ratio = []
#     for fdr in feedrate_dh.keys():
#         if fdr in feedrate_dw:
#             print(feedrate_dh[fdr])
#             print(feedrate_dw[fdr])
#             print("====")
#             dhdw = feedrate_dh[fdr]*feedrate_dw[fdr]
#             dhdw_ratio.append(dhdw)
#     # plt.scatter([data_VPD]*len(dhdw_ratio), dhdw_ratio)
#     plt.scatter(list(feedrate_dh.keys()), dhdw_ratio, label='VPD='+str(round(data_VPD)))
# plt.legend()
# plt.xlabel('feedrate')
# plt.ylabel('dh*dw')
# plt.title('feedrate vs dh*dw')
# plt.show()