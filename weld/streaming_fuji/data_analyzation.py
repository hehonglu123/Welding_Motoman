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
# logdata_dir_all = ['weld_fujicontrol_2025_03_12_18_27_33/']
# logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/']
logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/','weld_fujiscan_2025_02_26_16_24_21/','weld_fujicontrol_2025_03_12_18_27_33/']

### parameters ###
input_signals = ['cmd_v','cmd_feedrate']
# input_signals = ['v','cmd_feedrate']
# input_signals = ['cmd_v','cmd_VPD','torch_height']
# input_signals = ['v','feedrate','power']
# control_signals = ['v','feedrate','power']
control_signals = []
# output_signals = ['dheight','width','thermal']
output_signals = ['dheight','width']

## mimicing welding model recursive update (v_torch, dh_bead only)
P_mat = np.eye(2)*0.1 # initial covariance
lambda_fac = 0.99
theta_param = deepcopy(material_param['ER_4043']['100ipm'])
## mimicing welding model recursive update (v_torch, v_wire, dh_bead, dw_bead)
P_mat_dh = np.eye(4)*100 # initial covariance
P_mat_dw = np.eye(4)*100 # initial covariance
lambda_fac = 0.9
theta_param_dh = []
theta_param_dw = []
###

### drawing ###
v_torch_plot_min = 2
v_torch_plot_max = 15
# v_wire_plot_min = 80*inch2mm/60
# v_wire_plot_max = 250*inch2mm/60
v_wire_plot_min = 2
v_wire_plot_max = 7
dataScatters = []
modelcurveLines = []
cmap = plt.get_cmap('tab20')
# fig, ax = plt.subplots()
# ax.set_ylim(0.1,1.9)
# ax[0].set_ylim(0.1,1.9)
# z_model = np.polyfit(np.log([v_torch_plot_min,v_torch_plot_max]), np.log(v2dh_loglog([v_torch_plot_min,v_torch_plot_max],100)), 1)
# p_model = np.poly1d(z_model)
# line, = ax.plot(np.log([v_torch_plot_min,v_torch_plot_max]), p_model(np.log([v_torch_plot_min,v_torch_plot_max])), 'g--')
# modelcurveLines.append(line)

fig, ax = plt.subplots(2,1,figsize=(7,9), sharex=True)
# 
ax[0].set_xlim(np.log(v_torch_plot_min),np.log(v_torch_plot_max))
ax[0].set_ylim(np.log(v_wire_plot_min),np.log(v_wire_plot_max))
ax[1].set_xlim(np.log(v_torch_plot_min),np.log(v_torch_plot_max))
ax[1].set_ylim(np.log(v_wire_plot_min),np.log(v_wire_plot_max))
im_dh = ax[0].imshow(np.zeros((10,10)), extent=(np.log(v_torch_plot_min),np.log(v_torch_plot_max),np.log(v_wire_plot_min),np.log(v_wire_plot_max)), aspect='equal')
im_dw = ax[1].imshow(np.zeros((10,10)), extent=(np.log(v_torch_plot_min),np.log(v_torch_plot_max),np.log(v_wire_plot_min),np.log(v_wire_plot_max)), aspect='equal')

plt.ion()  # Turn on interactive mode
plt.show()
###

data_dh_all = []
data_dw_all = []
data_v_wire_all = []
data_v_torch_all = []

for logdata_dir_name in logdata_dir_all:
    logdata_dir = data_dir + logdata_dir_name

    # # read weld data
    # with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
    #     meta_data = yaml.safe_load(f)

    data_pairs = {}

    # for control_signal in control_signals:
    #     data_pairs[control_signal] = {}
    #     for output_signal in output_signals:
    #         data_pairs[control_signal][output_signal] = []

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

        weld_data['cmd_feedrate'] = np.array(weld_data['cmd_feedrate'])*inch2mm/60 # from inch/min to mm/sec

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
            v_torch = np.array(weld_data['cmd_v'])
            v_wire = np.array(weld_data['cmd_feedrate'])
            dh_bead = np.array(weld_data['dheight'])
            dw_bead = np.array(weld_data['width'])
            # v_torch = np.array(weld_data['v'])[::10]
            # v_wire = np.array(weld_data['feedrate'])[::10]
            # dh_bead = np.array(weld_data['dheight'])[::10]
            # dw_bead = np.array(weld_data['width'])[::10]
            
            assert len(v_torch) == len(v_wire) == len(dh_bead) == len(dw_bead), "The length of input signals are not the same"

            v_torch_mean = []
            v_wire_mean = []
            dh_bead_mean = []
            dw_bead_mean = []
            while len(v_torch)>0:
                # find the smallest index of v_torch different from v_torch[0]
                v_torch_diff_index = np.where(v_torch!=v_torch[0])[0]
                v_wire_diff_index = np.where(v_wire!=v_wire[0])[0]
                if len(v_torch_diff_index) == 0 and len(v_wire_diff_index) == 0:
                    diff_index = len(v_torch)
                elif len(v_torch_diff_index) == 0:
                    diff_index = v_wire_diff_index[0]
                elif len(v_wire_diff_index) == 0:
                    diff_index = v_torch_diff_index[0]
                else:
                    diff_index = np.min([v_torch_diff_index[0],v_wire_diff_index[0]])
                # get mean dh dw for the same v_torch and v_wire
                v_torch_mean.append(v_torch[0])
                v_wire_mean.append(v_wire[0])
                dh_bead_mean.append(np.mean(dh_bead[:diff_index]))
                dw_bead_mean.append(np.mean(dw_bead[:diff_index]))
                # remove the used data
                v_torch = v_torch[diff_index:]
                v_wire = v_wire[diff_index:]
                dh_bead = dh_bead[diff_index:]
                dw_bead = dw_bead[diff_index:]

            v_torch = np.array(v_torch_mean)
            v_wire = np.array(v_wire_mean)
            dh_bead = np.array(dh_bead_mean)
            dw_bead = np.array(dw_bead_mean)
            # remove data with dh_bead < 0
            v_torch = v_torch[dh_bead>0]
            v_wire = v_wire[dh_bead>0]
            dw_bead = dw_bead[dh_bead>0]
            dh_bead = dh_bead[dh_bead>0]
            # remove data with dw_bead <= 0
            v_torch = v_torch[dw_bead>0]
            v_wire = v_wire[dw_bead>0]
            dh_bead = dh_bead[dw_bead>0]
            dw_bead = dw_bead[dw_bead>0]
            # remove data with v_wire <= 0
            v_torch = v_torch[v_wire>0]
            dh_bead = dh_bead[v_wire>0]
            dw_bead = dw_bead[v_wire>0]
            v_wire = v_wire[v_wire>0]
            # add to all data
            data_dh_all.extend(dh_bead)
            data_dw_all.extend(dw_bead)
            data_v_wire_all.extend(v_wire)
            data_v_torch_all.extend(v_torch)

            ### RLS update (v_torch, dh_bead only)
            X_new_input = np.vstack((np.log(v_torch), np.ones_like(v_torch))).T
            dh_new_output = np.log(dh_bead)
            K_gain = P_mat@X_new_input.T@np.linalg.inv(lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@P_mat@X_new_input.T)
            theta_param = theta_param + K_gain@(dh_new_output-X_new_input@theta_param)
            P_mat = (P_mat-K_gain@X_new_input@P_mat)/lambda_fac
            # ### drawing
            # scatter = ax.scatter(X_new_input[:,0], Y_new_output,c=cmap(layer_n%10*2),s=5)
            # if len(modelcurveLines)<=1:
            #     line_newcurve, = ax.plot(np.log([v_torch_plot_min,v_torch_plot_max]), theta_param@np.vstack((np.log([v_torch_plot_min,v_torch_plot_max]),[1,1])), 'r--')
            #     modelcurveLines.append(line_newcurve)
            # else:
            #     line_newcurve.set_data(np.log([v_torch_plot_min,v_torch_plot_max]), theta_param@np.vstack((np.log([v_torch_plot_min,v_torch_plot_max]),[1,1])))
            # dataScatters.append(scatter)
            # print("The velocity when dh=2.3 using new theta_param", np.exp((np.log(2.3)-theta_param[1])/theta_param[0]))
            # # if lauer_n_id == 2:
            # #     plt.pause(15)
            # plt.pause(0.1)
            # plt.draw()
            # ###

            ### RLS update (v_torch, v_wire, dh_bead, dw_bead)
            # model logdh = a0*logv_torch*logv_wire + a1*logv_torch + a2*logv_wire + a3
            # model logdw = b0*logv_torch*logv_wire + b1*logv_torch + b2*logv_wire + b3
            X_new_input = np.vstack((np.log(v_torch)*np.log(v_wire), np.log(v_torch), np.log(v_wire), np.ones_like(v_torch))).T
            dh_new_output = np.log(dh_bead)
            dw_new_output = np.log(dw_bead)
            if len(theta_param_dh) == 0:
                theta_param_dh = np.linalg.pinv(X_new_input)@dh_new_output
                theta_param_dw = np.linalg.pinv(X_new_input)@dw_new_output
            else:
                K_gain_dh = P_mat_dh@X_new_input.T@np.linalg.inv(lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@P_mat_dh@X_new_input.T)
                theta_param_dh = theta_param_dh + K_gain_dh@(dh_new_output-X_new_input@theta_param_dh)
                P_mat_dh = (P_mat_dh-K_gain_dh@X_new_input@P_mat_dh)/lambda_fac
                K_gain_dw = P_mat_dw@X_new_input.T@np.linalg.inv(lambda_fac*np.eye(X_new_input.shape[0])+X_new_input@P_mat_dw@X_new_input.T)
                theta_param_dw = theta_param_dw + K_gain_dw@(dw_new_output-X_new_input@theta_param_dw)
                P_mat_dw = (P_mat_dw-K_gain_dw@X_new_input@P_mat_dw)/lambda_fac
            print("dh bead model",theta_param_dh)
            print("dw bead model",theta_param_dw)
            ### drawing
            # for ax_i in range(2):
            #     scatter = ax[ax_i].scatter(np.log(v_torch), np.log(v_wire),c=cmap(layer_n%10*2),s=5)
            #     dataScatters.append(scatter)
            v_torch_plot = np.linspace(np.log(v_torch_plot_min),np.log(v_torch_plot_max),100)
            v_wire_plot = np.linspace(np.log(v_wire_plot_min),np.log(v_wire_plot_max),100)
            v_torch_plot, v_wire_plot = np.meshgrid(v_torch_plot, v_wire_plot)
            dh_bead_plot = theta_param_dh[0]*v_torch_plot*v_wire_plot + theta_param_dh[1]*v_torch_plot + theta_param_dh[2]*v_wire_plot + theta_param_dh[3]
            dw_bead_plot = theta_param_dw[0]*v_torch_plot*v_wire_plot + theta_param_dw[1]*v_torch_plot + theta_param_dw[2]*v_wire_plot + theta_param_dw[3]
            im_dh.set_data(dh_bead_plot)
            im_dh.set_clim(np.min(dh_bead_plot),np.max(dh_bead_plot))
            cmap = im_dh.get_cmap()
            scatter_dh = ax[0].scatter(np.log(v_torch), np.log(v_wire),c=cmap((np.log(dh_bead)-np.min(dh_bead_plot))/np.max(dh_bead_plot)),s=5)
            dataScatters.append(scatter_dh)
            im_dw.set_data(dw_bead_plot)
            im_dw.set_clim(np.min(dw_bead_plot),np.max(dw_bead_plot))
            cmap = im_dw.get_cmap()
            scatter_dw = ax[1].scatter(np.log(v_torch), np.log(v_wire),c=cmap((np.log(dw_bead)-np.min(dw_bead_plot))/np.max(dw_bead_plot)),s=5)
            dataScatters.append(scatter_dw)

            plt.pause(0.001)
            plt.draw()

plt.ioff()  # Turn off interactive mode
plt.show()

print("===============")
### plot the data using all data
# X_new_input = np.vstack((np.log(v_torch)*np.log(v_wire), np.log(v_torch), np.log(v_wire), np.ones_like(v_torch))).T
# X_new_input = np.vstack((np.log(data_v_torch_all)*np.log(data_v_wire_all), np.log(data_v_torch_all), np.log(data_v_wire_all), np.ones_like(data_v_torch_all))).T
X_new_input = np.vstack((np.log(data_v_torch_all), np.log(data_v_wire_all), np.ones_like(data_v_torch_all))).T
dh_new_output = np.log(data_dh_all)
dw_new_output = np.log(data_dw_all)
theta_param_dh = np.linalg.pinv(X_new_input)@dh_new_output
theta_param_dw = np.linalg.pinv(X_new_input)@dw_new_output
print("dh bead model",theta_param_dh)
print("dw bead model",theta_param_dw)
fig, ax = plt.subplots(2,1,figsize=(7,9), sharex=True)
v_torch_plot = np.linspace(np.log(v_torch_plot_min),np.log(v_torch_plot_max),100)
v_wire_plot = np.linspace(np.log(v_wire_plot_min),np.log(v_wire_plot_max),100)
v_torch_plot, v_wire_plot = np.meshgrid(v_torch_plot, v_wire_plot)
# dh_bead_plot = theta_param_dh[0]*v_torch_plot*v_wire_plot + theta_param_dh[1]*v_torch_plot + theta_param_dh[2]*v_wire_plot + theta_param_dh[3]
# dw_bead_plot = theta_param_dw[0]*v_torch_plot*v_wire_plot + theta_param_dw[1]*v_torch_plot + theta_param_dw[2]*v_wire_plot + theta_param_dw[3]
dh_bead_plot = theta_param_dh[0]*v_torch_plot + theta_param_dh[1]*v_wire_plot + theta_param_dh[2]
dw_bead_plot = theta_param_dw[0]*v_torch_plot + theta_param_dw[1]*v_wire_plot + theta_param_dw[2]
im_dh = ax[0].imshow(dh_bead_plot, extent=(np.log(v_torch_plot_min),np.log(v_torch_plot_max),np.log(v_wire_plot_min),np.log(v_wire_plot_max)), aspect='equal')
cmap = im_dh.get_cmap()
ax[0].scatter(np.log(data_v_torch_all), np.log(data_v_wire_all),c=cmap((np.log(data_dh_all)-np.min(dh_bead_plot))/np.max(dh_bead_plot)),s=5)
im_dw = ax[1].imshow(dw_bead_plot, extent=(np.log(v_torch_plot_min),np.log(v_torch_plot_max),np.log(v_wire_plot_min),np.log(v_wire_plot_max)), aspect='equal')
cmap = im_dw.get_cmap()
ax[1].scatter(np.log(data_v_torch_all), np.log(data_v_wire_all),c=cmap((np.log(data_dw_all)-np.min(dw_bead_plot))/np.max(dw_bead_plot)),s=5)
plt.colorbar(im_dh, ax=ax[0])
plt.colorbar(im_dw, ax=ax[1])
ax[0].set_xlabel('log v_torch')
ax[1].set_xlabel('log v_torch')
ax[0].set_ylabel('log v_wire')
ax[1].set_ylabel('log v_wire')
ax[0].set_title('dh bead model')
ax[1].set_title('dw bead model')
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

            if input_sig_key in ['cmd_v', 'cmd_feedrate'] and output_sig_key in ['dheight','width']:
                print("Any y_values lower than 0?",np.any(np.array(y_values)<0))
                x_values = np.log(x_values)
                y_values = np.log(y_values)
                z_model = np.polyfit(np.log([3,4]), np.log(v2dh_loglog([3,4],100)), 1)
                p_model = np.poly1d(z_model)

            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            axs[j,i].scatter(x_values, y_values, s=5)
            if input_sig_key == 'cmd_v':
                axs[j,i].plot(np.log([v_torch_plot_min,v_torch_plot_max]), p(np.log([v_torch_plot_min,v_torch_plot_max])), 'r--')
            elif input_sig_key == 'cmd_feedrate':
                axs[j,i].plot(np.log([v_wire_plot_min,v_wire_plot_max]), p(np.log([v_wire_plot_min,v_wire_plot_max])), 'r--')
            if input_sig_key == 'cmd_v' and output_sig_key in ['dheight']:
                axs[j,i].plot(np.log([v_torch_plot_min,v_torch_plot_max]), p_model(np.log([v_torch_plot_min,v_torch_plot_max])), 'g--')
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