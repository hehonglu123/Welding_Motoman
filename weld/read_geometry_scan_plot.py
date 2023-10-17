import numpy as np
from matplotlib import pyplot as plt

#### data directory
# dataset='circle_large/'
# sliced_algs = ['static_stepwise_shift','static_stepwise_zero','static_stepwise_shift']
# collected_data = ['weld_scan_baseline_2023_09_18_16_17_34','weld_scan_correction_2023_09_19_18_03_53','weld_scan_correction_2023_09_19_16_32_31']
# legends=['Baseline','Correction Zero','Correction Shift']

dataset='blade0.1/'
sliced_algs = ['auto_slice','auto_slice']
collected_data = ['weld_scan_baseline_2023_10_09_16_01_52','weld_scan_correction_2023_10_10_16_56_32']
legends=['Baseline','Correction']

print("Average/Std of height std:")
total_datasets=len(collected_data)
for i in range(total_datasets):
    sliced_alg=sliced_algs[i]
    curve_data_dir = '../data/'+dataset+sliced_alg+'/'
    method=collected_data[i]
    data_dir=curve_data_dir+method+'/'
    height_std = np.load(data_dir+'height_std.npy')
    print(legends[i],':',round(np.mean(height_std),2),',',round(np.std(height_std),2))
    plt.plot(height_std,'-o',label=legends[i])
    if i==0:
        baseline_performance=np.mean(height_std)
    elif i==1:
        correction_performance=np.mean(height_std)
print("Improvements: ",round((baseline_performance-correction_performance)/baseline_performance*100),'%')
plt.legend()
plt.xlabel("Layer #")
plt.ylabel("STD (mm)")
plt.title('Height STD')
plt.show()
print("==========================")

print("Average/Std of height Tracking Error Norm:")
total_datasets=len(collected_data)
for i in range(total_datasets):
    sliced_alg=sliced_algs[i]
    curve_data_dir = '../data/'+dataset+sliced_alg+'/'
    method=collected_data[i]
    data_dir=curve_data_dir+method+'/'
    error_norm = np.load(data_dir+'height_error_norm.npy')
    print(legends[i],':',round(np.mean(error_norm),2),',',round(np.std(error_norm),2))
    plt.plot(error_norm,'-o',label=legends[i])
    if i==0:
        baseline_performance=np.mean(error_norm)
    elif i==1:
        correction_performance=np.mean(error_norm)
print("Improvements: ",round((baseline_performance-correction_performance)/baseline_performance*100),'%')
plt.legend()
plt.xlabel("Layer #")
plt.ylabel("Norm (mm)")
plt.title('Layer Tracking Error Norm')
plt.show()
print("==========================")

print("Average/Std of height Tracking RMSE:")
total_datasets=len(collected_data)
for i in range(total_datasets):
    sliced_alg=sliced_algs[i]
    curve_data_dir = '../data/'+dataset+sliced_alg+'/'
    method=collected_data[i]
    data_dir=curve_data_dir+method+'/'
    error_rmse = np.load(data_dir+'height_rmse.npy')
    print(legends[i],':',round(np.mean(error_rmse),2),',',round(np.std(error_rmse),2))
    plt.plot(error_rmse,'-o',label=legends[i])
    if i==0:
        baseline_performance=np.mean(error_rmse)
    elif i==1:
        correction_performance=np.mean(error_rmse)
print("Improvements: ",round((baseline_performance-correction_performance)/baseline_performance*100),'%')
plt.legend()
plt.xlabel("Layer #")
plt.ylabel("RMSE (mm)")
plt.title('Layer Tracking RMSE')
plt.show()
print('========================')