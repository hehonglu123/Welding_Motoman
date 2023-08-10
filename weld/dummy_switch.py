import numpy as np
import glob
import yaml

#### data directory
dataset='cup/'
sliced_alg='circular_slice_shifted/'
curve_data_dir = '../data/'+dataset+sliced_alg
scan_data_dir = '../data/'+dataset+sliced_alg+'curve_scan_js/'
scan_p_data_dir = '../data/'+dataset+sliced_alg+'curve_scan_relative/'

#### welding spec, goal
with open(curve_data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)
line_resolution = slicing_meta['line_resolution']

for i in range(0,slicing_meta['num_layers']):
    num_sections=len(glob.glob(curve_data_dir+'curve_sliced_relative/slice'+str(i)+'_*.csv'))
    
    for x in range(num_sections):
        curve_sliced_relative=np.loadtxt(curve_data_dir+'curve_sliced_relative/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',')
        if len(curve_sliced_relative.shape)!=2:
            continue
        q_out2=np.loadtxt(curve_data_dir+'curve_scan_js/MA1440_js'+str(i)+'_'+str(x)+'.csv',delimiter=',')
        q_out1=np.loadtxt(curve_data_dir+'curve_scan_js/D500B_js'+str(i)+'_'+str(x)+'.csv',delimiter=',').reshape((-1,6))

        np.savetxt(scan_data_dir+'MA1440_js'+str(i)+'_'+str(x)+'.csv',q_out1,delimiter=',')
        np.savetxt(scan_data_dir+'D500B_js'+str(i)+'_'+str(x)+'.csv',q_out2,delimiter=',')