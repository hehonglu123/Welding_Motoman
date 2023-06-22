from stl import mesh
import numpy as np
import sys, copy, traceback, glob

sys.path.append('../toolbox')
from utils import *
from lambda_calc import *
    

dataset='cup/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
num_layers=527

curve_dense=[]

num_sections_next=len(glob.glob(data_dir+'curve_sliced_relative/slice0_*.csv'))
for layer in num_layers-1:
    slice_ith_layer=[]
    num_sections=copy.deepcopy(num_sections_next)
    num_sections_next=len(glob.glob(data_dir+'curve_sliced_relative/slice'+str(layer+1)+'_*.csv'))
    ###if multiple disconnected section within a layer, do nothing
    if num_sections>1 or num_sections_next>1:
        for x in range(num_sections):
            curve_sliced=np.loadtxt(data_dir+'curve_sliced/slice'+str(layer)+'_'+str(x)+'.csv',delimiter=',')
            slice_ith_layer.append(curve_sliced)
        curve_dense.append(slice_ith_layer)
        continue
    else:
        ###if single section in the layer
        curve_cur=np.loadtxt(data_dir+'curve_sliced/slice'+str(layer)+'_0.csv',delimiter=',')
        curve_next=np.loadtxt(data_dir+'curve_sliced/slice'+str(layer+1)+'_0.csv',delimiter=',')


    
