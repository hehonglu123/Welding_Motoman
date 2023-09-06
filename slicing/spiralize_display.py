
import numpy as np
import sys, traceback, time, copy, glob, yaml
from matplotlib import pyplot as plt
sys.path.append('../toolbox')
from traj_manipulation import *

dataset='cup/'
sliced_alg='circular_slice/'
data_dir='../data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

prev_layer=None
slice_step=30
section_step=10
vis_step=1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

cur_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(0)+'_0.csv',delimiter=',').reshape((-1,6))
prev_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(0)+'_0.csv',delimiter=',').reshape((-1,6))

for i in range(0,128-section_step,section_step):
    next_layer=np.loadtxt(data_dir+'curve_sliced/slice0'+'_'+str(i+section_step)+'.csv',delimiter=',').reshape((-1,6))

    cur_layer_copy=copy.deepcopy(cur_layer)
    # cur_layer=spiralize(cur_layer,next_layer)
    # cur_layer=spiralize(cur_layer,prev_layer,reversed=True)
    ax.plot3D(cur_layer[:-3,0],cur_layer[:-3,1],cur_layer[:-3,2],'g.-')
    ax.scatter(cur_layer[-3,0],cur_layer[-3,1],cur_layer[-3,2],c='b')
    ax.scatter(cur_layer[0,0],cur_layer[0,1],cur_layer[0,2],c='b')

    prev_layer=copy.deepcopy(cur_layer_copy)
    cur_layer=copy.deepcopy(next_layer)

cur_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(1)+'_0.csv',delimiter=',').reshape((-1,6))
prev_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(0)+'_115.csv',delimiter=',').reshape((-1,6))
for i in range(1,slicing_meta['num_layers']-slice_step,slice_step):
    next_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(i+slice_step)+'_0.csv',delimiter=',').reshape((-1,6))
    cur_layer_copy=copy.deepcopy(cur_layer)
    # cur_layer=spiralize(cur_layer,next_layer)
    # cur_layer=spiralize(cur_layer,prev_layer,reversed=True)
    ax.plot3D(cur_layer[:-3,0],cur_layer[:-3,1],cur_layer[:-3,2],'r.-')
    ax.scatter(cur_layer[-3,0],cur_layer[-3,1],cur_layer[-3,2],c='b')
    ax.scatter(cur_layer[0,0],cur_layer[0,1],cur_layer[0,2],c='b')

    prev_layer=copy.deepcopy(cur_layer_copy)
    cur_layer=copy.deepcopy(next_layer)

plt.show()