
import numpy as np
import sys, traceback, time, copy, glob, yaml
from matplotlib import pyplot as plt
sys.path.append('../toolbox')
from traj_manipulation import *

dataset='tube/'
sliced_alg='dense_slice/'
data_dir='../../geometry_data/'+dataset+sliced_alg
with open(data_dir+'slicing.yml', 'r') as file:
    slicing_meta = yaml.safe_load(file)

prev_layer=None
slice_step=30
vis_step=1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

cur_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(0)+'_0.csv',delimiter=',').reshape((-1,6))
prev_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(0)+'_0.csv',delimiter=',').reshape((-1,6))

num_section_0=len(glob.glob(data_dir+'curve_sliced/slice0_*.csv'))
section_step=min(10,num_section_0)

for i in range(0,num_section_0-section_step,section_step):
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
prev_layer=np.loadtxt(data_dir+'curve_sliced/slice0_%i.csv'%(num_section_0-1),delimiter=',').reshape((-1,6))
# for i in range(1,slicing_meta['num_layers']-slice_step,slice_step):
for i in range(1,200-slice_step,slice_step):
    next_layer=np.loadtxt(data_dir+'curve_sliced/slice'+str(i+slice_step)+'_0.csv',delimiter=',').reshape((-1,6))
    cur_layer_copy=copy.deepcopy(cur_layer)
    cur_layer=spiralize(cur_layer,next_layer)
    cur_layer=spiralize(cur_layer,prev_layer,reversed=True)
    ax.plot3D(cur_layer[:-2,0],cur_layer[:-2,1],cur_layer[:-2,2])
    ax.scatter(cur_layer[-3,0],cur_layer[-3,1],cur_layer[-3,2],c='b')
    ax.scatter(cur_layer[0,0],cur_layer[0,1],cur_layer[0,2],c='b')

    prev_layer=copy.deepcopy(cur_layer_copy)
    cur_layer=copy.deepcopy(next_layer)

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.show()