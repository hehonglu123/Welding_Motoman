import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *
import pickle
import time
import sys
from matplotlib import pyplot as plt

from numpy.random import default_rng
rng = default_rng()

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

ph_dataset_date='0913'
test_dataset_date='0913'
config_dir='../config/'

robot_type='R1_part2'

PH_data_dir='PH_grad_data/test'+ph_dataset_date+'_'+robot_type+'/train_data_'

with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
    PH_q=pickle.load(file)
#### all train data q
train_q = []
training_error=[]
for qkey in PH_q.keys():
    train_q.append(np.array(qkey))
    training_error.append(PH_q[qkey]['train_pos_error'])
train_q=np.array(train_q)
training_error_last=[]
training_error_init=[]
for te in training_error:
    training_error_last.append(np.mean(te[-1]))
    training_error_init.append(np.mean(te[0]))
training_error=training_error_last
#####################

#### using zero config PH ####
train_q_zero_index = np.argmin(np.linalg.norm(train_q-np.zeros(2),ord=2,axis=1))
train_q_zero_key = tuple(train_q[train_q_zero_index])
qzero_P = PH_q[train_q_zero_key]['P']
qzero_H = PH_q[train_q_zero_key]['H']
#############################

try:
    with open(PH_data_dir+'calib_one_PH.pickle','rb') as file:
        PH_q_one=pickle.load(file)
except:
    PH_q_one=PH_q[train_q_zero_key]
#### one PH for all ####
universal_P = PH_q_one['P']
universal_H = PH_q_one['H']
training_error_universal=PH_q_one['train_pos_error']
# plt.plot(np.mean(training_error_universal,axis=1))
# plt.xlabel("Iteration")
# plt.ylabel("Average Position Error Norm (mm)")
# plt.title("Average Position error norm of all poses")
# plt.show()
########################

print("Training Data")
markdown_str=''
markdown_str+='||Mean (mm)|Std (mm)|Max (mm)|\n'
markdown_str+='|-|-|-|-|\n'
markdown_str+='|Nominal PH|'+format(round(np.mean(training_error_init),4),'.4f')+'|'+\
    format(round(np.std(training_error_init),4),'.4f')+'|'+format(round(np.max(training_error_init),4),'.4f')+'|\n'
markdown_str+='|One PH|'+format(round(np.mean(training_error_universal[-1]),4),'.4f')+'|'+\
    format(round(np.std(training_error_universal[-1]),4),'.4f')+'|'+format(round(np.max(training_error_universal[-1]),4),'.4f')+'|\n'
markdown_str+='|Optimize PH|'+format(round(np.mean(training_error),4),'.4f')+'|'+\
    format(round(np.std(training_error),4),'.4f')+'|'+format(round(np.max(training_error),4),'.4f')+'|\n'
print(markdown_str)