from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../')
sys.path.append('../../toolbox/')
sys.path.append('../../scan/scan_tools/')
sys.path.append('../../scan/scan_plan/')
sys.path.append('../../scan/scan_process/')
from robot_def import *
from utils import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *

from general_robotics_toolbox import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np

data_dir='data/weld_scan_2023_11_01_17_44_58/'
dh=2.5
yk_d=[dh]

total_iteration=9

plt.axhline(y = yk_d[0], color = 'r', linestyle = '-') 
for iter_read in range(total_iteration):
    yk_d=[dh,dh]

    yk=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk.csv',delimiter=',')
    # try:
    #     yk_prime=np.loadtxt(data_dir+'iteration_'+str(iter_read)+'/yk_prime.csv',delimiter=',')
    # except:
    #     yk_prime=deepcopy(yk)
    plt.scatter(np.arange(len(yk)),yk,label='iter '+str(iter_read))
    # plt.scatter(np.arange(len(yk_prime)),yk_prime)
plt.legend()
plt.show()