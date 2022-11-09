import numpy as np
import sys
sys.path.append('../toolbox')
from robots_def import *
from utils import *

artec_H=np.loadtxt('artec_spider_z_forward.csv',delimiter=',')
mounting_H=np.loadtxt('artec_spider_mounting.csv',delimiter=',')

H_TCP=mounting_H@artec_H
np.savetxt('scanner_tcp.csv',H_TCP,delimiter=',')