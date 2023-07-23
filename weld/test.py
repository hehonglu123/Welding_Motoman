import math
from copy import deepcopy
from pathlib import Path
import pickle
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from utils import *
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from weld_dh2v import *
from scipy.optimize import curve_fit
from weld_dh2v import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import open3d as o3d

V = 6
a_hat = -1.1242317716329688
b_hat = 0.801603997042284

log_h = a_hat * np.log(V) + b_hat
h = np.exp(log_h)
print(log_h)
print(h)