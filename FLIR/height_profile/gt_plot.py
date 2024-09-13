import cv2, time, os
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *
from robot_def import *

# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/weld_scan_job205_v152023_07_27_13_23_06/'
config_dir='../../config/'

folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


for folder in folders:
    height_profile = np.load(data_dir+folder+'/scans/height_profile.npy')
    plt.plot(height_profile[:,0],height_profile[:,1])

plt.title('Height Profile Ground Truth')
plt.show()