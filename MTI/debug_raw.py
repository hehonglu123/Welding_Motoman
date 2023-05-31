import time, sys, pickle

sys.path.append('../toolbox/')
from utils import *
from robot_def import *

def raw_data_filtering(single_scan):
    ###delete noise centered at x=0
    single_scan=np.delete(single_scan,np.argwhere(abs(single_scan[:,0])==0).flatten(),axis=0)
    ###clip data within range
    indices=np.argwhere(single_scan[:,1]>50).flatten()
    single_scan=single_scan[indices]
    ###filter out outlier noise
    outlier_indices=identify_outliers2(single_scan[:,1],threshold=1e-2)
    
    return np.delete(single_scan,outlier_indices,axis=0)


robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

joint_recording=np.loadtxt('recording3/joint_recording.csv',delimiter=',')
with open('recording3/mti_recording.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

pc=[]
fig = plt.figure(1)


for i in range(len(mti_recording)):
    line_scan=raw_data_filtering(mti_recording[i].T) ###filter out bad points
    plt.plot(line_scan[:,0],line_scan[:,1], "x")
    plt.xlim(-30,30)
    plt.ylim(0,120)
    plt.title('SCAN'+str(i))
    plt.pause(0.001)
    plt.clf()


# i=449
# line_scan=raw_data_filtering(mti_recording[i].T) ###filter out bad points
# plt.plot(line_scan[:,0],line_scan[:,1], "x")
# plt.xlim(-30,30)
# plt.ylim(0,120)
# plt.show()