import time, sys, pickle

sys.path.append('../toolbox/')
from utils import *
from robot_def import *

robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

joint_recording=np.loadtxt('recording1/joint_recording.csv',delimiter=',')
with open('recording1/mti_recording.pickle', 'rb') as file:
    mti_recording=pickle.load(file)

pc=[]
fig = plt.figure(1)

for i in range(len(mti_recording)):
    print(np.argwhere(mti_recording[i][1,:]<10))
    line_scan=np.vstack((np.zeros(len(mti_recording[i][0])),mti_recording[i])).T
    pose=robot2.fwd(joint_recording[i])
    pc.extend(transform_curve(line_scan,H_from_RT(pose.R,pose.p)))

    plt.plot(mti_recording[i][0],mti_recording[i][1], "x")
    plt.xlim(-30,30)
    plt.ylim(0,120)
    plt.pause(0.001)
    plt.clf()