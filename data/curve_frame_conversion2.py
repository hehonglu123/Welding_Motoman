import numpy as np
import sys, glob
from general_robotics_toolbox import *
 
dataset='blade0.1/'
sliced_alg='NX_slice2/'
data_dir='../data/'+dataset+sliced_alg
curve_pose = np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')

R_curve=curve_pose[:3,:3]
shift=curve_pose[:-1,-1]

for file in glob.glob(data_dir+'curve_sliced/*.csv'):
	curve_sliced=np.loadtxt(file,delimiter=',')

	curve_new=np.dot(R_curve,curve_sliced[:,:3].T).T+np.tile(shift,(len(curve_sliced),1))
	curve_normal_new=np.dot(R_curve,curve_sliced[:,3:].T).T

	filename=file.split('\\')[-1]
	np.savetxt(data_dir+'curve_sliced_relative/'+filename,np.hstack((curve_new,curve_normal_new)),delimiter=',')
