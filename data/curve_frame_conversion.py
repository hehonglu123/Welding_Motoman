import numpy as np
import sys, glob
from general_robotics_toolbox import *
 
data_dir='cylinder/'
solution_dir='baseline/'

###reference frame transformation
curve_pose = np.loadtxt(data_dir+'curve_pose.csv',delimiter=',')
R_curve=curve_pose[:3,:3]
shift=curve_pose[:-1,-1]

for file in glob.glob(data_dir+'curve_sliced/*.csv'):
	curve=np.loadtxt(file,delimiter=',')

	curve_new=np.dot(R_curve,curve[:,:3].T).T+np.tile(shift,(len(curve),1))
	curve_normal_new=np.dot(R_curve,curve[:,3:].T).T

	filename=file.split('\\')[-1]
	np.savetxt(data_dir+solution_dir+'curve_sliced_in_base_frame/'+filename,np.hstack((curve_new,curve_normal_new)),delimiter=',')
