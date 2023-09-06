from general_robotics_toolbox import *
import numpy as np
from scipy.interpolate import interp1d

def spiralize(traj1,traj2,reversed=False):
	###interpolate traj1 to traj2 with spiral printing
	###interp traj2 to be of same length

	traj2_interp=interp1d(np.linspace(0,1,num=len(traj2)),traj2,axis=0)(np.linspace(0,1,num=len(traj1)))
	if not reversed:
		weight=np.linspace(1,0.5,num=len(traj1))
	else:
		weight=np.linspace(0.5,1,num=len(traj1))
	
	# weight=np.tile(weight,(len(traj1[0]),1)).T
	# traj_new=weight*traj1+(1-weight)*traj2_interp

	traj_new = weight[:, np.newaxis] * traj1 + (1 - weight[:, np.newaxis]) * traj2_interp
	
	return traj_new

def warp_traj(rob1_js,rob2_js,positioner_js,rob1_js_x,rob2_js_x,positioner_js_x,reversed=False):
	if positioner_js_x.shape==(2,) and rob1_js_x.shape==(6,):
		return rob1_js,rob2_js,positioner_js
	traj_length_x_half=int(len(rob1_js_x)/2)
	traj_length_half=int(len(rob1_js)/2)
	if reversed:
		rob1_js[:traj_length_half]=spiralize(rob1_js[:traj_length_half],rob1_js_x[:traj_length_x_half],reversed)
		rob2_js[:traj_length_half]=spiralize(rob2_js[:traj_length_half],rob2_js_x[:traj_length_x_half],reversed)
		positioner_js[:traj_length_half]=spiralize(positioner_js[:traj_length_half],positioner_js_x[:traj_length_x_half],reversed)
	else:
		rob1_js[traj_length_half:]=spiralize(rob1_js[traj_length_half:],rob1_js_x[traj_length_x_half:],reversed)
		rob2_js[traj_length_half:]=spiralize(rob2_js[traj_length_half:],rob2_js_x[traj_length_x_half:],reversed)
		positioner_js[traj_length_half:]=spiralize(positioner_js[traj_length_half:],positioner_js_x[traj_length_x_half:],reversed)
	return rob1_js,rob2_js,positioner_js
