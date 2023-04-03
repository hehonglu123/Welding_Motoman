import numpy as np
import sys, traceback, time, copy, glob
import matplotlib.pyplot as plt
from matplotlib import cm
from general_robotics_toolbox import *
sys.path.append('../../../../toolbox')
from robot_def import *
from multi_robot import *
from error_check import *
from path_calc import *

def main():
	
	curve=np.loadtxt('tenth_scale_blade_gcode.csv',delimiter=',')
	print(curve.shape)

	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	


	ax.plot3D(curve[::vis_step,0],curve[::vis_step,1],curve[::vis_step,2],'r.-')
	# ax.quiver(curve[::vis_step,0],curve[::vis_step,1],curve[::vis_step,2],curve[::vis_step,0],curve[::vis_step,1],curve[::vis_step,2],length=0.3, normalize=True)

	plt.show()
if __name__ == '__main__':
	main()