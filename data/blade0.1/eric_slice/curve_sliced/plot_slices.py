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
	vis_step=5
	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	


	
	for i in range(1,90):
		print(i)
		slice_i=np.loadtxt(str(i)+'.csv',delimiter=',')
		ax.plot3D(slice_i[::vis_step,0],slice_i[::vis_step,1],slice_i[::vis_step,2],'r.-')

	
	# ax.quiver(curve[::vis_step,0],curve[::vis_step,1],curve[::vis_step,2],curve[::vis_step,0],curve[::vis_step,1],curve[::vis_step,2],length=0.3, normalize=True)

	plt.show()
if __name__ == '__main__':
	main()