import sys
sys.path.append('../../toolbox/')
from robot_def import *
from path_calc import calc_lam_cs

import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

p_start=np.array([1590,-1290,-231])
p_end=np.array([1590,-1190,-231])

curve=np.linspace(p_start,p_end)


def arc_motion_test():
	data=np.loadtxt('weldchange.csv',delimiter=',')
	curve_exe_js=np.radians(data[:,1:7]/np.tile(robot.pulse2deg,(len(data),1)))
	print(data[:,1:7])
	timestamp=data[:,0]
	curve_exe=robot.fwd(curve_exe_js).p_all
	lam=calc_lam_cs(curve_exe)
	speed=np.gradient(lam)/np.gradient(timestamp)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='desired')
	ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='executed')

	ax.set_xlabel('$X$')
	ax.set_ylabel('$Y$')
	ax.set_zlabel('$Z$')
	ax.axes.set_xlim3d(left=1589.8, right=1590.2) 
	ax.axes.set_ylim3d(bottom=-1290, top=-1190) 
	ax.axes.set_zlim3d(bottom=-232, top=-230) 
	ax.legend()
	plt.title('3D Plots')
	plt.show()

	plt.plot(lam,speed, c='green')
	plt.title('Speed vs. Path Length')
	plt.xlabel('lambda (mm)')
	plt.ylabel('speed (mm/s)')
	plt.ylim(0,22)
	# plt.savefig('arc_motion_test/speed_plots/pl2_off')
	# plt.clf()
	plt.show()


if __name__ == '__main__':
	arc_motion_test()