import sys
sys.path.append('../../toolbox/')
from robot_def import *
from path_calc import calc_lam_cs

import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

p_start=np.array([1650,0,-231])
p_end=np.array([1550,0,-231])

curve=np.linspace(p_start,p_end)


def blending_zone_test():
	pl_all=np.arange(0,9)
	for i in range(len(pl_all)):
		data=np.loadtxt('blending_zone_test/pl'+str(pl_all[i])+'.csv',delimiter=',')
		curve_exe_js=np.radians(data[:,1:])
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
		ax.axes.set_xlim3d(left=1550, right=1650) 
		ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
		ax.axes.set_zlim3d(bottom=-232, top=-230) 
		ax.legend()
		plt.title('3D Plots @ PL='+str(pl_all[i]))
		plt.savefig('blending_zone_test/trajectory_plots/pl'+str(pl_all[i]))
		plt.clf()

		plt.plot(lam,speed, c='green')
		plt.title('Speed vs. Path Length @ PL='+str(pl_all[i]))
		plt.xlabel('lambda (mm)')
		plt.ylabel('speed (mm/s)')
		plt.ylim(0,22)
		plt.savefig('blending_zone_test/speed_plots/pl'+str(pl_all[i]))
		plt.clf()


def arc_motion_test():
	data=np.loadtxt('arc_motion_test/arc_motion_test_off.csv',delimiter=',')
	curve_exe_js=np.radians(data[:,1:])
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
	ax.axes.set_xlim3d(left=1550, right=1650) 
	ax.axes.set_ylim3d(bottom=-0.5, top=0.5) 
	ax.axes.set_zlim3d(bottom=-232, top=-230) 
	ax.legend()
	plt.title('3D Plots @ PL=2')
	plt.savefig('arc_motion_test/trajectory_plots/pl2_off')
	plt.clf()

	plt.plot(lam,speed, c='green')
	plt.title('Speed vs. Path Length @ PL=2')
	plt.xlabel('lambda (mm)')
	plt.ylabel('speed (mm/s)')
	plt.ylim(0,22)
	plt.savefig('arc_motion_test/speed_plots/pl2_off')
	plt.clf()


def dense_points_test():
	pl_all=np.arange(0,9)
	for i in range(len(pl_all)):
		data=np.loadtxt('dense_points_test/pl'+str(pl_all[i])+'.csv',delimiter=',')
		curve_exe_js=np.radians(data[:,1:])
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
		ax.axes.set_xlim3d(left=1647.8, right=1648.2) 
		ax.axes.set_ylim3d(bottom=-1280, top=-1200) 
		ax.axes.set_zlim3d(bottom=-232, top=-230) 
		ax.legend()
		plt.title('3D Plots @ PL='+str(pl_all[i]))
		plt.savefig('dense_points_test/trajectory_plots/pl'+str(pl_all[i]))
		plt.clf()

		plt.plot(lam,speed, c='green')
		plt.title('Speed vs. Path Length @ PL='+str(pl_all[i]))
		plt.xlabel('lambda (mm)')
		plt.ylabel('speed (mm/s)')
		plt.ylim(0,22)
		plt.savefig('dense_points_test/speed_plots/pl'+str(pl_all[i]))
		plt.clf()


if __name__ == '__main__':
	blending_zone_test()