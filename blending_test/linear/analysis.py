import sys
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *

from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

p_start=np.array([1648,-1290,-231])
p_end=np.array([1648,-1190,-231])

curve=np.linspace(p_start,p_mid)


def blending_zone_test():
	z_all=np.linspace(0,10,num=20)
	for i in range(z_all):
		data=np.loadtxt('blending_zone_test/z'+str(z_all[i])+'.csv',delimiter=',')
		curve_exe_js=data[:,1:]
		timestamp=data[:,0]
		curve_exe=robot.fwd(curve_exe_js)
		lam=calc_lam_cs(curve_exe)
		speed=np.gradient(lam)/np.gradient(timestamp)


		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='desired')
		ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='executed')

		ax.set_xlabel('$X$')
		ax.set_ylabel('$Y$')
		ax.set_zlabel('$Z$')
		plt.savefig('blending_zone_test/trajectory_plots/z'+str(z_all[i]))
		plt.clf()
		plt.show()

		plt.plot(lam,speed, c='green')
		plt.title('Speed vs. Path Length')
		plt.xlabel('lambda (mm)')
		plt.xlabel('speed (mm/s)')
		plt.savefig('blending_zone_test/speed_plots/z'+str(z_all[i]))
		plt.clf()
		plt.show()


def arc_motion_test():
	data=np.loadtxt('arc_motion_test/arc_motion_test.csv',delimiter=',')
	curve_exe_js=data[:,1:]
	timestamp=data[:,0]
	curve_exe=robot.fwd(curve_exe_js)
	lam=calc_lam_cs(curve_exe)
	speed=np.gradient(lam)/np.gradient(timestamp)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='desired')
	ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='executed')

	ax.set_xlabel('$X$')
	ax.set_ylabel('$Y$')
	ax.set_zlabel('$Z$')
	plt.savefig('arc_motion_test/trajectory_plots/z'+str(z_all[i]))
	plt.clf()
	plt.show()

	plt.plot(lam,speed, c='green')
	plt.title('Speed vs. Path Length')
	plt.xlabel('lambda (mm)')
	plt.xlabel('speed (mm/s)')
	plt.savefig('arc_motion_test/speed_plots/z'+str(z_all[i]))
	plt.clf()
	plt.show()


def dense_points_test():
	z_all=np.linspace(0,10,num=20)
	for i in range(z_all):
		data=np.loadtxt('dense_points_test/z'+str(z_all[i])+'.csv',delimiter=',')
		curve_exe_js=data[:,1:]
		timestamp=data[:,0]
		curve_exe=robot.fwd(curve_exe_js)
		lam=calc_lam_cs(curve_exe)
		speed=np.gradient(lam)/np.gradient(timestamp)


		fig = plt.figure()
		ax = plt.axes(projection='3d')
		ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='desired')
		ax.plot3D(curve_exe[:,0], curve_exe[:,1], curve_exe[:,2], c='red',label='executed')

		ax.set_xlabel('$X$')
		ax.set_ylabel('$Y$')
		ax.set_zlabel('$Z$')
		plt.savefig('dense_points_test/trajectory_plots/z'+str(z_all[i]))
		plt.clf()
		plt.show()

		plt.plot(lam,speed, c='green')
		plt.title('Speed vs. Path Length')
		plt.xlabel('lambda (mm)')
		plt.xlabel('speed (mm/s)')
		plt.savefig('dense_points_test/speed_plots/z'+str(z_all[i]))
		plt.clf()
		plt.show()