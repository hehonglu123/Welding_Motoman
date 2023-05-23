import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *
from general_robotics_toolbox import *

from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

data_dir='test2/'
config_dir='../../config/'

scan_resolution=5 #scan every 5 mm
scan_per_pose=3 # take 3 scan every pose

robot=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')

cart_p=[]
joints_p=np.loadtxt(data_dir+'scan_js.csv',delimiter=",", dtype=np.float64)
for i in range(len(joints_p)):
	joints_p[i][5] = -joints_p[i][5]
	joints_p[i][3] = -joints_p[i][3]
	joints_p[i][1] = 90 - joints_p[i][1]
	joints_p[i][2] = joints_p[i][2] + joints_p[i][1]

joints_p=np.radians(joints_p)
for q in joints_p:
	cart_p.append(robot.fwd(q))
print("Joint Space")
print(np.degrees(joints_p))
print("Cart Space")
print(cart_p)

curve=[]
curve_R=[]
curve_js=[]
total_step=0
for i in range(len(cart_p)-1):
	travel_vec=cart_p[i+1].p-cart_p[i].p	
	travel_dis=np.linalg.norm(travel_vec)
	travel_vec=travel_vec/travel_dis*scan_resolution
	print("Travel Vector:",travel_vec)
	print("Travel Distance:",travel_dis)

	xp=np.append(np.arange(cart_p[i].p[0],cart_p[i+1].p[0],travel_vec[0]),cart_p[i+1].p[0])
	yp=np.append(np.arange(cart_p[i].p[1],cart_p[i+1].p[1],travel_vec[1]),cart_p[i+1].p[1])
	zp=np.append(np.arange(cart_p[i].p[2],cart_p[i+1].p[2],travel_vec[2]),cart_p[i+1].p[2])
	print(len(xp))
	print(len(yp))
	print(len(zp))

	for travel_i in range(len(xp)):
		this_p=Transform(cart_p[i].R,[xp[travel_i],yp[travel_i],zp[travel_i]])
		curve.append(this_p.p)
		curve_R.append(this_p.R)
		if len(curve_js)!=0:
			curve_js.append(robot.inv(this_p.p,this_p.R,curve_js[-1])[0])
		else:
			curve_js.append(robot.inv(this_p.p,this_p.R,joints_p[0])[0])
	total_step+=len(xp)
curve=np.array(curve)
curve_js=np.array(curve_js)
print(curve)
print(np.degrees(curve_js))
print("Total step:",total_step)


ms=MotionSend()

move_robot_only=False

if not move_robot_only:
	client=RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')

js_pose=[]
for i in range(len(curve)):
	# ms.exec_motions(robot,['movej'],[curve[i]],[[curve_js[i]]],5,0)
	timestamp, curve_exe_js=ms.exec_motions(robot,['movel'],[curve[i]],[[curve_js[i]]],5,0)
	js_pose.append(curve_exe_js[-1])
	time.sleep(0.3) # stop 1 sec

	if not move_robot_only:
		for scan_i in range(scan_per_pose):
			mesh=client.capture(True)
			scan_points = RRN.NamedArrayToArray(mesh.vertices)
			np.savetxt(data_dir + 'points_'+str(i)+'_'+str(scan_i)+'.csv',scan_points,delimiter=',')
			print("Pose:",i,",Scan:",scan_i,",points:",len(scan_points))
js_pose=np.array(js_pose)
np.savetxt(data_dir + 'curve_js_exe.csv',js_pose,delimiter=',')