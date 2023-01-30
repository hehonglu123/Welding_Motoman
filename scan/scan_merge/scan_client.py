import sys
sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

from RobotRaconteur.Client import *
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

data_dir='test1/'
config_dir='../../config/'

robot=robot_obj('MA_1440_A0',def_path=config_dir+'MA_1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	pulse2deg_file_path=config_dir+'MA_1440_A0_pulse2deg.csv')

cart_p=[]
joints_p=np.loadtxt(data_dir+'scan_js.csv',delimiter=",", dtype=np.float64)
joints_p=np.radians(joints_p)
for q in joints_p:
	cart_p.append(robot.fwd(q))
print("Joint Space")
print(np.degrees(joints_p))
print("Cart Space")
print(cart_p)

ms=MotionSend()

move_robot_only=True

if not move_robot_only:
	client=RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')

for i in range(len(cart_p)):
	ms.exec_motions(robot,['movej'],[cart_p[i].p],[[joints_p[i]]],10,0)
	time.sleep(1) # stop 1 sec

	if not move_robot_only:
		mesh=client.capture(True)
		scan_points = RRN.NamedArrayToArray(mesh.vertices)
		np.savetxt('points_'+str(i)+'.csv',scan_points,delimiter=',')
		print("Scan",i,", points:",len(scan_points))