import sys

import numpy as np

sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

robot=robot_obj('MA_2010_A0',def_path='../../config/MA_2010_A0_robot_default_config.yml',tool_file_path='../../config/weldgun.csv',\
	pulse2deg_file_path='../../config/MA_2010_A0_pulse2deg.csv')

##########################################FIGURE square TEST###############################################
R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
qseed = np.radians([-35.4291, 56.6333, 40.5194, 4.5177, -52.2505, -11.6546])
x0 =  1684	# Origin x coordinate
y0 = -1179	# Origin y coordinate
# y0 = - 1273
n_top = 0 	# number of top layers
n_test = 0	# number of test times
v0 = int(input("请输入初始速度(5或17)："))
# v0 = 5		# initial travel speed
# v0 = 17		# second travel speed
q0 = robot.inv([x0,y0,100], R, qseed)[0]
q_home = np.zeros(6)
client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
client.ACTIVE_TOOL=1
client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")
client.MoveJ(q_home,8,0)
client.MoveJ(np.degrees(q0), 10, 0)
client.ProgEnd()
client.execute_motion_program("AAA.JBI")
total_test = int(input("请输入所有想要的测试数："))

while n_test < total_test:
	desired_test = int(input("请输入本轮想要的测试数："))
	h = int(input('打印第几层：'))
	select = input("请输入是否weld (True or False)：")
	job_number = int(input("请输入weld需要的job number (70-250)："))
	print('\n本轮测试数：',desired_test,'\n',
		  '是否开启welder:', select,'\n',
		  '本次测试的job number:', job_number,'\n')
	print("测试开始运行...")

	layer_base = 0
	while n_test < desired_test:
		print("第",n_test + 1, '次测试')
		print('横向起始点:', y0)
		print('测试速度:', v0)
		layer_base = 0
		z0 = -244 + h # 10 mm distance to base

		while layer_base < 1:
			########################## setup motion points ###########################
			p1 = [x0, y0 - 12, z0]
			p2 = [x0, y0 - 12, z0 - 10]
			p3 = [x0 - 76, y0 - 12 , z0 - 10]
			p4 = [x0 - 76, y0 - 12 , z0]
			############ use inverse kinematics to calculate angle joints ############
			q1 = robot.inv(p1, R, qseed)[0]
			q2 = robot.inv(p2, R, qseed)[0]
			q3 = robot.inv(p3, R, qseed)[0]
			q4 = robot.inv(p4, R, qseed)[0]
			############ motion command ############
			client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
			client.ACTIVE_TOOL=1
			client.ProgStart(r"""AAA""")
			client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

			client.MoveL(np.degrees(q1), 30, 0)
			client.MoveL(np.degrees(q2), 10, 0)
			client.SetArc(select, job_number)
			client.MoveL(np.degrees(q3), v0, 0)
			client.SetArc(False)
			client.MoveL(np.degrees(q4), 10, 0)
			print('首层高度:', z0)

			client.ProgEnd()
			client.execute_motion_program("AAA.JBI")
			layer_base = layer_base + 1
		n_test = n_test + 1
		y0 = y0 - 27
		v0 = v0 + 5
	print("本轮测试结束...")
	print("共测试",n_test, '次')
	if n_test == total_test:
		break
	answer = input("测试是否继续 (Y/N):")
	if answer.upper() == 'N':
		break
client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
client.ACTIVE_TOOL=1
client.ProgStart(r"""AAA""")
client.setFrame(Pose([0,0,0,0,0,0]),-1,r"""Motoman MA2010 Base""")

client.MoveJ(np.degrees(q0), 10, 0)
client.MoveJ(q_home,8,0)
client.ProgEnd()
client.execute_motion_program("AAA.JBI")



