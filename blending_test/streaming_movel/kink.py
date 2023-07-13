import sys
sys.path.append('../../toolbox/')
from robot_def import *
from StreamingSend import *
from RobotRaconteur.Client import *
from dx200_motion_program_exec_client import *

import matplotlib.pyplot as plt
import time




def jog2start():
	client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
	client.ProgStart(r"""AAA""")
	client.MoveJ(np.degrees(q_start),1,0)
	client.ProgFinish(r"""AAA""")
	client.ProgSave(".","AAA",False)
	client.execute_motion_program("AAA.JBI")

def blending_zone_test():
	pl_all=np.arange(0,9)
	for i in range(len(pl_all)):
		jog2start()

		client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg)
		client.MoveL(np.degrees(q_mid),10,pl_all[i])
		client.MoveL(np.degrees(q_end),10,0)
		client.ProgEnd()
		timestamp, curve_exe_js=client.execute_motion_program("AAA.JBI")
		np.savetxt('blending_zone_test/pl'+str(pl_all[i])+'.csv',np.hstack((timestamp.reshape(-1, 1),curve_exe_js)),delimiter=',')


def main():
	###KINEMATICS
	robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
	pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)

	R=np.array([[-0.7071, 0.7071, -0.    ],
				[ 0.7071, 0.7071,  0.    ],
				[0.,      0.,     -1.    ]])

	displacement=400
	p_start=np.array([1648,-900,-100])
	p_mid=np.array([1648-displacement/(2*np.sqrt(3)),-900+displacement/2,-100])
	p_end=np.array([1648,-900+displacement,-100])

	qseed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
	q_start=robot.inv(p_start, R, qseed)[0]
	q_mid=robot.inv(p_mid, R, qseed)[0]
	q_end=robot.inv(p_end, R, qseed)[0]

	vd_all=[50,100,200,400,800]
	for vd in vd_all:
		client=MotionProgramExecClient()
		mp=MotionProgram(ROBOT_CHOICE='RB1',pulse2deg=robot.pulse2deg, tool_num=12)
		mp.MoveJ(np.degrees(q_start),1,None)
		client.execute_motion_program(mp)

	
		mp.MoveL(np.degrees(q_mid),vd,None)
		mp.MoveL(np.degrees(q_end),vd,None)
		timestamp_recording,joint_recording,job_line,job_step=client.execute_motion_program(mp)
		np.savetxt('movel_test/joint_recording_%i.csv'%vd,np.hstack((timestamp_recording.reshape(-1, 1),joint_recording[:,:6])),delimiter=',')

if __name__ == '__main__':
	main()