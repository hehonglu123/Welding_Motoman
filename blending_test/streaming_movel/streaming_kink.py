import sys
sys.path.append('../../toolbox/')
from robot_def import *
from StreamingSend import *
from RobotRaconteur.Client import *


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
	###RR Robot Connection
	RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
	RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
	RR_robot = RR_robot_sub.GetDefaultClientWait(1)
	robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
	halt_mode = robot_const["RobotCommandMode"]["halt"]
	position_mode = robot_const["RobotCommandMode"]["position_command"]
	RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
	RR_robot.reset_errors()
	RR_robot.enable()
	RR_robot.command_mode = halt_mode
	time.sleep(0.1)
	RR_robot.command_mode = position_mode
	SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate=125.)

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
		SS.jog2q(np.hstack((q_start,[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
		lam_totoal=np.linalg.norm(p_start-p_mid)+np.linalg.norm(p_mid-p_end)
		num_points=int((np.linalg.norm(p_start-p_mid)+np.linalg.norm(p_mid-p_end))*125/vd)
		d_lam=lam_totoal/num_points
		lam=np.linspace(0,lam_totoal,num_points)
		q_all=[]
		p_all=[]
		for i in range(num_points):
			if lam[i]<np.linalg.norm(p_start-p_mid):
				p_all.append(p_start+(p_mid-p_start)*(lam[i]/np.linalg.norm(p_start-p_mid)))
				q_all.append(robot.inv(p_all[-1],R,qseed)[0])
			else:
				p_all.append(p_mid+(p_end-p_mid)*(lam[i]-np.linalg.norm(p_start-p_mid))/np.linalg.norm(p_mid-p_end))
				q_all.append(robot.inv(p_all[-1],R,qseed)[0])
		
		q_all=np.array(q_all)
		timestamp_recording,joint_recording=SS.traj_streaming(q_all,ctrl_joints=np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,0]))
		np.savetxt('streaming_test/wofronius/joint_recording_%i.csv'%vd,np.hstack((timestamp_recording.reshape(-1, 1),joint_recording)),delimiter=',')

if __name__ == '__main__':
	main()