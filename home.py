import sys
sys.path.append('toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

config_dir='config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv')


q1=np.array([0,0,0,0,0,0])
q2=np.array([90,0,0,0,0,0])
# q2=np.array([0,37,-9,0,0,0])
# q2_1=np.array([0,36,-8,0,0,0])
q3=[-15,0]

robot_client=MotionProgramExecClient()

mp = MotionProgram(ROBOT_CHOICE='RB1', pulse2deg=robot_weld.pulse2deg)
mp.MoveJ(q1,5,0)
robot_stamps,curve_exe, job_line,job_step = robot_client.execute_motion_program(mp)
print("Test read joints:",np.degrees(curve_exe))

mp = MotionProgram(ROBOT_CHOICE='RB2', pulse2deg=robot_scan.pulse2deg)
mp.MoveJ(q2,5,0)
robot_client.execute_motion_program(mp)

mp = MotionProgram(ROBOT_CHOICE='ST1', pulse2deg=positioner.pulse2deg)
mp.MoveJ(q3,90,0)
robot_client.execute_motion_program(mp)
