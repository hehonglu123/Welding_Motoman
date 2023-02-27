import sys
sys.path.append('toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

config_dir='config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

station_pulse2deg=np.abs(np.loadtxt('config/D500B_pulse2deg.csv'))

q1=np.array([0,0,0,0,0,0])
q2=np.array([90,0,0,0,0,0])
q3=[-15,180]
	
client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
client.MoveJ(q1,2,0)
client.ProgEnd()
r_timestamps, curve_js_exe = client.execute_motion_program("AAA.JBI")
print(r_timestamps[-1]-r_timestamps[0])
print(np.divide(np.array(curve_js_exe[-1,0:6]),robot_weld.pulse2deg))

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
client.MoveJ(q2,2,0)
client.ProgEnd()
r_timestamps, curve_js_exe = client.execute_motion_program("AAA.JBI")
print(r_timestamps[-1]-r_timestamps[0])
print(np.divide(np.array(curve_js_exe[-1,6:12]),robot_scan.pulse2deg))

# client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='ST1',pulse2deg=turn_table.pulse2deg)
# client.MoveJ(q3,2,0)
# client.ProgEnd()
# client.execute_motion_program("AAA.JBI")

