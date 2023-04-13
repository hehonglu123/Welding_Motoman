from dx200_motion_program_exec_client import *
from robot_def import * 

config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv'\
,base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

client=MotionProgramExecClient(IP='192.168.1.31',ROBOT_CHOICE='RB1',pulse2deg=robot_weld.pulse2deg)
client.setWaitTime(0.1)
client.ProgEnd()
robot_stamps,curve_js_exe = client.execute_motion_program("AAA.JBI")

print(np.degrees(curve_js_exe[-1][:6]))
npstr = 'np.array(['
for i in range(6):
    npstr += str(np.degrees(curve_js_exe[-1][i]))
    if i!=5:
        npstr += ','
npstr += '])'
print(npstr)
