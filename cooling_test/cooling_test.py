import time
import sys
sys.path.append('../toolbox/')
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robot_def import *
from WeldSend import *
from dx200_motion_program_exec_client import *
from general_robotics_toolbox import *
from RobotRaconteur.Client import *
config_dir='../config/'
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config.yaml')

robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
ws.cooling_on(robot_weld)
# time.sleep(5)
# ws.cooling_off(robot_weld)