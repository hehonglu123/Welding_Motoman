from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
from threading import Thread
import numpy as np
import time
import yaml
from StreamingSend import *

def main():

    ############## Robot definition ##############
    config_dir='../../config/'
    robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    ########################################################RR STREAMING########################################################
    RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
    point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
    SS=StreamingSend(RR_robot_sub,streaming_rate=125.)

    ######################################### RR Fujicam ########################################################
    fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
    def connect_failed(s, client_id, url, err):
        print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
    sub=RRN.SubscribeService(fujicam_url)
    obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
    fuji_scan_wire=sub.SubscribeWire("lineProfile")
    sub.ClientConnectFailed += connect_failed

    ### get robot starting angle
    starting_q = deepcopy(SS.q_cur)
    T_tcp_1 = robot_1.fwd(starting_q[:6])
    T_tcp_2 = robot_2.fwd(starting_q[6:12],world=True)
    T_tcp1_tcp2 = T_tcp_2.inv() * T_tcp_1