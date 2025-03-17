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
from PH_interp import *
from StreamingSend import *

def main():

    #### motino parameters ####
    r2_inward = np.radians([0,10,20,30,40,50])
    r2_outward = np.array([0,-10,-20,-30,-40,-50])
    motion_points = 10000

    ############## Robot definition ##############
    config_dir='../../config/'
    robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_2=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    ### Nominal PH
    nom_P_r1=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H_r1=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
    nom_P_r2=np.array([[0,0,0],[155,0,0],[0,0,614],\
                   [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H_r2=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
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

    ####### PH Parameters #######
    calib_file_name = 'calib_PH_q_ana.pickle'
    PH_r1_data_dir='PH_grad_data/test0801_R1/train_data_'
    PH_r2_data_dir='PH_grad_data/test0804_R2/train_data_'
    with open(PH_r1_data_dir+calib_file_name,'rb') as file:
        PH_q_r1=pickle.load(file)
    ph_param_fbf_r1=PH_Param(nom_P_r1,nom_H_r1)
    ph_param_fbf_r1.fit(PH_q_r1,method='FBF')
    with open(PH_r2_data_dir+calib_file_name,'rb') as file:
        PH_q_r2=pickle.load(file)
    ph_param_fbf_r2=PH_Param(nom_P_r2,nom_H_r2)
    ph_param_fbf_r2.fit(PH_q_r2,method='FBF')

    ### get robot starting angle
    starting_q = deepcopy(SS.q_cur)
    r1_starting = starting_q[0:6]
    r2_starting = starting_q[6:12]
    T_tcp_1 = robot_1.fwd_ph(r1_starting,ph_param_fbf_r1)
    T_tcp_2 = robot_2.fwd_ph(r2_starting,ph_param_fbf_r2,world=True)
    T_tcp1_tcp2 = T_tcp_2.inv() * T_tcp_1

    ### get robot 2 joint path
    r2_inward_path = np.linspace(r2_starting,r2_inward,motion_points)
    r2_outward_path = np.linspace(r2_inward_path[-1],r2_outward,motion_points*2)

    ### get robot 1 joint path
    r1_inward_path = [r1_starting]
    for r2_wp in r2_inward_path:
        r1_tcp = robot_2.fwd_ph(r2_wp,ph_param_fbf_r2,world=True)*T_tcp1_tcp2
        r1_wp = robot_1.inv_iter(r1_tcp.p,r1_tcp.R,q_seed=r1_inward_path[-1])
        r1_inward_path.append(r1_wp)
    r1_outward_path = [r1_inward_path[-1]]
    for r2_wp in r2_outward_path:
        r1_tcp = robot_2.fwd_ph(r2_wp,ph_param_fbf_r2,world=True)*T_tcp1_tcp2
        r1_wp = robot_1.inv_iter(r1_tcp.p,r1_tcp.R,q_seed=r1_outward_path[-1])
        r1_outward_path.append(r1_wp)
    

    ### jog the robot
    q_cur = deepcopy(SS.q_cur)
    q_cmd = deepcopy(q_cur)
    q_cmd[0:6] = r1_inward_path[0]
    q_cmd[6:12] = r2_inward_path[0]
    q_table = q_cur[12:]
    SS.jog2q(q_cmd)
    time.sleep(2)

    print('Start moving')
    for r1_wp,r2_wp in zip(r1_inward_path,r2_inward_path):
        loop_start=time.perf_counter()
        q_cmd = np.hstack((r1_wp,r2_wp,q_table))
        SS.position_cmd(q_cmd,loop_start)