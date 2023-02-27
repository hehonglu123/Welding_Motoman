from audioop import reverse
from copy import deepcopy
import sys
sys.path.append('../../toolbox/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from dx200_motion_program_exec_client import *
from MotionSendMotoman import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import numpy as np

def robot_weld_path_gen(test_n):
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
    # x0 =  1684	# Origin x coordinate
    # y0 = -753.5	# Origin y coordinate
    # z0 = -245   # 10 mm distance to base

    # x0 = 980
    # y0 = 340
    # z0 = 1015-1340

    x0 = 1684
    y0 = -753.5
    z0 = -255
    # z0 = -205
    # z0 = -305

    all_path_T = []
    for n in range(test_n):
        p_start = [x0, y0, z0]
        p_end = [x0 - 76, y0, z0]

        T_start=Transform(R,p_start)
        T_end = Transform(R,p_end)
        all_path_T.append([T_start,T_end])

        y0 = y0 + 27
    
    return all_path_T

def get_bound_circle(p,R,pc,k,theta):

    p=np.array(p)
    R=np.array(R)
    pc=np.array(pc)
    k=np.array(k)
    
    rot_R_2 = rot(k,theta/2)
    rot_R = rot(k,theta)
    p_bound = np.matmul(rot_R,(p-pc))+pc
    p_bound_2 = np.matmul(rot_R_2,(p-pc))+pc
    R_bound = np.matmul(rot_R,R)
    # R_bound = np.matmul(rot_R,R)
    R_bound_2 = np.matmul(rot_R_2,R)
    
    return [p_bound,p_bound_2],[R_bound,R_bound_2]

data_dir='../../data/wall_weld_test/scan_cont_1/'
config_dir='../../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')


# test_q = np.radians([21.3515,20.3011,-34.1789,0.0020,55.0069,-0.3364])
# test_q = np.radians([23.1762,35.0222,-34.0818,-21.0999,71.6365,7.8284])
# test_T=robot_scan.fwd(np.array([0.,0.,0.,0.,0.,0.]))
# test_T=robot_scan.fwd(test_q)
# # tool_T = Transform(robot_scan.R_tool,robot_scan.p_tool)
# print("scanner zero-config",test_T)
zero_config=np.array([0.,0.,0.,0.,0.,0.])

# test_T_1 = deepcopy(test_T)
# test_T_1.R = np.array([[1,0,0],
#                        [0,-1,0],
#                        [0,0,-1]])
# test_T_1_q = robot_scan.inv(test_T_1.p,test_T_1.R)
# print(np.degrees(test_T_1_q))
# R_down = rot([1,0,0],np.pi)
# rotate_R = np.matmul(R_down.T,test_T.R)
# k,theta = R2rot(rotate_R)
# print(np.degrees(theta))
# exit()

R2base_R1base_H = np.linalg.inv(robot_scan.base_H)
R2base_table_H = np.matmul(R2base_R1base_H,turn_table.base_H)
print("Turn table relative to R2",R2base_table_H)

### scanner hardware
c = RRN.ConnectService('rr+tcp://192.168.55.27:64238?service=scanner')
cscanner = ContinuousScanner(c)

### get welding robot path
test_n=1 # how many test
all_path_T = robot_weld_path_gen(test_n)
Rz_turn_angle = np.radians(0)
Ry_turn_angle = np.radians(-20) # rotate in y a bit, z-axis not pointing down, to have ik solution
# Ry_turn_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution

### scan parameters
scan_stand_off_d = 220 ## mm
bounds_theta = np.radians(1) ## circular motion at start and end
# all_scan_angle = np.radians([-45,0,45]) ## scanning angles
all_scan_angle = np.radians([0]) ## scanning angles

for path_T in all_path_T:

    ### path gen ###
    scan_path=[]
    robot_path=deepcopy(path_T)

    scan_p=[]
    scan_R=[]
    reverse_path_flag=False
    for scan_angle in all_scan_angle:
        
        sub_scan_p=[]
        sub_scan_R=[]
        for pT_i in range(len(robot_path)):

            this_p = robot_path[pT_i].p
            this_R = robot_path[pT_i].R

            if pT_i == len(robot_path)-1:
                this_scan_R = deepcopy(sub_scan_R[-2])
                this_scan_p = this_p - this_scan_R[:,-1]*scan_stand_off_d # stand off distance to scan
                sub_scan_p.append(this_scan_p)
                sub_scan_R.append(this_scan_R)
                k = deepcopy(this_scan_R[:,1])
                p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,-bounds_theta)
                sub_scan_p.extend(p_bound_path[::-1])
                sub_scan_R.extend(R_bound_path[::-1])
                break

            next_p = robot_path[pT_i+1].p
            next_r = robot_path[pT_i+1].R
            travel_v = (next_p-this_p)
            travel_v = travel_v/np.linalg.norm(travel_v)

            # get scan R
            Rx = travel_v
            Rz = this_R[:,-1] # assume weld is perpendicular to the plane
            Rz = (Rz-np.dot(Rx,Rz)*Rx)
            Rz = Rz/np.linalg.norm(Rz)

            ## rotate
            Rz = np.matmul(rot(travel_v,scan_angle),Rz)

            Ry = np.cross(Rz,Rx)
            Ry = Ry/np.linalg.norm(Ry)
            this_scan_R = np.array([Rx,Ry,Rz]).T
            # get scan p)
            this_scan_p = this_p - Rz*scan_stand_off_d # stand off distance to scan

            # add start bound condition
            if pT_i == 0:
                k = deepcopy(Ry)
                p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,bounds_theta)
                sub_scan_p.extend(p_bound_path)
                sub_scan_R.extend(R_bound_path)
            
            # add scan p R to path
            sub_scan_p.append(this_scan_p)
            sub_scan_R.append(this_scan_R)
        
        if reverse_path_flag:
            scan_p.extend(sub_scan_p[::-1])
            scan_R.extend(sub_scan_R[::-1])
        else:
            scan_p.extend(sub_scan_p)
            scan_R.extend(sub_scan_R)
        reverse_path_flag= not reverse_path_flag

    scan_p=np.array(scan_p)
    scan_R=np.array(scan_R)

    ## turn Rx direction (in Rz direction)
    scan_R = np.matmul(scan_R,rot([0.,0.,1.],Rz_turn_angle))
    scan_R = np.matmul(scan_R,rot([0.,1.,0.],Ry_turn_angle))
    ### change everything to appropriate frame (currently: Robot2):
    scan_p_T=[]
    scan_p = np.matmul(R2base_R1base_H[:3,:3],scan_p.T).T + R2base_R1base_H[:3,-1]
    scan_R = np.matmul(R2base_R1base_H[:3,:3],scan_R)
    # visualize_frames(scan_R,scan_p,size=10)
    ############# path gen finish ################

    ### redundancy resolution ###
    # no resolving now, only get js
    curve_js = []
    q_init=robot_scan.inv(scan_p[0],scan_R[0],zero_config)[0]
    print(np.degrees(q_init))
    curve_js.append(q_init)
    ######## need to get the true transform
    for path_i in range(len(scan_p)):
        print(scan_p[path_i][2])
        this_js = robot_scan.inv(scan_p[path_i],scan_R[path_i],curve_js[-1])[0]
        curve_js.append(this_js)
    curve_js=np.array(curve_js)
    print(np.degrees(curve_js))

    #############################
    # exit()

    ### motion program generation ###
    primitives=[]
    q_bp=[]
    p_bp=[]
    speed_bp=[]
    zone_bp=[]
    for path_i in range(1,len(scan_p)):
        primitives.append('movel')
        q_bp.append([curve_js[path_i]])
        p_bp.append(scan_p[path_i])
        speed_bp.append(40) ## 10 mm/sec
        if path_i != len(scan_p)-1:
            zone_bp.append(0)
        else:
            zone_bp.append(0)
    #################################
    input("Press Enter to start moving")

    ### execute motion ###
    # ms=MotionSend()
    ## move to start
    # ms.exec_motions(robot_scan,['movej'],[scan_p[0]],[[curve_js[0]]],2,0)
    robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    robot_client.MoveJ(np.degrees(curve_js[0]), 2, 0)
    robot_client.ProgEnd()
    robot_client.execute_motion_program("AAA.JBI")

    ## scanner start
    cscanner.start_capture()
    ## motion start
    robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
    for path_i in range(0,len(q_bp)):
        robot_client.MoveL(np.degrees(q_bp[path_i][0]), speed_bp[path_i], zone_bp[path_i])
    robot_client.ProgEnd()
    robot_stamps,curve_pulse_exe = robot_client.execute_motion_program("AAA.JBI")
    curve_js_exe=np.divide(curve_pulse_exe[:,6:12],robot_scan.pulse2deg)
    ## scanner end
    cscanner.end_capture()
    st=time.perf_counter()
    scans,scan_stamps=cscanner.get_capture()
    dt=time.perf_counter()-st
    print("dt:",dt)
    print("Robot last joint:",curve_js_exe[-1])

    ## save traj
    # save poses
    np.savetxt(data_dir + 'curve_js_exe.csv',np.radians(curve_js_exe),delimiter=',')
    np.savetxt(data_dir + 'robot_stamps.csv',robot_stamps-robot_stamps[0],delimiter=',')
    scan_count=0
    for scan in scans:
        scan_points = RRN.NamedArrayToArray(scan.vertices)
        np.save(data_dir + 'points_'+str(scan_count)+'.npy',scan_points)
        if scan_count%10==0:
            print(len(scan_points))
        scan_count+=1
    np.savetxt(data_dir + 'scan_stamps.csv',scan_stamps,delimiter=',')
