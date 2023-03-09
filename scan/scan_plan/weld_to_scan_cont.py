from audioop import reverse
from copy import deepcopy
from pathlib import Path
import sys
sys.path.append('../../toolbox/')
sys.path.append('../../redundancy_resolution/')
sys.path.append('../scan_tools/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from redundancy_resolution_scanner import *
from tesseract_env import *
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

    x0 = 1684
    y0 = -753.5
    z0 = -255
    # z0 = -205
    # z0 = -305

    ## tune
    x0 = 1649
    y0 = -745.1
    z0 = -255

    all_path_T = []
    for n in range(test_n):
        p_start = [x0, y0, z0]
        p_end = [x0 - 76, y0, z0]

        T_start=Transform(R,p_start)
        T_end = Transform(R,p_end)
        all_path_T.append([np.append(T_start.p,T_start.R[:,-1]),np.append(T_end.p,T_end.R[:,-1])])

        y0 = y0 + 27
    
    return all_path_T

def robot_weld_path_gen_2():
    R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
    # x0 =  1684	# Origin x coordinate
    # y0 = -753.5	# Origin y coordinate
    # z0 = -245   # 10 mm distance to base

    # weld_p = np.array([[1651, -771, -255],[1651, -856, -255],[1651, -856, -254],
    # [1651, -771, -254],[1651, -771, -253],[1651, -856, -253],
    # [1651, -856, -252],[1651, -771, -252],[1651, -771, -251],
    # [1651, -856, -251],[1651, -856, -250],[1651, -771, -250],
    # [1651, -771, -249],[1651, -856, -249],[1651, -856, -248],
    # [1651, -771, -248],[1651, -771, -247],[1651, -856, -247],
    # [1651, -856, -246],[1651, -771, -246],[1651, -771, -245],
    # [1651, -856, -245],[1651, -856, -244],[1651, -771, -244]])
    weld_p = np.array([[1651, -771, -255],[1651, -856, -255]])

    ## tune
    dx = 0
    dy = 0
    dz = 0
    dp = np.array([dx,dy,dz])

    path_T=[]
    for p in weld_p:
        path_T.append(Transform(R,p+dp))

    all_path_T = [path_T]
    
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

config_dir='../../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
robot_scan_notool=robot_obj('MA_1440_A0_notool',def_path=config_dir+'MA1440_A0_robot_default_config.yml',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
    pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

## sim or real robot
sim=True

zero_config=np.array([0.,0.,0.,0.,0.,0.])
# print(robot_scan.fwd(zero_config))
# print(robot_scan_notool.fwd(zero_config))
# toolT=(Transform(np.array([[0,0,1],[0,-1,0],[1,0,0]]).T,[0,0,0])*robot_scan_notool.fwd(zero_config)).inv()*robot_scan.fwd(zero_config)
# print(toolT.p)
# print(R2rpy(toolT.R))
# exit()
# t=Tess_Env('../../config/urdf/combined')
# t.viewer_trajectory_dual(robot_scan.robot_name,turn_table.robot_name,[zero_config],[zero_config])
# input("Press enter to quit")

R2base_R1base_H = np.linalg.inv(robot_scan.base_H)
TBase_R1Base_H = np.linalg.inv(turn_table.base_H)
Table_home_T = turn_table.fwd(np.radians([-15,180]))
TBase_R1TCP_H = np.linalg.inv(np.matmul(turn_table.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
R2base_table_H = np.matmul(R2base_R1base_H,turn_table.base_H)
print("Turn table relative to R2",R2base_table_H)

### get welding robot path
## welding wall test
# data_dir='../../data/wall_weld_test/scan_cont_3/'
# test_n=1 # how many test
# all_path_T = robot_weld_path_gen(test_n)
# curve_sliced_R1=np.array(all_path_T[0])
# curve_sliced_relative=[]
# for path_p in curve_sliced_R1:
#     this_p = np.matmul(TBase_R1Base_H[:3,:3],path_p[:3])+TBase_R1Base_H[:3,-1]
#     this_n = np.matmul(TBase_R1Base_H[:3,:3],path_p[3:])
#     curve_sliced_relative.append(np.append(this_p,this_n))
# curve_sliced_relative=np.array(curve_sliced_relative)
## wall test 2
data_dir='../../data/wall_weld_test/scan_cont_3/'
all_path_T = robot_weld_path_gen_2()
curve_sliced_R1=np.array(all_path_T[0])
curve_sliced_relative=[]
for path_p in curve_sliced_R1:
    this_p = np.matmul(TBase_R1TCP_H[:3,:3],path_p.p)+TBase_R1TCP_H[:3,-1]
    this_n = np.matmul(TBase_R1TCP_H[:3,:3],path_p.R[:,-1])
    curve_sliced_relative.append(np.append(this_p,this_n))
curve_sliced_relative=np.array(curve_sliced_relative)
## blade
# dataset='blade0.1/'
# sliced_alg='NX_slice2/'
# data_dir='../../data/'+dataset+sliced_alg
# out_dir=data_dir+'scans/'
# curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice0.csv',delimiter=',')
##############################

### scan parameters
scan_speed=30 # scanning speed
Rz_turn_angle = np.radians(0) # point direction w.r.t welds
Ry_turn_angle = np.radians(-10) # rotate in y a bit, z-axis not pointing down, to have ik solution
# Ry_turn_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution
scan_stand_off_d = 243 ## mm
bounds_theta = np.radians(45) ## circular motion at start and end
# all_scan_angle = np.radians([-45,0,45,0]) ## scanning angles
all_scan_angle = np.radians([0]) ## scanning angles

### path gen ###
scan_path=[]

scan_p=[]
scan_R=[]
reverse_path_flag=False
for scan_angle in all_scan_angle:
    
    sub_scan_p=[]
    sub_scan_R=[]
    for pT_i in range(len(curve_sliced_relative)):

        this_p = curve_sliced_relative[pT_i][:3]
        this_n = curve_sliced_relative[pT_i][3:]

        if pT_i == len(curve_sliced_relative)-1:
            this_scan_R = deepcopy(sub_scan_R[-1])
            this_scan_p = this_p + this_scan_R[:,-1]*scan_stand_off_d # stand off distance to scan
            sub_scan_p.append(this_scan_p)
            sub_scan_R.append(this_scan_R)
            k = deepcopy(this_scan_R[:,0])
            p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,-bounds_theta)
            sub_scan_p.extend(p_bound_path[::-1])
            sub_scan_R.extend(R_bound_path[::-1])
            break

        next_p = curve_sliced_relative[pT_i+1][:3]
        travel_v = (next_p-this_p)
        travel_v = travel_v/np.linalg.norm(travel_v)

        # get scan R
        Rz = -deepcopy(this_n) # assume weld is perpendicular to the plane
        Rz = Rz/np.linalg.norm(Rz)
        Ry = travel_v
        Ry = (Ry-np.dot(Ry,Rz)*Rz)
        # rotate z-axis around y-axis
        Rz = np.matmul(rot(Ry,scan_angle),Rz)

        Rx = np.cross(Ry,Rz)
        Rx = Rx/np.linalg.norm(Rx)
        this_scan_R = np.array([Rx,Ry,Rz]).T
        # get scan p)
        this_scan_p = this_p + Rz*scan_stand_off_d # stand off distance to scan

        # add start bound condition
        if pT_i == 0:
            k = deepcopy(Rx)
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

####### add detail path ################
delta_p = 0.1
scan_p_detail=None
scan_R_detail=None
for scan_p_i in range(len(scan_p)-1):
    this_scan_p=deepcopy(scan_p[scan_p_i])
    next_scan_p=deepcopy(scan_p[scan_p_i+1])

    if np.all(this_scan_p==next_scan_p):
        continue

    travel_v=next_scan_p-this_scan_p
    total_l=np.linalg.norm(travel_v)
    travel_v=travel_v/total_l
    travel_v=travel_v*delta_p

    scan_p_mid=[]
    this_scan_p_mid=deepcopy(this_scan_p)
    while True:
        scan_p_mid.append(deepcopy(this_scan_p_mid))
        this_scan_p_mid+=travel_v
        if np.linalg.norm(this_scan_p_mid-this_scan_p)>=total_l:
            break
    scan_p_mid=np.array(scan_p_mid)
    scan_R_mid = np.tile(scan_R[scan_p_i],(len(scan_p_mid),1,1))
    if scan_p_detail is None:
        scan_p_detail=deepcopy(scan_p_mid)
        scan_R_detail=deepcopy(scan_R_mid)
    else:
        scan_p_detail = np.vstack((scan_p_detail,scan_p_mid))
        scan_R_detail = np.vstack((scan_R_detail,scan_R_mid))
########################################
scan_p_detail=np.vstack((scan_p_detail,[scan_p[-1]]))
scan_R_detail=np.vstack((scan_R_detail,[scan_R[-1]]))
scan_p=deepcopy(scan_p_detail)
scan_R=deepcopy(scan_R_detail)

####### add extension for time sync ####

########################################

# visualize_frames(scan_R[::10],scan_p[::10],size=1)
############# path gen finish ################

############ redundancy resolution ###
method=1
if method==0:
    ### Method1: no resolving now, only get js, and no turntable
    # change everything to appropriate frame (currently: Robot2):
    scan_p_T=[]
    scan_p = np.matmul(R2base_table_H[:3,:3],scan_p.T).T + R2base_table_H[:3,-1]
    scan_R = np.matmul(R2base_table_H[:3,:3],scan_R)
    curve_js = []
    # ik
    q_init=robot_scan.inv(scan_p[0],scan_R[0],zero_config)[0]
    curve_js.append(q_init)
    for path_i in range(len(scan_p)):
        this_js = robot_scan.inv(scan_p[path_i],scan_R[path_i],curve_js[-1])[0]
        curve_js.append(this_js)
    curve_js=np.array(curve_js)
elif method==1:
    ### Method 2: stepwise qp, turntable involved
    rrs = redundancy_resolution_scanner(robot_scan,turn_table,scan_p,scan_R)
    q_init_table=np.radians([60,-60])
    pose_R2_table=Transform(R2base_table_H[:3,:3],R2base_table_H[:3,-1])*turn_table.fwd(q_init_table)
    print(np.matmul(pose_R2_table.R,scan_p[0])+pose_R2_table.p,np.matmul(pose_R2_table.R,scan_R[0]))
    q_init_robot = robot_scan.inv(np.matmul(pose_R2_table.R,scan_p[0])+pose_R2_table.p,np.matmul(pose_R2_table.R,scan_R[0]),zero_config)[0]
    print(np.degrees(q_init_robot))
    # q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt(q_init_robot,q_init_table,w2=0.03)
    q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt_Rz(q_init_robot,q_init_table,w2=0.03)

    scan_act_R=[]
    scan_act_p=[]
    for i in range(len(q_out1)):
        poset1_1=robot_scan.fwd(q_out1[i])
        poset2_2=turn_table.fwd(q_out2[i])
        poset2_1=Transform(R2base_table_H[:3,:3],R2base_table_H[:3,-1])*poset2_2
        pose1_t2=poset2_1.inv()
        poset1_t2=pose1_t2*poset1_1
        scan_act_R.append(poset1_t2.R)
        scan_act_p.append(poset1_t2.p)
    scan_act_R=np.array(scan_act_R)
    scan_act_p=np.array(scan_act_p)

    scan_R_show=scan_R[::50]
    scan_p_show=scan_p[::50]

    # scan_R_show=np.vstack((scan_R_show,scan_act_R[::10]))
    # scan_p_show=np.vstack((scan_p_show,scan_act_p[::10]))

    visualize_frames(scan_R_show,scan_p_show,size=5)

if sim:
    t=Tess_Env('../../config/urdf/combined')				#create obj
    # for i in range(num_layers):
    # 	t.viewer_trajectory_dual(robot.robot_name,positioner.robot_name,curve_sliced_js[i][::20],positioner_js[i][::20])
    # 	time.sleep(10)
    t.viewer_trajectory_dual(robot_scan.robot_name,turn_table.robot_name,q_out1[::10],q_out2[::10])
    input("Press enter to quit")
    exit()

#############################

### motion program generation ###
primitives=[]
q_bp=[]
p_bp=[]
speed_bp=[]
zone_bp=[]
for path_i in range(0,len(scan_p)):
    primitives.append('movel')
    q_bp.append([curve_js[path_i]])
    p_bp.append(scan_p[path_i])
    speed_bp.append(scan_speed) ## mm/sec
    if path_i != len(scan_p)-1:
        zone_bp.append(0)
    else:
        zone_bp.append(0)
#################################

print(robot_scan.fwd(curve_js[0]))
print(robot_scan.fwd(np.radians([27.6768,28.7206,-39.6049,-28.0847,78.0972,10.1109])))

input("Press Enter to start moving")

### scanner hardware
c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')
cscanner = ContinuousScanner(c)

### execute motion ###
# ms=MotionSend()
## move to start
# ms.exec_motions(robot_scan,['movej'],[scan_p[0]],[[curve_js[0]]],2,0)
robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
robot_client.MoveJ(np.degrees(q_bp[0][0]), 2, 0)
robot_client.ProgEnd()
robot_client.execute_motion_program("AAA.JBI")

## scanner start
cscanner.start_capture()
## motion start
robot_client=MotionProgramExecClient(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
for path_i in range(1,len(q_bp)-1):
    robot_client.MoveL(np.degrees(q_bp[path_i][0]), speed_bp[path_i])
robot_client.MoveL(np.degrees(q_bp[-1][0]), speed_bp[-1], 0)
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
Path(out_dir).mkdir(exist_ok=True)
# save poses
np.savetxt(out_dir + 'scan_js_exe.csv',np.radians(curve_js_exe),delimiter=',')
np.savetxt(out_dir + 'robot_stamps.csv',robot_stamps-robot_stamps[0],delimiter=',')
scan_count=0
for scan in scans:
    scan_points = RRN.NamedArrayToArray(scan.vertices)
    np.save(out_dir + 'points_'+str(scan_count)+'.npy',scan_points)
    if scan_count%10==0:
        print(len(scan_points))
    scan_count+=1
print('Total scans:',scan_count)
np.savetxt(out_dir + 'scan_stamps.csv',scan_stamps,delimiter=',')
