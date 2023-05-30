from audioop import reverse
from copy import deepcopy
from pathlib import Path
import sys
sys.path.append('../../toolbox/')
sys.path.append('../../redundancy_resolution/')
sys.path.append('../scan_tools/')
from robot_def import *
from multi_robot import *
from scan_utils import *
from scan_continuous import *
from redundancy_resolution_scanner import *
from tesseract_env import *
from dx200_motion_program_exec_client import *
# from MotionSendMotoman import *

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
    # dx = 20
    # dy = 60
    # dz = -15
    dx = 0
    dy = 30
    dz = 0
    dp = np.array([dx,dy,dz])

    path_T=[]
    for p in weld_p:
        path_T.append(Transform(R,p+dp))

    all_path_T = [path_T]
    
    return all_path_T

def get_bound_circle(p,R,pc,k,theta):

    if theta==0:
        return [p,p],[R,R]

    p=np.array(p)
    R=np.array(R)
    pc=np.array(pc)
    k=np.array(k)
    
    all_p_bound=[]
    all_R_bound=[]
    for th in np.arange(0,theta,np.sign(theta)*np.radians(1)):
        rot_R = rot(k,th)
        p_bound = np.matmul(rot_R,(p-pc))+pc
        R_bound = np.matmul(rot_R,R)
        all_p_bound.append(p_bound)
        all_R_bound.append(R_bound)
    rot_R = rot(k,theta)
    all_p_bound.append(np.matmul(rot_R,(p-pc))+pc)
    all_R_bound.append(np.matmul(rot_R,R))
    all_p_bound=all_p_bound[::-1]
    all_R_bound=all_R_bound[::-1]
    
    return all_p_bound,all_R_bound

def get_connect_circle(start_p,start_R,end_p,end_R,pc):

    k = np.cross((end_p-start_p),start_R[:,-1])
    k = k/np.linalg.norm(k)

    theta = -2*np.arcsin(np.linalg.norm(end_p-start_p)/2/scan_stand_off_d)
    print(np.degrees(theta))

    dR = np.matmul(start_R.T,end_R)
    dRk,dRtheta = R2rot(dR)
    
    trans_theta = np.arange(0,theta,np.sign(theta)*np.radians(1))
    rot_theta = np.linspace(0,dRtheta,len(trans_theta)+1)

    all_p_bound=[]
    all_R_bound=[]
    for i in range(len(trans_theta)):
        rot_R = rot(k,trans_theta[i])
        p_bound = np.matmul(rot_R,(start_p-pc))+pc
        R_bound = np.matmul(start_R,rot(dRk,rot_theta[i]))
        all_p_bound.append(p_bound)
        all_R_bound.append(R_bound)
    rot_R = rot(k,theta)
    p_bound = np.matmul(rot_R,(start_p-pc))+pc
    R_bound = np.matmul(start_R,rot(dRk,rot_theta[-1]))
    all_p_bound.append(p_bound)
    all_R_bound.append(R_bound)

    return all_p_bound,all_R_bound

config_dir='../../config/'

robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'weldgun_old.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
# robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'weldgun_old.csv',d=15,\
# 	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg.csv')
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv')
turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv')

## sim or real robot
sim=False

zero_config=np.array([0.,0.,0.,0.,0.,0.])

R2base_R1base_H = np.linalg.inv(robot_scan.base_H)
# R2base_R1base_H[1,3] = R2base_R1base_H[1,3]-100
TBase_R1Base_H = np.linalg.inv(turn_table.base_H)
Table_home_T = turn_table.fwd(np.radians([-15,180]))
TBase_R1TCP_H = np.linalg.inv(np.matmul(turn_table.base_H,H_from_RT(Table_home_T.R,Table_home_T.p)))
R2base_table_H = np.matmul(R2base_R1base_H,turn_table.base_H)
print("Turn table relative to R2",R2base_table_H)

R2base_tableJ1_H = Transform(rpy2R(np.radians([-1.9178,-13.5699,87.8922])),[1049.384,418.432,-512.403])
tableJ1_base_H = Transform(np.eye(3),[0,0,-380])
R2base_tablebase_H = R2base_tableJ1_H*tableJ1_base_H
# R2base_table_H_act = R2base_tablebase_H*turn_table.fwd(np.array([0,0]))
R2base_table_H_act=H_from_RT(R2base_tablebase_H.R,R2base_tablebase_H.p)
print("ACt table relative to R2",R2base_table_H_act)

# exit()

use_R2S1=False
if use_R2S1:
    R2base_table_H=R2base_table_H_act

# exit()
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
data_dir='../../data/wall_weld_test/wall_param_data_collection/'
all_path_T = robot_weld_path_gen_2()
curve_sliced_R1=np.array(all_path_T[0])
curve_sliced_relative=[]
for path_p in curve_sliced_R1:
    this_p = np.matmul(TBase_R1TCP_H[:3,:3],path_p.p)+TBase_R1TCP_H[:3,-1]
    this_n = np.matmul(TBase_R1TCP_H[:3,:3],path_p.R[:,-1])
    curve_sliced_relative.append(np.append(this_p,this_n))
curve_sliced_relative=np.array(curve_sliced_relative)
curve_sliced_relative_origin=deepcopy(curve_sliced_relative)
print(curve_sliced_relative[:,:3])
## blade
# dataset='blade0.1/'
# sliced_alg='NX_slice2/'
# data_dir='../../data/'+dataset+sliced_alg
# out_dir=data_dir+'scans/'
# curve_sliced_relative=np.loadtxt(data_dir+'curve_sliced_relative/slice0.csv',delimiter=',')
##############################

### scan parameters
scan_speed=30 # scanning speed
Rz_turn_angle = np.radians(-45) # point direction w.r.t welds
# Ry_turn_angle = np.radians(-10) # rotate in y a bit, z-axis not pointing down, to have ik solution
Ry_turn_angle = np.radians(0) # rotate in y a bit, z-axis not pointing down, to have ik solution
# scan_stand_off_d = 243 ## mm
scan_stand_off_d = 243 ## mm
bounds_theta = np.radians(45) ## circular motion at start and end
# all_scan_angle = np.radians([-45,0,45,0]) ## scanning angles
# all_scan_angle = np.radians([0]) ## scanning angles

## scan angle
all_scan_angle = np.radians([-45,45]) ## scanning angles

#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
######### enter your wanted z height ########
all_layer_z=[50] ## all layer z height
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################

# param set 1
out_dir=data_dir+'path_Rz'+str(int(np.degrees(Rz_turn_angle)))+'_Ry'+str(int(np.degrees(Ry_turn_angle)))+\
    '_stand_off_d'+str(int(scan_stand_off_d))+'_b_theta'+str(int(np.degrees(bounds_theta)))
# param set scan angle
out_dir=out_dir+'_scan_angle'
for s_angle in all_scan_angle:
    out_dir=out_dir+str(int(np.degrees(s_angle)))+'_'
# param set z height
out_dir=out_dir+'z'
for z_height in all_layer_z:
    out_dir=out_dir+str(int(z_height))+'_'
out_dir=out_dir+'/'
out_scan_dir=out_dir+str(int(np.degrees(s_angle)))+'_'
print("out_dir:",out_dir)

### path gen ###
scan_p=[]
scan_R=[]
reverse_path_flag=False
reverse_scan_angle_flag=True
for layer_z in all_layer_z:
    
    curve_sliced_relative[:,2] = curve_sliced_relative_origin[:,2]+layer_z
    all_scan_angle=all_scan_angle[::-1]
    if layer_z==all_layer_z[-1]: # last layer scan
        all_scan_angle = np.append(all_scan_angle,0)
    scan_angle_i=0
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
            sub_scan_p=sub_scan_p[::-1]
            sub_scan_R=sub_scan_R[::-1]
        
        if scan_angle_i!=0:
            ## add connection path
            last_end_p = scan_p[-1]
            last_end_R = scan_R[-1]
            this_start_p = sub_scan_p[0]
            this_start_R = sub_scan_R[0]

            curve_slice_start_p = deepcopy(curve_sliced_relative[0][:3])
            if reverse_path_flag:
                curve_slice_start_p = deepcopy(curve_sliced_relative[-1][:3])

            p_bound_path,R_bound_path=get_connect_circle(last_end_p,last_end_R,this_start_p,this_start_R,curve_slice_start_p)
            p_bound_path.extend(sub_scan_p)
            sub_scan_p=deepcopy(p_bound_path)
            R_bound_path.extend(sub_scan_R)
            sub_scan_R=deepcopy(R_bound_path)
            ######################

        scan_p.extend(sub_scan_p)
        scan_R.extend(sub_scan_R)
        reverse_path_flag= not reverse_path_flag
        scan_angle_i+=1

scan_p=np.array(scan_p)
scan_R=np.array(scan_R)

# visualize_frames(scan_R,scan_p,size=3)

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

## visualize scanning path in positioner tool frame
# visualize_frames(scan_R[::20],scan_p[::20],size=10)
# exit()
############# path gen finish ################

############ redundancy resolution ###
method=1
if method==0:
    ### Method1: no resolving now, only get js, and no turntable
    # change everything to appropriate frame (currently: Robot2):
    scan_p_T=[]
    scan_p = np.matmul(R2base_table_H[:3,:3],scan_p.T).T + R2base_table_H[:3,-1]
    scan_R = np.matmul(R2base_table_H[:3,:3],scan_R)
    q_out1 = []
    # ik
    q_init=robot_scan.inv(scan_p[0],scan_R[0],zero_config)[0]
    q_out1.append(q_init)
    for path_i in range(len(scan_p)):
        this_js = robot_scan.inv(scan_p[path_i],scan_R[path_i],q_out1[-1])[0]
        q_out1.append(this_js)
    q_out1=np.array(q_out1)
elif method==1:

    try:
        q_out1=np.loadtxt(out_dir + 'scan_js1.csv',delimiter=',')
        q_out2=np.loadtxt(out_dir + 'scan_js2.csv',delimiter=',')
        print("Find Existing Solution")
    except:
        ### Method 2: stepwise qp, turntable involved
        rrs = redundancy_resolution_scanner(robot_scan,turn_table,scan_p,scan_R)
        q_init_table=np.radians([-70,150])
        pose_R2_table=Transform(R2base_table_H[:3,:3],R2base_table_H[:3,-1])*turn_table.fwd(q_init_table)
        q_init_robot = robot_scan.inv(np.matmul(pose_R2_table.R,scan_p[0])+pose_R2_table.p,np.matmul(pose_R2_table.R,scan_R[0]),zero_config)[0]
        print(np.degrees(q_init_robot))
        # q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt(q_init_robot,q_init_table,w2=0.03)
        q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt_Rz(q_init_robot,q_init_table,w2=0.03)
        # q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt_Rz(q_init_robot,q_init_table,w2=5)

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

        if np.any(q_out2[:,0]<np.radians(-76)):
            print("The output table traj is too tilted. Dont use this result.")
            exit()

        # scan_R_show=np.vstack((scan_R_show,scan_act_R[::10]))
        # scan_p_show=np.vstack((scan_p_show,scan_act_p[::10]))

        # visualize_frames(scan_R_show,scan_p_show,size=5)

        ## save redundancy resolution results
        Path(out_dir).mkdir(exist_ok=True)
        np.savetxt(out_dir + 'scan_js1.csv',q_out1,delimiter=',')
        np.savetxt(out_dir + 'scan_js2.csv',q_out2,delimiter=',')

# print(len(q_out1))
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
q_bp1=[]
q_bp2=[]
p_bp1=[]
p_bp2=[]
speed_bp=[]
zone_bp=[]
if method==0:
    for path_i in range(0,len(scan_p)):
        primitives.append('movel')
        q_bp1.append([q_out1[path_i]])
        q_bp2.append([np.radians([-15,180])])
        p_bp1.append(scan_p[path_i])
        speed_bp.append(scan_speed) ## mm/sec
        if path_i != len(scan_p)-1:
            zone_bp.append(0)
        else:
            zone_bp.append(0)
elif method==1:
    step_motion=50
    for path_i in range(0,len(scan_p),step_motion):
        primitives.append('movel')
        q_bp1.append([q_out1[path_i]])
        q_bp2.append([q_out2[path_i]])
        p_bp1.append(scan_p[path_i])
        speed_bp.append(scan_speed) ## mm/sec
#################################

####### add extension for time sync ####
init_T = robot_scan.fwd(q_bp1[0][0])
init_x_dir = -init_T.R[:,0]
init_T_align = deepcopy(init_T)
init_T_align.p = init_T_align.p+init_x_dir*50 # move 50 mm
init_q_align = robot_scan.inv(init_T_align.p,init_T_align.R,q_bp1[0][0])[0]
q_bp1.insert(0,[init_q_align])
q_bp2.insert(0,[q_out2[0]])
p_bp1.insert(0,init_T_align.p)
speed_bp.insert(0,scan_speed) ## mm/sec
########################################

vd_relative=scan_speed
lam1=calc_lam_js(q_out1,robot_scan)
lam2=calc_lam_js(q_out2,turn_table)
lam_relative=calc_lam_cs(scan_p)
s1_all,_=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,np.arange(0,len(scan_p),20).astype(int))
# print(s1_all)

use_artec_studio=True
input("Press Enter to start moving")

if not use_artec_studio:
    ### scanner hardware
    c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')
    cscanner = ContinuousScanner(c)

### execute motion ###
## move to start
client = MotionProgramExecClient()
to_start_speed=10
robot_client=MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
target2=['MOVJ',np.degrees(q_bp2[0][0]),to_start_speed]
robot_client.MoveJ(np.degrees(q_bp1[0][0]), to_start_speed, 0, target2=target2)
client.execute_motion_program(robot_client)

input("Open Artec Studio or Scanner and Press Enter to start moving")

if not use_artec_studio:
    ## scanner start
    cscanner.start_capture()

## motion start
robot_client=MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)

# calibration motion
target2=['MOVJ',np.degrees(q_bp2[1][0]),10]
robot_client.MoveL(np.degrees(q_bp1[1][0]), scan_speed, 0, target2=target2)
# routine motion
for path_i in range(2,len(q_bp1)-1):
    target2=['MOVJ',np.degrees(q_bp2[path_i][0]),10]
    robot_client.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
target2=['MOVJ',np.degrees(q_bp2[-1][0]),10]
robot_client.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)
robot_stamps,curve_pulse_exe,_,_ = client.execute_motion_program(robot_client)
# q_out_exe=np.divide(curve_pulse_exe[:,6:],robot_scan.pulse2deg)
q_out_exe=curve_pulse_exe[:,6:]

if not use_artec_studio:
    ## scanner end
    cscanner.end_capture()
    st=time.perf_counter()
    scans,scan_stamps=cscanner.get_capture()
    # dt=time.perf_counter()-st
    # print("dt:",dt)
    # print("Robot last joint:",q_out1_exe[-1])

input("Press Stop on Artec Studio and Move Home")
robot_client=MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=turn_table.pulse2deg)
# move robot to home
q2=np.zeros(6)
q2[0]=90
q3=[-15,180]
robot_client=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot_scan.pulse2deg)
robot_client.MoveJ(q2,to_start_speed,0)
robot_stamps,curve_pulse_exe,_,_ = client.execute_motion_program(robot_client)

robot_client=MotionProgram(ROBOT_CHOICE='ST1',pulse2deg=turn_table.pulse2deg)
robot_client.MoveJ(q3,to_start_speed,0)
robot_stamps,curve_pulse_exe,_,_ = client.execute_motion_program(robot_client)
#####################
# exit()

print(q_out_exe)
if not use_artec_studio:
    ## save traj
    Path(out_dir).mkdir(exist_ok=True)
    # save poses
    np.savetxt(out_dir + 'scan_js_exe.csv',np.radians(q_out_exe),delimiter=',')
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
