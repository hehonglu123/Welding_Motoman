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
from dx200_motion_program_exec_client import *

from general_robotics_toolbox import *
from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time
import numpy as np

zero_config=np.array([0.,0.,0.,0.,0.,0.])

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

def get_connect_circle(start_p,start_R,end_p,end_R,pc,scan_stand_off_d):

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

class ScanPathGen():
    def __init__(self,robot,positioner,scan_stand_off_d=243,Rz_angle=0,Ry_angle=0,bounds_theta=0) -> None:
        
        self.robot=robot
        self.positioner=positioner
        self.scan_stand_off_d=scan_stand_off_d
        self.Rz_angle=Rz_angle
        self.Ry_angle=Ry_angle
        self.bounds_theta=bounds_theta

    def _gen_scan_path(self,curve_sliced_relative,all_layer_z,all_scan_angle):

        curve_sliced_relative_origin=deepcopy(curve_sliced_relative)

        ### path gen ###
        scan_p=[]
        scan_R=[]
        reverse_path_flag=False
        reverse_scan_angle_flag=True
        for layer_z in all_layer_z:
            
            curve_sliced_relative[:,2] = curve_sliced_relative_origin[:,2]+layer_z
            all_scan_angle=all_scan_angle[::-1]
            if layer_z==all_layer_z[-1]: # last layer scan
                if all_scan_angle[-1]!=0:
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
                        this_scan_p = this_p + this_scan_R[:,-1]*self.scan_stand_off_d # stand off distance to scan
                        sub_scan_p.append(this_scan_p)
                        sub_scan_R.append(this_scan_R)
                        k = deepcopy(this_scan_R[:,0])
                        p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,-self.bounds_theta)
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
                    this_scan_p = this_p + Rz*self.scan_stand_off_d # stand off distance to scan

                    # add start bound condition
                    if pT_i == 0:
                        k = deepcopy(Rx)
                        p_bound_path,R_bound_path=get_bound_circle(this_scan_p,this_scan_R,this_p,k,self.bounds_theta)
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
        scan_R = np.matmul(scan_R,rot([0.,0.,1.],self.Rz_angle))
        scan_R = np.matmul(scan_R,rot([0.,1.,0.],self.Ry_angle))

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

        return scan_p,scan_R
    
    def _gen_js_path(self,scan_p,scan_R,solve_js_method,q_init_table=np.radians([-15,180])):

        T_S1Base_R2Base = np.matmul(np.linalg.inv(self.robot.base_H),self.positioner.base_H)
        T_S1Base_R2Base = Transform(T_S1Base_R2Base[:3,:3],T_S1Base_R2Base[:3,-1])

        ############ redundancy resolution ###
        if solve_js_method==0:
            q_out2 = q_init_table # positioner joint angle
            T_S1TCP_S1Base = self.positioner.fwd(q_out2)
            T_S1TCP_R2Base = T_S1Base_R2Base*T_S1TCP_S1Base

            ### Method1: no resolving now, only get js, and no turntable
            # change everything to appropriate frame (currently: Robot2):
            scan_p_R2Base = np.matmul(T_S1TCP_R2Base.R,scan_p.T).T + T_S1TCP_R2Base.p
            scan_R_R2Base = np.matmul(T_S1TCP_R2Base.R,scan_R)
            q_out1 = []
            # ik
            q_init=self.robot.inv(scan_p_R2Base[0],scan_R_R2Base[0],zero_config)[0]
            print(np.degrees(q_init))
            q_out1.append(q_init)
            for path_i in range(len(scan_p_R2Base)):
                this_js = self.robot.inv(scan_p_R2Base[path_i],scan_R_R2Base[path_i],q_out1[-1])
                this_js=this_js[0]
                q_out1.append(np.array(this_js))
            q_out1=np.array(q_out1)
            q_out2=np.tile(q_out2,(len(q_out1),1))

        elif solve_js_method==1:
            ### Method 2: stepwise qp, turntable involved
            rrs = redundancy_resolution_scanner(self.robot,self.positioner,scan_p,scan_R)
            q_init_table=np.radians([-70,150])
            
            pose_R2_table=T_S1Base_R2Base*self.positioner.fwd(q_init_table)
            q_init_robot = self.robot.inv(np.matmul(pose_R2_table.R,scan_p[0])+pose_R2_table.p,np.matmul(pose_R2_table.R,scan_R[0]),zero_config)[0]
            # q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt(q_init_robot,q_init_table,w2=0.03)
            q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt_Rz(q_init_robot,q_init_table,w2=0.03)
            # q_out1, q_out2, j_out1, j_out2=rrs.arm_table_stepwise_opt_Rz(q_init_robot,q_init_table,w2=5)

            scan_act_R=[]
            scan_act_p=[]
            for i in range(len(q_out1)):
                poset1_1=self.robot.fwd(q_out1[i])
                poset2_2=self.positioner.fwd(q_out2[i])
                poset2_1=T_S1Base_R2Base*poset2_2
                pose1_t2=poset2_1.inv()
                poset1_t2=pose1_t2*poset1_1
                scan_act_R.append(poset1_t2.R)
                scan_act_p.append(poset1_t2.p)
            scan_act_R=np.array(scan_act_R)
            scan_act_p=np.array(scan_act_p)

            if np.any(q_out2[:,0]<np.radians(-76)):
                print("The output table traj is too tilted. Dont use this result.")
                exit()

            # scan_R_show=scan_R[::50]
            # scan_p_show=scan_p[::50]
            # scan_R_show=np.vstack((scan_R_show,scan_act_R[::10]))
            # scan_p_show=np.vstack((scan_p_show,scan_act_p[::10]))
            # visualize_frames(scan_R_show,scan_p_show,size=5)
        
        return q_out1,q_out2
    
    def gen_scan_path(self,curve_sliced_relative,all_layer_z,all_scan_angle,solve_js_method=1,q_init_table=np.radians([-15,180]),scan_path_dir=None):
        
        # generate cartesian path
        scan_p,scan_R = self._gen_scan_path(curve_sliced_relative,all_layer_z,all_scan_angle)
        # generate joint space path
        if scan_path_dir is not None:
            q_out1=np.loadtxt(scan_path_dir + 'scan_js1.csv',delimiter=',')
            q_out2=np.loadtxt(scan_path_dir + 'scan_js2.csv',delimiter=',')
        else:
            q_out1,q_out2 = self._gen_js_path(scan_p,scan_R,solve_js_method,q_init_table)

        return scan_p,scan_R,q_out1,q_out2

    def gen_motion_program(self,q_out1,q_out2,scan_p,scan_speed):
        
        primitives=[]
        q_bp1=[]
        q_bp2=[]
        p_bp1=[]
        p_bp2=[]
        speed_bp=[]
        zone_bp=[]
        step_motion=50
        for path_i in range(0,len(scan_p),step_motion):
            primitives.append('movel')
            q_bp1.append([q_out1[path_i]])
            q_bp2.append([q_out2[path_i]])
            p_bp1.append(scan_p[path_i])
            speed_bp.append(scan_speed) ## mm/sec
        #################################

        ####### add extension for time sync ####
        init_sync_move = 50 # move 50 mm
        init_T = self.robot.fwd(q_bp1[0][0])
        init_x_dir = -init_T.R[:,0]
        init_T_align = deepcopy(init_T)
        init_T_align.p = init_T_align.p+init_x_dir*init_sync_move 
        init_q_align = self.robot.inv(init_T_align.p,init_T_align.R,q_bp1[0][0])[0]
        q_bp1.insert(0,[init_q_align])
        q_bp2.insert(0,[q_out2[0]])
        p_bp1.insert(0,init_T_align.p)
        speed_bp.insert(0,scan_speed) ## mm/sec
        ########################################

        vd_relative=scan_speed
        lam1=calc_lam_js(q_out1,self.robot)
        lam2=calc_lam_js(q_out2,self.positioner)
        lam_relative=calc_lam_cs(scan_p)
        if lam2[-1]!=0:
            s1_all,s2_all=calc_individual_speed(vd_relative,lam1,lam2,lam_relative,np.arange(0,len(scan_p),20).astype(int))
        else:
            s1_all=deepcopy(speed_bp)
            s2_all=np.zeros(len(s1_all))

        return q_bp1,q_bp2,s1_all,s2_all