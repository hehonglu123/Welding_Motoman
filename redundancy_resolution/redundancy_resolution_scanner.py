import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *
# from utils import *
from math import floor

class redundancy_resolution_scanner(object):
    ###robot1 hold weld torch, positioner hold welded part
    def __init__(self,robot,positioner,scan_p,scan_R):
        # curve_sliced: list of sliced layers, in curve frame
        # robot: welder robot
        # positioner: 2DOF rotational positioner
        self.robot=robot
        self.positioner=positioner
        self.scan_p=scan_p
        self.scan_R=scan_R
        self.scan_n=scan_R[:,:,-1]

        R2base_R1base_H = np.linalg.inv(robot.base_H)
        R2base_table_H = np.matmul(R2base_R1base_H,positioner.base_H)
        self.R2baseR_table=Transform(R2base_table_H[:3,:3],R2base_table_H[:3,-1])
    
        self.lim_factor=np.radians(5)

    def arm_table_stepwise_opt(self,q_init1,q_init2,w1=0.01,w2=0.01):
        
        ## robot q output
        q_all1=[q_init1]
        q_out1=[q_init1]
        q_all2=[q_init2]
        q_out2=[q_init2]
        j_all1=[self.robot.jacobian(q_init1)]
        j_all2=[self.positioner.jacobian(q_init2)]
        j_out1=[j_all1[0]]
        j_out2=[j_all2[0]]

        #####weights
        Kw=0.1

        Kq=w1*np.eye(8)    #small value to make sure positive definite
        Kq[6:,6:]=w2*np.eye(2)	#larger weights for turntable for it moves slower

        upper_limit=np.hstack((self.robot.upper_limit,self.positioner.upper_limit))
        lower_limit=np.hstack((self.robot.lower_limit,self.positioner.lower_limit))

        for i in range(len(self.scan_p)):
            # if i%100==0:
            #     print(i)
            # print("==================")
            # print(i,'/',len(self.scan_p))
            # print(np.degrees(np.append(q_all1[-1],q_all2[-1])))
            try:
                error_fb=999
                while error_fb>0.002:
                    
                    # print(error_fb)
                    poset1_1=self.robot.fwd(q_all1[-1])
                    poset2_2=self.positioner.fwd(q_all2[-1])
                    poset2_1=self.R2baseR_table*poset2_2
                    pose1_t2=poset2_1.inv()
                    poset1_t2=pose1_t2*poset1_1
                    dpt1t2_t2=np.matmul(pose1_t2.R,(poset1_1.p-poset2_1.p))
                    Rt1_t2=np.matmul(pose1_t2.R,poset1_1.R)

                    ## error=euclideans norm (p)+forbinius norm (R)
                    p_norm= np.linalg.norm(poset1_t2.p-self.scan_p[i])
                    R_norm=np.linalg.norm(np.matmul(Rt1_t2.T,self.scan_R[i])-np.eye(3))
                    error_fb=p_norm+R_norm

                    if error_fb>1000:
                        print("Error too large:",error_fb)
                        raise AssertionError
                    
                    ## prepare Jacobian matrix w.r.t positioner
                    J1=j_all1[-1]
                    J1p=np.matmul(pose1_t2.R,J1[3:,:])
                    J1R=np.matmul(pose1_t2.R,J1[:3,:])
                    J2=j_all2[-1]
                    J2p=np.matmul(poset2_2.inv().R,J2[3:,:])
                    J2R=np.matmul(poset2_2.inv().R,J2[:3,:])
                    J_all_p=np.hstack((J1p,-J2p+hat(dpt1t2_t2)@J2R))
                    J_all_R=np.hstack((J1R,-J2R))

                    if i==0:
                        J_all = np.vstack((J_all_R,J_all_p))
                        u,s,v=np.linalg.svd(J_all)
                        u1,s1,v1=np.linalg.svd(J1)
                        if np.min(s)<0.01:
                            print(np.min(s))
                            print(u[:,-1])
                            print(np.min(s1))
                            return [],[],[],[]

                    H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
                    H=(H+np.transpose(H))/2

                    vd = self.scan_p[i]-dpt1t2_t2
                    omega_d=s_err_func(Rt1_t2@self.scan_R[i].T)
                    # omega_d=s_err_func(self.scan_R[i].T@Rt1_t2)

                    f=-np.dot(np.transpose(J_all_p),vd)+Kw*np.dot(np.transpose(J_all_R),omega_d)
                    qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(8),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(8),solver='quadprog')
                    
                    alpha=1
                    q_all1.append(q_all1[-1]+alpha*qdot[:6])
                    q_all2.append(q_all2[-1]+alpha*qdot[6:])
                    j_all1.append(self.robot.jacobian(q_all1[-1]))
                    j_all2.append(self.positioner.jacobian(q_all2[-1]))

            except:
                print("Error fb:",error_fb)
                print("Min S:",np.min(s))
                print("Joint:",np.degrees(np.append(q_all1[-1],q_all2[-1])))
                traceback.print_exc()
                q_out1.append(q_all1[-1])
                q_out2.append(q_all2[-1])
                j_out1.append(j_all1[-1])
                j_out2.append(j_all2[-1])		
                raise AssertionError
                break
            q_out1.append(q_all1[-1])
            q_out2.append(q_all2[-1])
            j_out1.append(j_all1[-1])
            j_out2.append(j_all2[-1])

        q_out1=np.array(q_out1)[1:]
        q_out2=np.array(q_out2)[1:]
        j_out1=j_out1[1:]
        j_out2=j_out2[1:]
        return q_out1, q_out2, j_out1, j_out2
    
    def arm_table_stepwise_opt_Rz(self,q_init1,q_init2,w1=0.01,w2=0.01):
        
        ## robot q output
        q_all1=[q_init1]
        q_out1=[q_init1]
        q_all2=[q_init2]
        q_out2=[q_init2]
        j_all1=[self.robot.jacobian(q_init1)]
        j_all2=[self.positioner.jacobian(q_init2)]
        j_out1=[j_all1[0]]
        j_out2=[j_all2[0]]

        #####weights
        Kw=0.1

        Kq=w1*np.eye(8)    #small value to make sure positive definite
        Kq[6:,6:]=w2*np.eye(2)	#larger weights for turntable for it moves slower

        upper_limit=np.hstack((self.robot.upper_limit,self.positioner.upper_limit))
        lower_limit=np.hstack((self.robot.lower_limit,self.positioner.lower_limit))

        show_perc = True
        show_perc_before=-1
        for i in range(len(self.scan_p)):
            # if show_perc_before<floor(i/len(self.scan_p)*100):
            #     show_perc = True
            # if floor(i/len(self.scan_p)*100)%10==0 and show_perc:
            #     print("Solve Redundancy:",str(floor(i/len(self.scan_p)*100))+"%")
            #     show_perc=False
            if i % 1000 ==0:
                print("Solve Redundancy:",str(floor(i/len(self.scan_p)*100))+"%")
            try:
                error_fb=999
                while error_fb>0.01:
                    poset1_1=self.robot.fwd(q_all1[-1])
                    poset2_2=self.positioner.fwd(q_all2[-1])
                    poset2_1=self.R2baseR_table*poset2_2
                    pose1_t2=poset2_1.inv()
                    poset1_t2=pose1_t2*poset1_1
                    dpt1t2_t2=np.matmul(pose1_t2.R,(poset1_1.p-poset2_1.p))
                    Rt1_t2=np.matmul(pose1_t2.R,poset1_1.R)

                    ## error=euclideans norm (p)+forbinius norm (R)
                    p_norm= np.linalg.norm(poset1_t2.p-self.scan_p[i])
                    n_norm=np.linalg.norm(poset1_t2.R[:,-1]-self.scan_n[i])
                    error_fb=p_norm+n_norm

                    if error_fb>1000:
                        print("Error too large:",error_fb)
                        raise AssertionError
                    
                    ## prepare Jacobian matrix w.r.t positioner
                    J1=j_all1[-1]
                    J1p=np.matmul(pose1_t2.R,J1[3:,:])
                    J1R=np.matmul(pose1_t2.R,J1[:3,:])
                    J1R_mod=-np.dot(hat(np.dot(poset2_1.R.T,poset1_1.R[:,-1])),J1R)
                    J2=j_all2[-1]
                    J2p=np.matmul(poset2_2.inv().R,J2[3:,:])
                    J2R=np.matmul(poset2_2.inv().R,J2[:3,:])
                    J2R_mod=-np.dot(hat(np.dot(poset2_1.R.T,poset1_1.R[:,-1])),J2R)
                    J_all_p=np.hstack((J1p,-J2p+hat(dpt1t2_t2)@J2R))
                    J_all_R=np.hstack((J1R_mod,-J2R_mod))

                    H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
                    H=(H+np.transpose(H))/2

                    vd = self.scan_p[i]-dpt1t2_t2
                    ezdotd=self.scan_n[i]-Rt1_t2[:,-1]

                    f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),ezdotd)
                    qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(8),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(8),solver='quadprog')
                    
                    alpha=1
                    q_all1.append(q_all1[-1]+alpha*qdot[:6])
                    q_all2.append(q_all2[-1]+alpha*qdot[6:])
                    j_all1.append(self.robot.jacobian(q_all1[-1]))
                    j_all2.append(self.positioner.jacobian(q_all2[-1]))

            except:
                traceback.print_exc()
                q_out1.append(q_all1[-1])
                q_out2.append(q_all2[-1])
                j_out1.append(j_all1[-1])
                j_out2.append(j_all2[-1])		
                raise AssertionError
                break
            q_out1.append(q_all1[-1])
            q_out2.append(q_all2[-1])
            j_out1.append(j_all1[-1])
            j_out2.append(j_all2[-1])

        q_out1=np.array(q_out1)[1:]
        q_out2=np.array(q_out2)[1:]
        j_out1=j_out1[1:]
        j_out2=j_out2[1:]
        return q_out1, q_out2, j_out1, j_out2

    def arm_table_stepwise_opt_Rzup(self,q_table_seed):

        pass
        ####baseline redundancy resolution, with fixed orientation
        positioner_js=self.positioner_resolution(curve_sliced_relative,q_seed=q_positioner_seed,smooth_filter=smooth_filter)		#solve for positioner first
        
        ###singularity js smoothing
        positioner_js=self.introducing_tolerance2(positioner_js)
        positioner_js=self.conditional_rolling_average(positioner_js)
        if smooth_filter:
            positioner_js=self.rolling_average(positioner_js)
        positioner_js[0][0][:,1]=positioner_js[1][0][0,1]

def main():

    config_dir='../config/'
    robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	    base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
    turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
        pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

if __name__=='__main__':
    main()