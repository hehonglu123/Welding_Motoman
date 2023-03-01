import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from path_calc import *
# from utils import *

class redundancy_resolution(object):
    ###robot1 hold weld torch, positioner hold welded part
    def __init__(self,robot,positioner,scan_p,scan_R):
        # curve_sliced: list of sliced layers, in curve frame
        # robot: welder robot
        # positioner: 2DOF rotational positioner
        self.robot=robot
        self.positioner=positioner
        self.scan_p=scan_p
        self.scan_R=scan_R

        R2base_R1base_H = np.linalg.inv(robot.base_H)
        R2base_table_H = np.matmul(R2base_R1base_H,positioner.base_H)
        self.R2baseR_table=Transform(R2base_table_H[:3,:3],R2base_table_H[:3,-1])
    
        self.lim_factor=np.radians(1)

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

        Kq=w1*np.eye(12)    #small value to make sure positive definite
        Kq[6:,6:]=w2*np.eye(6)	#larger weights for turntable for it moves slower

        upper_limit=np.hstack(self.robot.upper_limit,self.positioner.upper_limit)
        lower_limit=np.hstack(self.robot.lower_limit,self.positioner.lower_limit)

        for i in range(len(self.scan_path)):
            try:
                error_fb=999
                while error_fb>0.1:
                    poset1_1=self.robot.fwd(q_all1[-1])
                    poset2_2=self.positioner.fwd(q_all2[-1])
                    poset2_1=self.R2baseR_table*poset2_2
                    pose1_t2=poset2_1.inv()
                    poset1_t2=pose1_t2*poset1_1
                    dpt1t2_t2=np.matmul(pose1_t2.R,(poset1_1-poset2_1))
                    Rt1_t2=np.matmul(pose1_t2.R,poset1_1.R)

                    ## error=euclideans norm (p)+forbinius norm (R)
                    error_fb=np.linalg.norm(poset1_t2.p-self.scan_p[i])+np.linalg.norm(np.matmul(self.scan_R.T,poset1_t2.R))

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

                    H=np.dot(np.transpose(J_all_p),J_all_p)+Kq+Kw*np.dot(np.transpose(J_all_R),J_all_R)
                    H=(H+np.transpose(H))/2

                    vd = self.scan_p[i]-dpt1t2_t2
                    omega_d=s_err_func(Rt1_t2.T@self.scan_R[i])

                    f=-np.dot(np.transpose(J_all_p),vd)-Kw*np.dot(np.transpose(J_all_R),omega_d)
                    qdot=solve_qp(H,f,lb=lower_limit-np.hstack((q_all1[-1],q_all2[-1]))+self.lim_factor*np.ones(12),ub=upper_limit-np.hstack((q_all1[-1],q_all2[-1]))-self.lim_factor*np.ones(12),solver='quadprog')
                    
                    alpha=1
                    q_all1.append(q_all1[-1]+alpha*qdot[:6])
                    q_all2.append(q_all2[-1]+alpha*qdot[6:])
                    j_all1.append(self.robot1.jacobian(q_all1[-1]))
                    j_all2.append(self.robot2.jacobian(q_all2[-1]))

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

def main():

    config_dir='../config/'
    robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'scanner_tcp2.csv',\
	    base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg.csv')
    turn_table=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',base_transformation_file=config_dir+'D500B_pose.csv',\
        pulse2deg_file_path=config_dir+'D500B_pulse2deg.csv')

if __name__=='__main__':
    main()