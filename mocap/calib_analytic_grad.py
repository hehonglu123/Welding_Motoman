import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from general_robotics_toolbox import *
import sys
sys.path.append('../toolbox/')
from robot_def import *

Rx=np.array([1,0,0])
Ry=np.array([0,1,0])
Rz=np.array([0,0,1])

def s_err(ER,mode=3):
    
    mode_choice=[3]
    if mode not in mode_choice:
        assert AssertionError,'Mode choise '+str(mode_choice)
    
    if mode==3:
        k,theta=R2rot(ER)
        derr=2*theta*k
    
    return derr

def jacobian_param(param,robot,theta):
    
    jN=len(theta)
    
    # get current P,H (given param)
    P=param[:(jN+1)*3]
    P=np.reshape(P,((jN+1),3))
    total_p = (jN+1)*3
    param_h=param[total_p:]
    H=[]
    rot_k1_alpha=[]
    rot_k2_beta=[]
    for j in range(jN):
        rot_k1_alpha.append(rot(robot.param_k1[j],param_h[2*j]))
        rot_k2_beta.append(rot(robot.param_k2[j],param_h[2*j+1]))
        hi = rot_k2_beta[-1]@\
           rot_k1_alpha[-1]@robot.robot.H[:,j]
        H.append(hi)
    H=np.array(H)
    Pn=deepcopy(robot.P_nominal)
    Hn=deepcopy(robot.H_nominal)
    
    # foward kinematics
    robot.robot.P=P.T
    robot.robot.H=H.T
    T_tool = robot.fwd(theta)
    p0T=T_tool.p
    R0T=T_tool.R
    
    # get jacobian
    J=np.zeros((6,len(param))) # J=[JR;Jp]
    last_R0j=np.eye(3)
    p0j=np.zeros(3)
    for j in range(jN):
        Rj1j=rot(H[j],theta[j]) # R_{j-1,j}
        R0j=last_R0j@Rj1j # R_{0,j}
        RjT = R0j.T@R0T
        p0j=p0j+last_R0j@P[j] # p0j_0
        pjT_j=(R0j.T)@(p0T-p0j)
        # gradient of P w.r.t p0T
        J[3:,3*j:3*(j+1)]=last_R0j
        # gradient of R0T,P0T w.r.t alpha  
        drot_alpha = last_R0j@rot_k2_beta[j]@hat(robot.param_k1[j])@rot_k1_alpha[j]@\
                     hat(Hn[j])*theta[j]@Rj1j
        J[:3,total_p+2*j]=invhat(drot_alpha@RjT)
        J[3:,total_p+2*j]=drot_alpha@pjT_j/180
        # gradient of R0T,P0T w.r.t beta
        drot_beta = last_R0j@hat(robot.param_k2[j])@rot_k2_beta[j]@rot_k1_alpha[j]@\
                    hat(Hn[j])*theta[j]@Rj1j
        # if j==3:
        #     print(hat(robot.param_k2[j]))
        #     print(last_R0j)
        #     print(last_R0j@hat(robot.param_k2[j]))
        #     print(drot_beta@RjT)
        #     exit
        J[:3,total_p+2*j+1]=invhat(drot_beta@RjT)
        J[3:,total_p+2*j+1]=drot_beta@pjT_j/180
        last_R0j=R0j
    J[3:,total_p-3:total_p] = last_R0j # p6T
    
    return J

def main():
    config_dir='../config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot.P_nominal=deepcopy(robot.robot.P)
    robot.H_nominal=deepcopy(robot.robot.H)
    robot.P_nominal=robot.P_nominal.T
    robot.H_nominal=robot.H_nominal.T
    
    jN = len(robot.robot.H[0])
    k1=[]
    k2=[]
    for j in range(jN):
        if np.fabs(np.dot(Rx,robot.robot.H[:,j]))>0.999:
            if np.dot(Rx,robot.robot.H[:,j])>0:
                k1.append(Ry)
                k2.append(Rz)
            else:
                k1.append(Rz)
                k2.append(Ry)
        elif np.fabs(np.dot(Ry,robot.robot.H[:,j]))>0.999:
            if  np.dot(Ry,robot.robot.H[:,j])>0:
                k1.append(Rz)
                k2.append(Rx)
            else:
                k1.append(Rx)
                k2.append(Rz)
        elif np.fabs(np.dot(Rz,robot.robot.H[:,j]))>0.999:
            if np.dot(Rz,robot.robot.H[:,j])>0:
                k1.append(Rx)
                k2.append(Ry)
            else:
                k1.append(Ry)
                k2.append(Rx)
        else:
            assert AssertionError,'Assume h is aligned well with x or y or z axis.'
    robot.param_k1=np.array(k1)
    robot.param_k2=np.array(k2)
    
    test_theta = np.radians([[10,10,10,10,10,10]])
    
    param = np.zeros(3*(jN+1)+2*jN)
    for test_th in test_theta:
        # analytical J
        J_ana = jacobian_param(param,robot,test_th)
        # print(J_ana)
        fig, ax = plt.subplots()
        im = ax.matshow(J_ana,cmap='RdBu')
        fig.colorbar(im)
        plt.show()
        
        # numerical J
        


if __name__=='__main__':
    
    main()