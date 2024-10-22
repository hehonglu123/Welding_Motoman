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

def jacobian_param_numerical(param,robot,theta,unit='radians'):
    from numpy.random import default_rng
    rng = default_rng()
    
    jN=len(theta)
    numerical_iteration=1000
    dP_up_range = 0.05
    dP_low_range = 0.01
    dab_up_range = np.radians(0.1)
    dab_low_range = np.radians(0.03)
    Pn=deepcopy(robot.P_nominal)
    Hn=deepcopy(robot.H_nominal)
    
    d_T_all = [] # difference in robot T
    d_param_all = [] # difference in param
    for iter_i in range(numerical_iteration):
        param_init = deepcopy(param)
        
        dP=rng.uniform(low=-(dP_up_range-dP_low_range),high=(dP_up_range-dP_low_range),size=((jN+1)*3,))
        dP=dP+dP/np.fabs(dP)*dP_low_range
        dab=rng.uniform(low=-(dab_up_range-dab_low_range),high=(dab_up_range-dab_low_range),size=(jN*2,))
        dab=dab+dab/np.fabs(dab)*dab_low_range
        # dP[:] = np.zeros((jN+1)*3)
        # dab[:-2] = np.zeros(jN*2-2)
        # dab[-1] = 0
        dparam = np.append(dP,dab)
        param_pert = param_init+dparam
        # dparam[(jN+1)*3:]=np.degrees(dparam[(jN+1)*3:])
        # get P,H (given param)
        P_init=param_init[:(jN+1)*3]
        P_init=np.reshape(P_init,((jN+1),3))
        P_pert=param_pert[:(jN+1)*3]
        P_pert=np.reshape(P_pert,((jN+1),3))
        
        param_h_init=param_init[(jN+1)*3:]
        param_h_pert=param_pert[(jN+1)*3:]
        
        H_init=[]
        H_pert=[]
        for j in range(jN):
            k1=robot.param_k1[j]
            k2=robot.param_k2[j]
            H_init.append(rot(k2,param_h_init[2*j+1])@rot(k1,param_h_init[2*j])@Hn[j])
            H_pert.append(rot(k2,param_h_pert[2*j+1])@rot(k1,param_h_pert[2*j])@Hn[j])
        H_init=np.array(H_init)
        H_pert=np.array(H_pert)
        
        # find dp dRR^T
        robot.robot.P=P_init.T
        robot.robot.H=H_init.T
        T_init = robot.fwd(theta)
        robot.robot.P=P_pert.T
        robot.robot.H=H_pert.T
        T_pert = robot.fwd(theta)
        dp = T_pert.p-T_init.p
        # input()
        # dR = T_init.R.T@T_pert.R
        # k,th = R2rot(dR)
        # if th == 0:
        #     ktheta=np.zeros(3)
        # else:
        #     ktheta=k*th
        dR = T_pert.R-T_init.R
        dRRT = dR@T_init.R.T
        ktheta = invhat(dRRT)
        # print(ktheta)
        dT = np.append(ktheta,dp)
        
        # append dT dparam
        d_T_all.append(dT)
        d_param_all.append(dparam)
    
    d_T_all=np.array(d_T_all)
    d_param_all=np.array(d_param_all)
    
    num_J = np.linalg.pinv(d_param_all)@d_T_all
    num_J = num_J.T
    # num_J = (d_T_all.T)@np.linalg.pinv(d_param_all.T)
    
    if unit!='radians':
        num_J[3:,(jN+1)*3:] = num_J[3:,(jN+1)*3:]*(np.pi/180)
    
    return num_J

def get_H_param_axis(robot):
    
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
    return robot

def get_PH_from_param(param,robot,unit='radians'):
    
    jN=len(robot.robot.H[0])
    # get current P,H (given param)
    P=deepcopy(param[:(jN+1)*3])
    P=np.reshape(P,((jN+1),3))
    total_p = (jN+1)*3
    param_h=deepcopy(param[total_p:])
    if unit!='radians':
        param_h = np.radians(param_h)
    H=[]
    rot_k1_alpha=[]
    rot_k2_beta=[]
    for j in range(jN):
        rot_k1_alpha.append(rot(robot.param_k1[j],param_h[2*j]))
        rot_k2_beta.append(rot(robot.param_k2[j],param_h[2*j+1]))
        hi = rot_k2_beta[-1]@\
           rot_k1_alpha[-1]@robot.H_nominal[j]
        H.append(hi)
    H=np.array(H)
    robot.robot.P=P.T
    robot.robot.H=H.T
    return robot

def get_param_from_PH(robot,this_P,this_H,nom_H):

    param_H = []
    for i,h in enumerate(this_H.T):
        theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
        theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
        param_H.extend(theta_sol[::-1])

    return np.reshape(this_P.T,-1),np.array(param_H)

def jacobian_param(param,robot,theta,unit='radians',minimal=False):
    
    jN=len(theta)
    
    # get current P,H (given param)
    # robot = get_PH_from_param(param,robot)
    
    P=param[:(jN+1)*3]
    P=np.reshape(P,((jN+1),3))
    total_p = (jN+1)*3
    param_h=param[total_p:]
    if unit!='radians':
        param_h = np.radians(param_h)
    H=[]
    rot_k1_alpha=[]
    rot_k2_beta=[]
    for j in range(jN):
        rot_k1_alpha.append(rot(robot.param_k1[j],param_h[2*j]))
        rot_k2_beta.append(rot(robot.param_k2[j],param_h[2*j+1]))
        hi = rot_k2_beta[-1]@\
           rot_k1_alpha[-1]@robot.H_nominal[j]
        H.append(hi)
    H=np.array(H)
    
    robot.robot.P=P.T
    robot.robot.H=H.T
    Pn=deepcopy(robot.P_nominal)
    Hn=deepcopy(robot.H_nominal)
    
    # foward kinematics
    
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
        # Hn_base = last_R0j@Hn[j]
        projection_P = np.eye(3)-np.outer(Hn[j],Hn[j])/np.dot(Hn[j],Hn[j])
        if minimal:
            J[3:,3*j:3*(j+1)]=last_R0j@projection_P # move only in the direction orthogonal to Hn (a plane)
        else:
            J[3:,3*j:3*(j+1)]=last_R0j
        # gradient of R0T,P0T w.r.t alpha  
        dhda = rot_k2_beta[j]@hat(robot.param_k1[j])@rot_k1_alpha[j]@Hn[j]
        dRda = np.sin(theta[j])*hat(dhda)+(1-np.cos(theta[j]))*(hat(dhda)@hat(H[j])+hat(H[j])@hat(dhda))
        dR0jda = last_R0j@dRda
        J[:3,total_p+2*j]=invhat(dR0jda@RjT@(R0T.T))
        J[3:,total_p+2*j]=dR0jda@pjT_j
        # gradient of R0T,P0T w.r.t beta
        dhdb = hat(robot.param_k2[j])@rot_k2_beta[j]@rot_k1_alpha[j]@Hn[j]
        dRdb = np.sin(theta[j])*hat(dhdb)+(1-np.cos(theta[j]))*(hat(dhdb)@hat(H[j])+hat(H[j])@hat(dhdb))
        dR0jdb = last_R0j@dRdb
        J[:3,total_p+2*j+1]=invhat(dR0jdb@RjT@(R0T.T))
        J[3:,total_p+2*j+1]=dR0jdb@pjT_j
        last_R0j=R0j
    J[3:,total_p-3:total_p] = last_R0j # p6T
    
    if unit!='radians':
        J[3:,(jN+1)*3:] = J[3:,(jN+1)*3:]*(np.pi/180)
        # J[3:,:] = J[3:,:]*(np.pi/180)
    
    return J

def get_PH_from_param_minimal(param,robot,unit='radians'):
    
    jN=len(robot.robot.H[0])
    # get current P,H (given param)
    P = []
    for oi in range(6):
        delta_o = param[2*oi]*robot.param_k1[oi]+param[2*oi+1]*robot.param_k2[oi]
        if oi==0:
            delta_P = deepcopy(delta_o)
        else:
            delta_P = delta_o-last_delta_o
        P.append(robot.P_nominal[oi]+delta_P)
        last_delta_o = deepcopy(delta_o)
    P.append(robot.P_nominal[-1]+(param[2*jN:2*jN+3]-last_delta_o))
    P=np.array(P)
    P=np.reshape(P,((jN+1),3))
    total_p = (jN)*2+3

    # get H
    param_h=deepcopy(param[total_p:])
    if unit!='radians':
        param_h = np.radians(param_h)
    H=[]
    rot_k1_alpha=[]
    rot_k2_beta=[]
    for j in range(jN):
        rot_k1_alpha.append(rot(robot.param_k1[j],param_h[2*j]))
        rot_k2_beta.append(rot(robot.param_k2[j],param_h[2*j+1]))
        hi = rot_k2_beta[-1]@\
           rot_k1_alpha[-1]@robot.H_nominal[j]
        H.append(hi)
    H=np.array(H)
    robot.robot.P=P.T
    robot.robot.H=H.T
    return robot

def get_param_from_PH_minimal(robot,this_P,this_H,nom_P,nom_H):

    param_P = []
    delta_oi = []
    residual_delta_oi = np.zeros(3)
    for i,p in enumerate(this_P[:,:-1].T):
        delta_p = p-nom_P[:,i]
        if i == 0:
            delta_oi_red = delta_p
        else:
            delta_oi_red = delta_p+delta_oi[-1]
        vi,wi,_ = np.linalg.inv(np.vstack((robot.param_k1[i],robot.param_k2[i],this_H[:,i])).T)@delta_oi_red
        # vi = np.dot(delta_oi_red,robot.param_k1[i])
        # wi = np.dot(delta_oi_red,robot.param_k2[i])
        # delta_oi.append(vi*robot.param_k1[i]+wi*robot.param_k2[i])
        residual_delta_oi = residual_delta_oi+delta_oi_red-(vi*robot.param_k1[i]+wi*robot.param_k2[i])
        delta_oi.append(delta_oi_red)
        param_P.append(vi)
        param_P.append(wi)
    param_P.extend(this_P[:,-1]-nom_P[:,-1]+delta_oi[-1])
    param_P = np.array(param_P)

    param_H = []
    for i,h in enumerate(this_H.T):
        theta_sol = subproblem2(nom_H[:,i], h, robot.param_k2[i], robot.param_k1[i])
        theta_sol = theta_sol[0] if theta_sol[0][0]<np.pi/2 and theta_sol[0][0]>-np.pi/2 else theta_sol[1]
        param_H.extend(theta_sol[::-1])
    
    return param_P,np.array(param_H)

def jacobian_param_minimal(param,robot,theta,unit='radians'):

    jN=len(theta)
    
    # get current P,H (given param)
    P = []
    for oi in range(6):
        delta_o = param[2*oi]*robot.param_k1[oi]+param[2*oi+1]*robot.param_k2[oi]
        if oi==0:
            delta_P = deepcopy(delta_o)
        else:
            delta_P = delta_o-last_delta_o
        P.append(robot.P_nominal[oi]+delta_P)
        last_delta_o = deepcopy(delta_o)
    P.append(robot.P_nominal[-1]+(param[2*jN:2*jN+3]-last_delta_o))
    P=np.array(P)
    P=np.reshape(P,((jN+1),3))
    total_p = (jN)*2+3
    param_h=param[total_p:]
    if unit!='radians':
        param_h = np.radians(param_h)
    H=[]
    rot_k1_alpha=[]
    rot_k2_beta=[]
    for j in range(jN):
        rot_k1_alpha.append(rot(robot.param_k1[j],param_h[2*j]))
        rot_k2_beta.append(rot(robot.param_k2[j],param_h[2*j+1]))
        hi = rot_k2_beta[-1]@\
           rot_k1_alpha[-1]@robot.H_nominal[j]
        H.append(hi)
    H=np.array(H)
    
    robot.robot.P = P.T
    robot.robot.H = H.T
    Pn=deepcopy(robot.P_nominal)
    Hn=deepcopy(robot.H_nominal)
    
    # foward kinematics
    T_tool = robot.fwd(theta)
    p0T=T_tool.p
    R0T=T_tool.R
    
    # get jacobian
    J=np.zeros((6,len(param))) # J=[JR;Jp]
    last_R0j=np.eye(3)
    p0j=np.zeros(3)
    for j in range(jN):
        # last_R0j: R_{0,j-1}
        Rj1j=rot(H[j],theta[j]) # R_{j-1,j}
        R0j=last_R0j@Rj1j # R_{0,j}
        RjT = R0j.T@R0T # R_{j,T}
        p0j=p0j+last_R0j@P[j] # p0j_0
        pjT_j=(R0j.T)@(p0T-p0j)
        # gradient of P w.r.t p0T
        J[3:,2*j:2*(j+1)]=last_R0j@(np.eye(3)-Rj1j)@(np.vstack((robot.param_k1[j],robot.param_k2[j])).T) # delta oi move only in the direction orthogonal to Hn (a plane)
        
        # gradient of R0T,P0T w.r.t alpha  
        dhda = rot_k2_beta[j]@hat(robot.param_k1[j])@rot_k1_alpha[j]@Hn[j]
        dRda = np.sin(theta[j])*hat(dhda)+(1-np.cos(theta[j]))*(hat(dhda)@hat(H[j])+hat(H[j])@hat(dhda))
        dR0jda = last_R0j@dRda
        J[:3,total_p+2*j]=invhat(dR0jda@RjT@(R0T.T))
        J[3:,total_p+2*j]=dR0jda@pjT_j
        # gradient of R0T,P0T w.r.t beta
        dhdb = hat(robot.param_k2[j])@rot_k2_beta[j]@rot_k1_alpha[j]@Hn[j]
        dRdb = np.sin(theta[j])*hat(dhdb)+(1-np.cos(theta[j]))*(hat(dhdb)@hat(H[j])+hat(H[j])@hat(dhdb))
        dR0jdb = last_R0j@dRdb
        J[:3,total_p+2*j+1]=invhat(dR0jdb@RjT@(R0T.T))
        J[3:,total_p+2*j+1]=dR0jdb@pjT_j
        last_R0j=R0j

    J[3:,total_p-3:total_p] = last_R0j # p6T
    
    if unit!='radians':
        J[3:,total_p:] = J[3:,total_p:]*(np.pi/180)
        # J[3:,:] = J[3:,:]*(np.pi/180)
    
    return J

def minimal_test(robot):

    # get orthogonal axis of each joint
    robot = get_H_param_axis(robot)

    param_p,param_H = get_param_from_PH_minimal(robot,robot.calib_P,robot.calib_H,robot.P_nominal.T,robot.H_nominal.T)
    param_min = np.append(param_p,param_H)

    from numpy.random import default_rng
    rng = default_rng()

    # test redundancy removal
    for i in range(100):
        q1 = rng.uniform(low=robot.lower_limit, high=robot.upper_limit)
        
        robot.robot.P = deepcopy(robot.calib_P)
        robot.robot.H = deepcopy(robot.calib_H)
        T_red = robot.fwd(q1)
        robot = get_PH_from_param_minimal(param_min,robot)
        T_min = robot.fwd(q1)
        
        TT_inv = T_red*(T_min.inv())
        if np.linalg.norm(H_from_RT(TT_inv.R,TT_inv.p)-np.eye(4),'fro')>1e-3:
            print(TT_inv)
            print("redundancy removal failed")
            exit()

    print("redundancy removal passed")

    # test jacobian
    q_test = np.radians([10,10,10,10,10,10])*3
    J_ana = jacobian_param_minimal(param_min,robot,q_test,unit='degrees')
    fig, ax = plt.subplots()
    im = ax.matshow(J_ana,cmap='RdBu')
    ax.tick_params(axis='both', labelsize=24)
    cbar = fig.colorbar(im)
    cbar.ax.tick_params(labelsize=24)
    plt.show()

def raw_test(robot):

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
    
    test_theta = np.radians([[10,10,10,10,10,10]])*3
    # test_theta = np.radians([[10]*link_N])*3
    # test_theta = np.radians([[0,0,0,0,0,10]])*3
    
    param = np.zeros(3*(jN+1)+2*jN)
    # param = np.random.rand(3*(jN+1)+2*jN)*0.01
    param[:3*(jN+1)] = np.reshape(robot.P_nominal,(3*(jN+1),))
    for test_th in test_theta:
        # analytical J
        J_ana = jacobian_param(param,robot,test_th,unit='degrees')
        fig, ax = plt.subplots()
        im = ax.matshow(J_ana,cmap='RdBu')
        ax.tick_params(axis='both', labelsize=24)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=24)
        plt.show()
        
        # # numerical J
        J_num = jacobian_param_numerical(param,robot,test_th,unit='degrees')
        fig, ax = plt.subplots()
        im = ax.matshow(J_num,cmap='RdBu')
        ax.tick_params(axis='both', labelsize=24)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=24)
        plt.show()
        
        J_diff = J_num-J_ana
        J_diff = np.fabs(J_diff)
        fig, ax = plt.subplots()
        im = ax.matshow(J_diff,cmap='RdBu')
        ax.tick_params(axis='both', labelsize=24)
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=24)
        plt.show()

def main():
    
    config_dir='../config/'
    # robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
    #                     tool_file_path=config_dir+'torch.csv',d=15,\
    #                     #  tool_file_path='',d=0,\
    #                     pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_marker_dir=config_dir+'MA2010_marker_config/'
    tool_marker_dir=config_dir+'weldgun_marker_config/'
    robot=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                        tool_file_path=config_dir+'torch.csv',d=15,\
                        #  tool_file_path='',d=0,\
                        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                        base_marker_config_file=robot_marker_dir+'MA2010_0801_marker_config.yaml',\
                        tool_marker_config_file=tool_marker_dir+'weldgun_0801_marker_config.yaml')
    
    robot.P_nominal=deepcopy(robot.robot.P)
    robot.H_nominal=deepcopy(robot.robot.H)
    robot.P_nominal=robot.P_nominal.T
    robot.H_nominal=robot.H_nominal.T

    # raw_test(robot)
    minimal_test(robot)

if __name__=='__main__':
    
    main()