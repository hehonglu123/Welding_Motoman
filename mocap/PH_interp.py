from typing import Any
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator,RBFInterpolator
from copy import deepcopy

class RBFFourierInterpolator(object):
    def __init__(self,train_q,value,basis_function_num=2) -> None:
        
        self.train_x = np.array(train_q)
        self.train_y = np.array(value)
        self.basis_function_num = basis_function_num
        self.train()
        
    def __call__(self, q):
        
        return self.predict(q)
    
    def train(self):
        
        basis_func_q2q3=[]
        for q in self.train_x:
            this_basis = self.build_basis(q)
            basis_func_q2q3.append(this_basis)
        
        basis_func_q2q3=np.array(basis_func_q2q3).T
        
        self.coeff_A = self.train_y@np.linalg.pinv(basis_func_q2q3)
    
    def predict(self,test_x):
        
        basis_func_q2q3=[]
        for q in test_x:
            this_basis = self.build_basis(q)
            basis_func_q2q3.append(this_basis)
        basis_func_q2q3=np.array(basis_func_q2q3).T
        
        # if test_x not in self.train_x:
        #     return np.nan
        return self.coeff_A@basis_func_q2q3
    
    def build_basis(self,q):
        
        ###### define basis function ######
        basis_func=[]
        basis_func.append(lambda q2,q3,a: np.sin(a*q2))
        basis_func.append(lambda q2,q3,a: np.sin(a*q3))
        basis_func.append(lambda q2,q3,a: np.cos(a*q2))
        basis_func.append(lambda q2,q3,a: np.cos(a*q3))
        basis_func.append(lambda q2,q3,a: np.sin(a*(q2+q3)))
        basis_func.append(lambda q2,q3,a: np.cos(a*(q2+q3)))
        ###################################
        
        this_basis = []
        for a in range(1,self.basis_function_num+1):
            for func in basis_func:
                this_basis.append(func(q[0],q[1],a))
        this_basis.append(1) # constant function
        
        return this_basis

class PH_Param(object):
    def __init__(self,nom_P,nom_H) -> None:
        
        self.method=None
        self.data=None
        self.train_q=None
        self.predict_func=None
        self.nom_P = deepcopy(nom_P)
        self.nom_H = deepcopy(nom_H)

    def fit(self,data,method='nearest',useHRotation=False,useMinimal=False):

        train_q = []
        for qkey in data.keys():
            train_q.append(np.array(qkey))
        train_q=np.array(train_q)
        self.train_q=train_q
        self.method=method
        self.data=data
        self.useHRotation=(useHRotation or useMinimal)
        self.useMinimal=useMinimal

        if method=='nearest':
            self.predict_func=self._predict_nearest
        elif method=='linear':
            self._fit_interp(LinearNDInterpolator)
            self.predict_func=self._predict_interp
        elif method=='cubic':
            self._fit_interp(CloughTocher2DInterpolator)
            self.predict_func=self._predict_interp
        elif method=='RBF':
            self._fit_interp(RBFInterpolator)
            self.predict_func=self._predict_interp
        elif method=='FBF':
            self._fit_interp(RBFFourierInterpolator)
            self.predict_func=self._predict_interp
        elif method=='CPA':
            self.predict_func=self._predict_interp
        else:
            print("Choose a method")
            self.train_q=None
            self.method=None
            return False

    def _fit_interp(self,interp_func):
        
        value_P=[]
        value_H=[]
        if self.useMinimal:
            for i in range(6*2+3):
                value_P.append([])
        else:
            for i in range(7*3):
                value_P.append([])
        if self.useHRotation:
            for i in range(6*2):
                value_H.append([])
        else:
            for i in range(6*3):
                value_H.append([])

        for q in self.train_q:
            i=0
            if self.useMinimal:
                for P in self.data[tuple(q)]['P']:
                    value_P[i].append(P)
                    i+=1
            else:
                for P in self.data[tuple(q)]['P']:
                    diff_P = P-self.nom_P[round(i/7.)]
                    for qi in diff_P:
                        value_P[i].append(qi)
                        i+=1
            i=0
            if self.useHRotation:
                for H in self.data[tuple(q)]['H']:
                    value_H[i].append(H)
                    i+=1
            else:
                for H in self.data[tuple(q)]['H']:
                    diff_H = H-self.nom_H[round(i/7.)]
                    for qi in diff_H:
                        value_H[i].append(qi)
                        i+=1
        ## fitting function
        fit_P=[]
        fit_H=[]
        for val_p in value_P:
            fit_P.append(interp_func(self.train_q, val_p))
        for val_h in value_H:
            fit_H.append(interp_func(self.train_q, val_h))
        self.fit_P=fit_P
        self.fit_H=fit_H

    def predict(self,q2q3):
        
        q2q3=np.array(q2q3)
        P,H = self.predict_func(q2q3)

        if not self.useMinimal:
            opt_P=P+self.nom_P
        else:
            opt_P=P

        if not self.useHRotation:
            opt_H=H+self.nom_H
            for j in range(len(opt_H[0])):
                opt_H[:,j]=opt_H[:,j]/np.linalg.norm(opt_H[:,j])
        else:
            opt_H=H

        return opt_P,opt_H

    def _predict_interp(self,q2q3):

        if self.useMinimal:
            opt_P = np.zeros(6*2+3,dtype=float)
            opt_H = np.zeros(6*2,dtype=float)
        elif self.useHRotation:
            opt_P = np.zeros_like(self.nom_P,dtype=float)
            opt_H = np.zeros(6*2,dtype=float)
        if not self.useHRotation:
            opt_P,opt_H = deepcopy(self._predict_nearest(q2q3))
        
        if np.isnan(self.fit_P[0]([[q2q3[0],q2q3[1]]])):
            # if outside training data, use nearest neighbor
            pass
        else:
            if self.useMinimal:
                for i in range(len(opt_P)):
                    opt_P[i]=self.fit_P[i]([[q2q3[0],q2q3[1]]])
            else:
                for i in range(len(opt_P)):
                    for j in range(len(opt_P[i])):
                        opt_P[i][j]=self.fit_P[i*len(opt_P[i])+j]([[q2q3[0],q2q3[1]]])
            if self.useHRotation:
                for i in range(len(opt_H)):
                    opt_H[i]=self.fit_H[i]([[q2q3[0],q2q3[1]]])
            else:
                for i in range(len(opt_H)):
                    for j in range(len(opt_H[i])):
                        opt_H[i][j]=self.fit_H[i*len(opt_H[i])+j]([[q2q3[0],q2q3[1]]])
            # for j in range(len(opt_H[0])):
            #     opt_H[:,j]=opt_H[:,j]/np.linalg.norm(opt_H[:,j])
            # print("P",opt_P.T)
            # print("H",opt_H.T)
        
        # print("====================")
        return opt_P,opt_H

    def _predict_nearest(self,q2q3):
        
        train_q_index = np.argmin(np.linalg.norm(self.train_q-q2q3,ord=2,axis=1))
        train_q_key = tuple(self.train_q[train_q_index])

        return self.data[train_q_key]['P']-self.nom_P,self.data[train_q_key]['H']-self.nom_H
    
    def get_basis_weights(self):
        if self.method=='FBF':
            P_coeff=[]
            for fit_p in self.fit_P:
                P_coeff.append(fit_p.coeff_A)
            H_coeff=[]
            for fit_h in self.fit_H:
                H_coeff.append(fit_h.coeff_A)
            return P_coeff,H_coeff
        else:
            return None, None

    def compare_nominal(self,nom_P,nom_H):

        X = np.linspace(min(self.train_q[:,0]), max(self.train_q[:,0]),1000)
        Y = np.linspace(min(self.train_q[:,1]), max(self.train_q[:,1]),1000)
        X, Y = np.meshgrid(X, Y)
        # print(X)

        # plot P
        print("**P Variation**")
        markdown_str=''
        markdown_str+='||Mean (mm)|Std (mm)|\n'
        markdown_str+='|-|-|-|\n'
        draw_mean=[]
        draw_std=[]
        fig,axs = plt.subplots(2,4)
        for i in range(len(nom_P[0])):
            p_dist = []
            for q in self.train_q:
                p_dist.append(np.linalg.norm(self.data[tuple(q)]['P'][:,i]-nom_P[:,i]))
            # use linear interp to plot
            interp = LinearNDInterpolator(self.train_q, p_dist)
            Z = interp(X, Y)
            
            # interp = RBFFourierInterpolator(self.train_q, p_dist)
            
            im=axs[int(i/4),int(i%4)].pcolormesh(np.degrees(X), np.degrees(Y), Z, shading='auto')
            plt.colorbar(im,ax=axs[int(i/4),int(i%4)])
            axs[int(i/4),int(i%4)].plot(np.degrees(self.train_q[:,0]), np.degrees(self.train_q[:,1]), "ok",ms=3, label="Training Poses (q2q3)")
            axs[int(i/4),int(i%4)].set_xlabel('q2 (deg)')
            axs[int(i/4),int(i%4)].set_ylabel('q3 (deg)')
            axs[int(i/4),int(i%4)].set_title("Distance to Nominal (mm), P"+str(i+1))
            axs[int(i/4),int(i%4)].legend(loc="upper left")

            markdown_str+='|P'+str(i+1)+'|'+format(round(np.mean(p_dist),4),'.4f')+'|'+format(round(np.std(p_dist),4),'.4f')+'|\n'
            draw_mean.append(np.mean(p_dist))
            draw_std.append(np.std(p_dist))
        # plt.legend()
        # plt.colorbar()
        # plt.axis("equal")
        # plt.title("Deviated Angle to Nominal (deg) of Rotation Axis")
        fig.suptitle("Deviated Distance to Nominal Position Vector P (mm)",fontsize=16)
        plt.show()
        
        print(markdown_str)
        plt.errorbar(np.arange(len(draw_mean)),draw_mean,draw_std)
        plt.xticks(np.arange(len(draw_mean)),['P1','P2','P3','P4','P5','P6','P7'],fontsize=15)
        plt.ylabel("Deviation Distance (mm)",fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('P Variation',fontsize=18)
        plt.show()

        # plot H
        print("**H Variation**")
        markdown_str=''
        markdown_str+='||Mean (deg)|Std (deg)|\n'
        markdown_str+='|-|-|-|\n'
        draw_mean=[]
        draw_std=[]
        fig,axs = plt.subplots(2,3)
        for i in range(len(nom_H[0])):
            h_ang = []
            for q in self.train_q:
                opt_H = self.data[tuple(q)]['H'][:,i]/np.linalg.norm(self.data[tuple(q)]['H'][:,i])
                ori_H = nom_H[:,i]/np.linalg.norm(nom_H[:,i])
                costh = np.dot(opt_H,ori_H)
                sinth = np.linalg.norm(np.cross(opt_H,ori_H))
                th = np.arctan2(sinth,costh)
                h_ang.append(np.degrees(th))
            # use linear interp to plot
            interp = LinearNDInterpolator(self.train_q, h_ang)
            Z = interp(X, Y)
            # interp = RBFFourierInterpolator(self.train_q, h_ang)
            
            im=axs[int(i/3),int(i%3)].pcolormesh(np.degrees(X), np.degrees(Y), Z, shading='auto')
            plt.colorbar(im,ax=axs[int(i/3),int(i%3)])
            axs[int(i/3),int(i%3)].plot(np.degrees(self.train_q[:,0]), np.degrees(self.train_q[:,1]), "ok",ms=3, label="Training Poses (q2q3)")
            axs[int(i/3),int(i%3)].set_xlabel('q2 (deg)')
            axs[int(i/3),int(i%3)].set_ylabel('q3 (deg)')
            axs[int(i/3),int(i%3)].set_title("Angle to Nominal (deg), H"+str(i+1))
            axs[int(i/3),int(i%3)].legend(loc="upper left")

            markdown_str+='|H'+str(i+1)+'|'+format(round(np.mean(h_ang),4),'.4f')+'|'+format(round(np.std(h_ang),4),'.4f')+'|\n'
            draw_mean.append(np.mean(h_ang))
            draw_std.append(np.std(h_ang))
            
        # plt.legend()
        # plt.colorbar()
        # plt.axis("equal")
        # plt.title("Deviated Angle to Nominal (deg) of Rotation Axis")
        fig.suptitle("Deviated Angle to Nominal Rotation Axis H (deg)",fontsize=16)
        plt.show()
        
        print(markdown_str)
        plt.errorbar(np.arange(len(draw_mean)),draw_mean,draw_std)
        plt.xticks(np.arange(len(draw_mean)),['H1','H2','H3','H4','H5','H6'],fontsize=15)
        plt.ylabel("Deviated Angle (deg)",fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('H Variation',fontsize=18)
        plt.show()

if __name__=='__main__':

    PH_data_dir='PH_grad_data/test0801_R1/train_data_'
    test_data_dir='kinematic_raw_data/test0801/'
    
    nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
    
    # PH_data_dir='PH_grad_data/test0804_R2/train_data_'
    # test_data_dir='kinematic_raw_data/test0801/'
    
    # nom_P=np.array([[0,0,0],[155,0,0],[0,0,614],\
    #                [640,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    # nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
    #             [-1,0,0],[0,-1,0],[-1,0,0]]).T

    import pickle
    with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
        PH_q=pickle.load(file)

    ph_param=PH_Param(nom_P,nom_H)
    ph_param.fit(PH_q,method='linear')

    ph_param.compare_nominal(nom_P,nom_H)