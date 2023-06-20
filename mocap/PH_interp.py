import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator,RBFInterpolator
from copy import deepcopy

class PH_Param(object):
    def __init__(self) -> None:
        
        self.method=None
        self.data=None
        self.train_q=None
        self.predict_func=None

    def fit(self,data,method='nearest'):

        train_q = []
        for qkey in data.keys():
            train_q.append(np.array(qkey))
        train_q=np.array(train_q)
        self.train_q=train_q
        self.method=method
        self.data=data

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
        else:
            print("Choose a method")
            self.train_q=None
            self.method=None
            return False

    def _fit_interp(self,interp_func):
        
        value_P=[]
        value_H=[]
        for i in range(7*3):
            value_P.append([])
        for i in range(6*3):
            value_H.append([])

        for q in self.train_q:
            i=0
            for P in self.data[tuple(q)]['P']:
                for qi in P:
                    value_P[i].append(qi)
                    i+=1
            i=0
            for H in self.data[tuple(q)]['H']:
                for qi in H:
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

        return P,H

    def _predict_interp(self,q2q3):

        opt_P,opt_H = deepcopy(self._predict_nearest(q2q3))
        # print("P",opt_P.T)
        # print("H",opt_H.T)
        if np.isnan(self.fit_P[0]([[q2q3[0],q2q3[1]]])):
            # if outside training data, use nearest neighbor
            pass
        else:
            for i in range(len(opt_P)):
                for j in range(len(opt_P[i])):
                    opt_P[i][j]=self.fit_P[i*len(opt_P[i])+j]([[q2q3[0],q2q3[1]]])
            for i in range(len(opt_H)):
                for j in range(len(opt_H[i])):
                    opt_H[i][j]=self.fit_H[i*len(opt_H[i])+j]([[q2q3[0],q2q3[1]]])
            for j in range(len(opt_H[0])):
                opt_H[:,j]=opt_H[:,j]/np.linalg.norm(opt_H[:,j])
            # print("P",opt_P.T)
            # print("H",opt_H.T)
        
        # print("====================")
        return opt_P,opt_H

    def _predict_nearest(self,q2q3):
        
        train_q_index = np.argmin(np.linalg.norm(self.train_q-q2q3,ord=2,axis=1))
        train_q_key = tuple(self.train_q[train_q_index])

        return self.data[train_q_key]['P'],self.data[train_q_key]['H']

    def compare_nominal(self,nom_P,nom_H):

        X = np.linspace(min(self.train_q[:,0]), max(self.train_q[:,0]),1000)
        Y = np.linspace(min(self.train_q[:,1]), max(self.train_q[:,1]),1000)
        X, Y = np.meshgrid(X, Y)

        # plot P
        print("**P Variation**")
        markdown_str=''
        markdown_str+='||Mean (mm)|Std (mm)|\n'
        markdown_str+='|-|-|-|\n'
        for i in range(len(nom_P[0])):
            p_dist = []
            for q in self.train_q:
                p_dist.append(np.linalg.norm(self.data[tuple(q)]['P'][:,i]-nom_P[:,i]))
            # use linear interp to plot
            interp = LinearNDInterpolator(self.train_q, p_dist)
            
            Z = interp(X, Y)
            # plt.pcolormesh(np.degrees(X), np.degrees(Y), Z, shading='auto')
            # plt.plot(np.degrees(self.train_q[:,0]), np.degrees(self.train_q[:,1]), "ok",ms=3, label="Training Poses (q2q3)")
            # plt.legend()
            # plt.colorbar()
            # # plt.axis("equal")
            # plt.xlabel('q2 (deg)')
            # plt.ylabel('q3 (deg)')
            # plt.title("Distance to Nominal (mm), P"+str(i+1))
            # plt.show()

            markdown_str+='|P'+str(i+1)+'|'+format(round(np.mean(p_dist),4),'.4f')+'|'+format(round(np.std(p_dist),4),'.4f')+'|\n'
        print(markdown_str)

        # plot H
        print("**H Variation**")
        markdown_str=''
        markdown_str+='||Mean (mm)|Std (mm)|\n'
        markdown_str+='|-|-|-|\n'
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
            # plt.pcolormesh(np.degrees(X), np.degrees(Y), Z, shading='auto')
            # plt.plot(np.degrees(self.train_q[:,0]), np.degrees(self.train_q[:,1]), "ok",ms=3, label="Training Poses (q2q3)")
            # plt.legend()
            # plt.colorbar()
            # # plt.axis("equal")
            # plt.xlabel('q2 (deg)')
            # plt.ylabel('q3 (deg)')
            # plt.title("Angle to Nominal (deg), H"+str(i+1))
            # plt.show()

            markdown_str+='|H'+str(i+1)+'|'+format(round(np.mean(h_ang),4),'.4f')+'|'+format(round(np.std(h_ang),4),'.4f')+'|\n'
        print(markdown_str)

if __name__=='__main__':

    PH_data_dir='PH_grad_data/test0516_R1/train_data_'
    test_data_dir='kinematic_raw_data/test0516/'

    import pickle
    with open(PH_data_dir+'calib_PH_q_torch.pickle','rb') as file:
        PH_q=pickle.load(file)

    ph_param=PH_Param()
    ph_param.fit(PH_q,method='linear')

    nom_P=np.array([[0,0,0],[150,0,0],[0,0,760],\
                   [1082,0,200],[0,0,0],[0,0,0],[100,0,0]]).T
    nom_H=np.array([[0,0,1],[0,1,0],[0,-1,0],\
                   [-1,0,0],[0,-1,0],[-1,0,0]]).T
    ph_param.compare_nominal(nom_P,nom_H)