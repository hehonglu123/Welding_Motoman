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

if __name__=='__main__':

    PH_data_dir='PH_grad_data/test0516_R1/train_data_'
    test_data_dir='kinematic_raw_data/test0516/'

    import pickle
    with open(PH_data_dir+'calib_PH_q.pickle','rb') as file:
        PH_q=pickle.load(file)

    ph_param=PH_Param()
    ph_param.fit(PH_q,method='linear_interp')