import numpy as np
import matplotlib.pyplot as plt
import pickle

filename='IR_92023_05_12_11_43_51_ir_'
with open(filename+'recording.pickle', 'rb') as file:
    ir_recording=pickle.load(file)
timestamp=np.loadtxt(filename+'ts.csv',delimiter=',')
for i in range(len(ir_recording)):
    max_temp=np.max(ir_recording[i])
np.savetxt('max_temp'+filename+'.csv',np.vstack((timestamp,max_temp)).T)
