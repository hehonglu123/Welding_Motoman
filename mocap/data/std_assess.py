import numpy as np
from general_robotics_toolbox import *

marker_N = [5,4,3]
repeat_N = 5
q_N = 2

for qN in range(q_N):
    for mN in marker_N:
        tool_p = []
        tool_rpy = []
        tool_p_N0 = []
        tool_rpy_N0 = []
        for rN in range(repeat_N):
            tool_T = np.loadtxt('tool_mocap_'+str(mN)+'_q'+str(qN)+'_N'+str(rN)+'.csv',delimiter=',')

            # for i in range(len(tool_T)):
            #     tool_p.append(tool_T[i,:3])
            #     tool_rpy.append(R2rpy(q2R(tool_T[i,3:])))
            tool_p.append(tool_T[-1,:3])
            tool_rpy.append(R2rpy(q2R(tool_T[-1,3:])))

            if rN==0:
                for i in range(len(tool_T)):
                    tool_p_N0.append(tool_T[i,:3])
                    tool_rpy_N0.append(R2rpy(q2R(tool_T[i,3:])))
            
        print("qN, Marker N:",qN,mN)
        print("Position Vec std",np.std(tool_p,axis=0))
        print("Position std",np.std(np.linalg.norm(tool_p-np.mean(tool_p,axis=0),2,axis=1),axis=0))
        print("Orientation Vec std",np.std(tool_rpy,axis=0))
        print("Orientation std",np.std(np.linalg.norm(tool_rpy-np.mean(tool_rpy,axis=0),2,axis=1),axis=0))

exit()
# base
base_T = np.loadtxt('base_mocap_'+str(mN)+'.csv',delimiter=',')

base_p = []
base_rpy = []

for i in range(len(base_T)):
    base_p.append(base_T[i,:3])
    base_rpy.append(R2rpy(q2R(base_T[i,3:])))

print("Base:")
print("Position Vec std",np.std(base_p,axis=0))
print("Position std",np.std(np.linalg.norm(base_p-np.mean(base_p,axis=0),2,axis=1),axis=0))
print("Orientation Vec std",np.std(base_rpy,axis=0))
print("Orientation std",np.std(np.linalg.norm(base_rpy-np.mean(base_rpy,axis=0),2,axis=1),axis=0))