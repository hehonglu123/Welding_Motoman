from general_robotics_toolbox import *
from fanuc_motion_program_exec_client import *

# utool_T=Transform(wpr2R(np.radians([91.653,87.390,-88.948])),[160.094,-123.001,67.714])
utool_T=Transform(wpr2R(np.radians([-78.247,89.212,100.385])),[163.676,116.964,54.107])

flange_T=Transform(rot([0,0,1],np.pi),[0,0,0])
total_T=flange_T*utool_T
print(total_T)
print(total_T.R)
print(total_T.p)
tool_H=np.hstack((np.vstack((total_T.R,np.zeros(3))),np.append(total_T.p,1).reshape(4,1)))

# np.savetxt('ge_R1_tool.csv',tool_H,delimiter=',')
np.savetxt('ge_R2_tool.csv',tool_H,delimiter=',')