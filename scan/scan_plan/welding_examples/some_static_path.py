import sys
import numpy as np

sys.path.append('../../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

client=MotionProgramExecClient(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=[1.435355447016790322e+03,1.300329111270902331e+03,1.422225409601069941e+03,9.699560942607320158e+02,9.802408285708806943e+02,4.547552630640436178e+02],pulse2deg_2=[1994.3054,1376.714])
############## Angles Rules ###################
# q1 = 实际 q1
# q2 = - (实际 q2)
# q3 = 实际q3
# q4 = - (实际 q4)
# q5 = 实际 q5
# q6 = - (实际 q6)
###############################################
q1=np.array([43.9704, 69.2684, 39.7750, - 78.7448, 25.0877, 88.1245])
q2=np.array([28.7058, 52.7440, 0.3853, 58.5666, 89.9525, -30.5505])
q3=np.array([32.5383, 52.5302, 6.9630, 65.8277, 70.3430, -37.1343])
q4=np.array([34.7476, 55.2934, 12.0979, 64.8555, 67.9181, -38.7352])
q5=np.array([37.9175, 58.6948, 21.4432, 75.3962, 54.2387, -49.0660])

q6=np.array([41.2526, 67.3697, 36.1898, 91.7557, 44.0331, -65.4353])
q7=np.array([38.5487, 60.8028, 23.8661, 87.5627, 46.3131, -57.3386])
q8=np.array([31.9357, 54.9688, 6.2205, 58.3067, 82.7573, -32.7165])
q9=np.array([26.9181, 50.9167, -2.2788, 61.2708, 89.0354, -30.4098])


target2J_1=['MOVJ',[-15,180],3,0]
target2J_2=['MOVJ',[-15,220],3,0]
target2J_3=['MOVJ',[-15,260],3,0]
target2J_4=['MOVJ',[-15,300],3,0]
target2J_5=['MOVJ',[-15,340],3,0]

# client.MoveJ(q1, 3, 0)
client.MoveJ(q6, 3, 0)
client.MoveJ(q7, 3, 0)
client.MoveJ(q8, 3, 0)
client.MoveJ(q9, 3, 0)
client.MoveJ(q2, 8, 0)
client.MoveJ(q3, 6, 0)
client.MoveJ(q4, 8, 0)
client.MoveJ(q5, 6, 0)
client.MoveJ(q4, 10, 0, target2=target2J_1)
client.MoveJ(q3, 10, 0, target2=target2J_2)
client.MoveJ(q2, 10, 0, target2=target2J_3)
client.MoveJ(q4, 10, 0, target2=target2J_4)
client.MoveJ(q2, 10, 0, target2=target2J_5)
client.MoveJ(q6, 3, 0)
client.MoveJ(q7, 3, 0)
client.MoveJ(q8, 3, 0)
client.MoveJ(q9, 3, 0)
client.MoveJ(q2, 8, 0)
client.MoveJ(q3, 6, 0)
client.MoveJ(q4, 8, 0)
client.MoveJ(q5, 6, 0)
client.MoveJ(q3, 15, 0, target2=target2J_1)
# client.MoveJ(q7, 3, 1)
# client.MoveJ(q8, 3, 1)
# client.MoveJ(q9, 3, 0)
# client.MoveL(q2, 10, 0,target2=target2J_2)
# client.MoveL(q3, 10, 0,target2=target2J_3)
# client.MoveL(q2, 10, 0, target2=target2J_4)
# client.MoveL(q1, 10, 0, target2=target2J_5)

client.ProgEnd()
print(client.execute_motion_program("AAA.JBI"))