import time
from RobotRaconteur.Client import *
import numpy as np

rate = RRN.CreateRate(125)

time_recorded=[]
for i in range(1000):
	now=time.perf_counter()
	# time.sleep(0.001)
	# rate.Sleep()
	while time.perf_counter()-now<0.0079:
		time.sleep(0)
		continue
	
	# time_recorded.append(time.perf_counter()-now)
	print(time.perf_counter()-now)

# print(time_recorded)
# print(len(time_recorded),np.average(time_recorded))