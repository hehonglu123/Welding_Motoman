import sys, time, os, copy
import numpy as np
from RobotRaconteur.Client import *

def my_handler(exp):
	if (exp is not None):
		# If "err" is not None it means that an exception occurred.
		# "err" contains the exception object
		print ("An error occured! " + str(exp))
		return

fronius_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:60823?service=welder')
fronius_client = fronius_sub.GetDefaultClientWait(1)      #connect, timeout=30s

time_setting=[]
for i in range(100):
    now=time.perf_counter()
    fronius_client.job_number=460
    print(time.perf_counter()-now)
    time_setting.append(time.perf_counter()-now)
    

print('average time:', np.mean(time_setting))