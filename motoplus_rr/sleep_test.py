import time
from RobotRaconteur.Client import *

rate = RRN.CreateRate(125)

while True:
	now=time.time()
	# time.sleep(0.001)
	rate.Sleep()

	print(time.time()-now)