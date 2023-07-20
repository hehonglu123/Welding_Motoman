from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys

now=time.time()
def my_handler(sub, value, ts):
	global now
	# Handle new value
	print(time.time()-now)
	now=time.time()


RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')


RR_robot_state.WireValueChanged += my_handler
input('press enter to quit')