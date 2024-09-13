from RobotRaconteur.Client import *
import sys, time

url='rr+tcp://192.168.55.12:59945?service=robot'

robot_sub=RRN.SubscribeService(url)
robot=robot_sub.GetDefaultClientWait(1)
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", robot)
state_flags_enum = robot_const['RobotStateFlags']

state_w = robot_sub.SubscribeWire("robot_state")

while True:
	if state_w.TryGetInValue()[0]:
		flags_text = ""
		for flag_name, flag_code in state_flags_enum.items():
			if flag_code & state_w.InValue.robot_state_flags != 0:
				flags_text += flag_name + "\n"
		print(f"Robot state flags: {flags_text}")
	time.sleep(1)