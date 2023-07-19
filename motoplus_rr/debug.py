import sys, glob
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../toolbox/')
from StreamingSend import *



########################################################RR STREAMING########################################################

# RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.10:59945?service=robot')
RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.15:59945?service=robot')
RR_robot_state = RR_robot_sub.SubscribeWire('robot_state')
RR_robot = RR_robot_sub.GetDefaultClientWait(1)
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.reset_errors()
RR_robot.enable()
RR_robot.command_mode = halt_mode
time.sleep(0.1)
RR_robot.command_mode = position_mode
streaming_rate=125.
point_distance=0.04		###STREAMING POINT INTERPOLATED DISTANCE
SS=StreamingSend(RR_robot,RR_robot_state,RobotJointCommand,streaming_rate)

###########################################base layer welding############################################
while True:
    
    SS.jog2q(np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
    SS.jog2q(np.hstack((-0.5*np.ones(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
    