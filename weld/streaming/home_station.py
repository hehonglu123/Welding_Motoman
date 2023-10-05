import sys, glob, copy
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from StreamingSend import *

########################################################RR STREAMING########################################################
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


res, robot_state, _ = RR_robot_state.TryGetInValue()
q14=robot_state.joint_position
qd=copy.deepcopy(q14)
qd[-1]=0
point_distance=2
q_cmd=[]
q_record=[]
ts_record=[]

num_points_jogging=SS.streaming_rate*np.max(np.abs(q14-qd))/point_distance

try:
    for j in range(int(num_points_jogging)):
        q_target = (q14*(num_points_jogging-j))/num_points_jogging+qd*j/num_points_jogging
        ts,js=SS.position_cmd(q_target,time.time())
        q_cmd.append(q_target[-1])
        q_record.append(js[-1])
        ts_record.append(ts)
        
    ###init point wait
    for i in range(20):
        SS.position_cmd(qd,time.time())
except:
    np.savetxt('streaming_debug.csv',np.vstack((ts_record,q_record,q_cmd)).T,delimiter=',')