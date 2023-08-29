import sys, traceback
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import *
from multi_robot import *
from dx200_motion_program_exec_client import *
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

###JOG TO starting pose first
res, robot_state, _ = SS.RR_robot_state.TryGetInValue()
q_cur=robot_state.joint_position
num_points_jogging=SS.streaming_rate*np.max(np.abs(q_cur[-1]))/1

ts_all=[]
js_all=[]
try:
    for i in range(num_points_jogging):
        q_target = (q_cur*(num_points_jogging-i))/num_points_jogging
        ts,js = SS.position_cmd(np.append(q_cur[:-1],q_target),time.time())
        ts_all.append(ts)
        js_all.append(js[-1])
except:
    traceback.print_exc()

finally:
    np.savetxt('recorded_data.csv',np.vstack((ts_all,js_all)).T,delimiter=',')

