import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from RobotRaconteur.Client import *
from motoman_def import *
from WeldSend import *
import open3d as o3d

############## Single scan test ################

# MTI connect to RR
mti_client = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
mti_client.setExposureTime("25")

line_scan = np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])

## remove all data with z < 40
line_scan = line_scan[:,line_scan[1]>40]

plt.plot(line_scan[0],line_scan[1])
plt.xlabel('X (mm)')
plt.ylabel('Z (mm)')
plt.show()

############## Continuous scan test using robot motion ################

## rr drivers and all other drivers
robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)

config_dir='../config/'
robot_scan=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'mti_tuned0719.csv',\
    base_transformation_file=config_dir+'MA1440_pose.csv',pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA1440_marker_config/MA1440_marker_config.yaml')
positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
    base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')

scan_speed = 10 # mm/s

## motion start
mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=robot_scan.pulse2deg,pulse2deg_2=positioner.pulse2deg)
# calibration motion
target2=['MOVJ',np.array([-15,0]),10]

# motion program
starting_q = np.radians([24.9,33.8,-22,0,-34.8,3])
ending_q = np.radians([25.0,37.45,-16,0,-37.2,3])
mp.MoveL(np.degrees(starting_q), 50, 0, target2=target2)
mp.MoveL(np.degrees(ending_q), scan_speed, 0, target2=target2)

# execute motion program
ws.client.execute_motion_program_nonblocking(mp)
###streaming
ws.client.StartStreaming()
start_time=time.time()
state_flag=0
joint_recording=[]
robot_stamps=[]
mti_recording=None
mti_recording=[]
r_pulse2deg = np.append(robot_scan.pulse2deg,positioner.pulse2deg)
while True:
    if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
        break 
    res, fb_data = ws.client.fb.try_receive_state_sync(ws.client.controller_info, 0.001)
    if res:
        joint_angle=np.hstack((fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
        state_flag=fb_data.controller_flags
        joint_recording.append(joint_angle)
        timestamp=fb_data.time
        robot_stamps.append(timestamp)
        ###MTI scans YZ point from tool frame
        try:
            mti_recording.append(deepcopy(np.array([mti_client.lineProfile.X_data,mti_client.lineProfile.Z_data])))
        except Exception as e:
            if not mti_break_flag:
                print(e)
            mti_break_flag=True
ws.client.servoMH(False)

# mti_recording=np.array(mti_recording)
q_out_exe=np.array(joint_recording)[:,:6]

pcd_all = o3d.geometry.PointCloud()
for i in range(len(mti_recording)):
    scanner_T = robot_scan.fwd(q_out_exe[i])

    ## remove all data with z < 40
    # scan_points = mti_recording[i][:,mti_recording[i][1]>40]
    
    scan_points = np.insert(mti_recording[i],1,np.zeros(len(mti_recording[i][0])),axis=0)
    scan_points[0]=scan_points[0]*-1 # reversed x-axis
    
    scan_points = scan_points.T
    scan_points = np.transpose(np.matmul(scanner_T.R,np.transpose(scan_points)))+scanner_T.p

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(scan_points)
    pcd_all+=pcd

# visualize pcd with frames
o3d.visualization.draw_geometries([pcd_all])