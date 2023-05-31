# Read a frame from the scanner and plot

from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time, sys, pickle
import open3d as o3d

sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *

###MTI connect to RR
c = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
c.setExposureTime("25")
###Robot Initialization
robot2=robot_obj('MA1440_A0',def_path='../config/MA1440_A0_robot_default_config.yml',tool_file_path='../config/mti.csv',\
    pulse2deg_file_path='../config/MA1440_A0_pulse2deg_real.csv')

R=np.array([[0., -1,  0.    ],
            [-1, 0.,  0.    ],
            [0., 0., -1.    ]])
p_start=np.array([1000,400,-230])
p_end=np.array([1000,300,-230])
q_seed=np.zeros(6)
q_init=np.degrees(robot2.inv(p_start,R,q_seed)[0])
q_end=np.degrees(robot2.inv(p_end,R,q_seed)[0])

mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot2.pulse2deg,tool_num=10)
client=MotionProgramExecClient()

base_layer_height=2
layer_height=0.8


for i in range(0,1):
    mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot2.pulse2deg,tool_num=10)

    if i%2==0:
        p1=p_start+np.array([0,0,2*base_layer_height+i*layer_height])
        p2=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
    else:
        p1=p_end+np.array([0,0,2*base_layer_height+i*layer_height])
        p2=p_start+np.array([0,0,2*base_layer_height+i*layer_height])

    q_init=np.degrees(robot2.inv(p1,R,q_seed)[0])
    q_end=np.degrees(robot2.inv(p2,R,q_seed)[0])
    mp.MoveJ(q_init,1,0)
    client.execute_motion_program(mp)
    mp=MotionProgram(ROBOT_CHOICE='RB2',pulse2deg=robot2.pulse2deg,tool_num=10)
    mp.MoveL(q_end,12,0)
    client.execute_motion_program_nonblocking(mp)
    ###streaming
    client.StartStreaming()
    start_time=time.time()
    state_flag=0
    joint_recording=[]
    mti_recording=[]
    while True:
        if state_flag & 0x08 == 0 and time.time()-start_time>1.:
            break
        res, data = client.receive_from_robot(0.01)
        if res:
            joint_angle=np.radians(np.divide(np.array(data[26:32]),robot2.pulse2deg))
            state_flag=data[16]
            joint_recording.append(joint_angle)
            ###MTI scans YZ point from tool frame
            mti_recording.append(np.array([-c.lineProfile.X_data,c.lineProfile.Z_data]))

    client.servoMH(False)

mti_recording=np.array(mti_recording)

pc=[]
for i in range(len(mti_recording)):
    line_scan=np.vstack((np.zeros(len(mti_recording[i][0])),mti_recording[i])).T
    pose=robot2.fwd(joint_recording[i])
    pc.extend(transform_curve(line_scan,H_from_RT(pose.R,pose.p)))

pc=np.array(pc)
pointcloud=o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pointcloud])

# ax = plt.figure().add_subplot(projection='3d')
# ax.scatter(pc[:,0],pc[:,1],pc[:,2])
# plt.show()


with open('mti_recording.pickle', 'wb') as file:
    pickle.dump(mti_recording, file)
np.savetxt('joint_recording.csv',joint_recording,delimiter=',')
