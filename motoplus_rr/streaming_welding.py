import asyncio
from motoplus_rr_driver_command_client import MotoPlusRRDriverCommandClient, StreamingMotionTarget
import numpy as np
import traceback
import time, sys
from RobotRaconteur.Client import *
import time
sys.path.append('../toolbox/')
from robot_def import *
import matplotlib.pyplot as plt
from motoplus_rr_driver_feedback import MotoPlusRRDriverFeedbackClient

# Adjust the connection URL to the driver
# fronius_client = RRN.ConnectService('rr+tcp://localhost:60823?service=welder')

# # Set the job number to use for this weld
# fronius_client.job_number = 215
# fronius_client.prepare_welder()

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
p_start=np.array([1630,-860,-260])
p_end=np.array([1630,-760,-260])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
base_layer_height=2
layer_height=1.0

for i in range(0,1):
    if i%2==0:
        p1=p_start+np.array([0,0,i*base_layer_height])
        p2=p_end+np.array([0,0,i*base_layer_height])
    else:
        p1=p_end+np.array([0,0,i*base_layer_height])
        p2=p_start+np.array([0,0,i*base_layer_height])

    v=10
    num_points=np.ceil(250*np.linalg.norm(p1-p2)/v)
    q_all=np.zeros((int(num_points),6))
    for j in range(int(num_points)):
        q_all[j]=robot.inv(p1*(num_points-j)/num_points+p2*j/num_points,R,q_seed)[0]
        
async def amain():
    try:
        c = MotoPlusRRDriverCommandClient()
        c.start('192.168.1.31')        
        # c.start('127.0.0.1')
        await c.wait_ready(10)
        info = await c.get_controller_info()   
        print(info)
        f = MotoPlusRRDriverFeedbackClient(info)
        f.start('192.168.1.31')

        # try:
        #     c.disable()
        # except:
        #     pass

        await c.enable()
        await c.motion_streaming_clear_error()
        #info = await c.get_controller_info()
        state = await c.read_controller_state()

        rob1_pos = np.multiply(state.group_state[0].command_position, info.control_groups[0].pulse_to_radians)
        # print(state.group_state[0].command_position,info.control_groups[0].pulse_to_radians)
        print(rob1_pos)

        await c.start_motion_streaming()
        await asyncio.sleep(0.1)

        t1 = time.perf_counter()   

        ###JOG TO starting pose first
        num_points_jogging=np.linalg.norm(state.group_state[0].command_position-q_all[0])/0.001
        print(int(num_points_jogging))
        print(num_points_jogging/250)
        for j in range(int(num_points_jogging)):
            print(j)

            rob1_target = (rob1_pos*(num_points_jogging-j))/num_points_jogging+np.multiply(q_all[0],info.control_groups[0].pulse_to_radians)*j/num_points_jogging
            print(rob1_target)
            
            target = [
                StreamingMotionTarget(0, rob1_target,max_pulse_error=[1000,1000,1000,1000,1000,1000])
            ]
            await c.update_motion_streaming_pulse_target(target)
            await asyncio.sleep(0.004)


        # return


        ###start welding
        joint_recording=[]
        timestamp=[]
        
        

        # fronius_client.start_weld()
        for j in range(int(num_points)):
            #state = await c.read_controller_state()
            state=None
            while True:
                res, state1 = await f.try_receive_state(5)
                if res:
                    state=state1
            
            if state:
                print(state.group_state[0].feedback_position)
                joint_recording.append(state.group_state[0].feedback_position)
                timestamp.append(time.time())

            rob1_target = np.multiply(q_all[j],info.control_groups[0].pulse_to_radians)  ###modify units here

            target = [
                StreamingMotionTarget(0, rob1_target,max_pulse_error=[1000,1000,1000,1000,1000,1000])
            ]
            #await c.update_motion_streaming_pulse_target(target)
            #await asyncio.sleep(0.004)

        await c.stop_motion_streaming()

        # fronius_client.stop_weld() 
            
    finally:
        c.stop()
        # fronius_client.stop_weld()
        print("Program exit")

        # pose_all=robot.fwd(np.array(joint_recording))
        # lam=calc_lam_cs(pose_all.p_all)
        # speed=np.diff(lam)/np.diff(timestamp)

if __name__ == "__main__":
    asyncio.run(amain())