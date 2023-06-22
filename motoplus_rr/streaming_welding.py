import asyncio
from motoplus_rr_driver_command_client import MotoPlusRRDriverCommandClient, StreamingMotionTarget
import numpy as np
import traceback
import time, sys
from RobotRaconteur.Client import *
import time
sys.path.append('../toolbox/')
from robots_def import *


# Adjust the connection URL to the driver
fronius_client = RRN.ConnectService('rr+tcp://localhost:60823?service=welder')

# Set the job number to use for this weld
fronius_client.job_number = 215
fronius_client.prepare_welder()

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])
p_start=np.array([1610,-860,-260])
p_end=np.array([1610,-760,-260])
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

    v=5
    num_points=np.ceil(250*np.linalg.norm(p1-p2)/v)
    q_all=np.zeros((num_points,6))
    for j in range(int(num_points)):
        q_all[j]=robot.inv(p1*(num_points-j)/num_points+p2*j/num_points,R,q_seed)[0]
        
async def amain():
    try:
        c = MotoPlusRRDriverCommandClient()
        c.start('192.168.55.1')        
        # c.start('127.0.0.1')
        await c.wait_ready(10)
        res = await c.get_controller_info()   
        print(res)

        await c.enable()

        info = await c.get_controller_info()
        state = await c.read_controller_state()

        rob1_pos = np.multiply(state.group_state[0].command_position, info.control_groups[0].pulse_to_radians)

        await c.start_motion_streaming()
        await asyncio.sleep(0.1)

        t1 = time.perf_counter()   

        ###JOG TO starting pose first
        num_points_jogging=np.linalg.norm(rob1_pos-q_all[0])/0.1    
        for j in range(int(num_points_jogging)):

            rob1_target = rob1_pos*(num_points-j)+q_all[0]*j/num_points

            target = [
                StreamingMotionTarget(0, rob1_target)
            ]
            await c.update_motion_streaming_pulse_target(target)


        ###start welding
        fronius_client.start_weld()
        for j in range(int(num_points)):

            rob1_target = q_all[j]  ###modify units here

            target = [
                StreamingMotionTarget(0, rob1_target)
            ]
            await c.update_motion_streaming_pulse_target(target)
        await c.stop_motion_streaming()
        fronius_client.stop_weld() 
            
    finally:
        c.stop()
        fronius_client.stop_weld()
        print("Program exit")

if __name__ == "__main__":
    asyncio.run(amain())