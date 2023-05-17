import asyncio
from motoplus_rr_driver_command_client import MotoPlusRRDriverCommandClient, StreamingMotionTarget
import numpy as np
import traceback, socket, struct
import time

target = [
    StreamingMotionTarget(0, [50060,2080,1097,-1003,-7007,1502]),
    StreamingMotionTarget(1, [30000,2200,3300,4400,5500,6600]),
    StreamingMotionTarget(2, [8000,9000])
]

async def amain():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('0.0.0.0',11000))

        c = MotoPlusRRDriverCommandClient()
        c.start('192.168.1.31')        
        # c.start('127.0.0.1')
        await c.wait_ready(10)
        res = await c.get_controller_info()   

        await c.enable()

        info = await c.get_controller_info()
        state = await c.read_controller_state()

        rob1_pos = np.multiply(state.group_state[0].command_position, info.control_groups[0].pulse_to_radians)
        s1_pos = np.multiply(state.group_state[2].command_position, info.control_groups[2].pulse_to_radians)

        await c.start_motion_streaming()
        await asyncio.sleep(0.1)

        t1 = time.perf_counter()

        while True:
            t2 = time.perf_counter()
            dt = t2-t1
            if dt > 20:
                break

            rob1_target = rob1_pos + (np.array([0,0,0,0,0,1], dtype=np.float64) * 1000 * np.sin(dt))

            target = [
                StreamingMotionTarget(0, rob1_target),
            ]
            await c.update_motion_streaming_pulse_target(target)

            time.sleep(0.002)

            buf = s.recv(1024)
            data = struct.unpack("<34i",buf)
        

        await c.stop_motion_streaming()
            
    finally:
        c.stop()
        print("Program exit")
        print(data)

if __name__ == "__main__":
    asyncio.run(amain())