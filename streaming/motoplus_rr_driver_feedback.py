from enum import IntEnum, IntFlag
from typing import NamedTuple, List
import numpy as np
import struct
import socket
import asyncio
import traceback
import time
import io
from contextlib import suppress

RR_MOTOPLUS_FEEDBACK_SERVER_PORT=24582
RR_MOTOPLUS_FEEDBACK_MAGIC=0x46415252

class ControllerStateFlags(IntFlag):
    step = 0x1
    one_cycle = 0x2
    auto = 0x4
    running = 0x8
    hold = 0x10
    safety_speed = 0x20
    teach = 0x40
    play = 0x80
    remote = 0x100
    alarm = 0x200
    error = 0x400
    servo_power = 0x800
    api_error_robot = 0x4000
    api_error_status = 0x8000

class ControllerState(NamedTuple):
    version: int
    time: float
    seqno: int
    controller_flags: int
    group_state: List
    task_state: List
    motion_streaming_state: "MotionStreamingState"

class GroupStateFlags(IntFlag):
    pass

class GroupState(NamedTuple):
    axes_count: int
    group_flags: int
    command_position: np.array
    feedback_position: np.array
    command_speed: np.array
    feedback_speed: np.array
    torque: np.array

class TaskStateFlags(IntFlag):
    group_1_used = 0x01
    group_2_used = 0x02
    group_3_used = 0x04
    group_4_used = 0x08
    motion_program_running = 0x10
    motion_program_paused = 0x20
    streaming_running = 0x40
    error = 0x100

class TaskState(NamedTuple):
    task_flags: int
    queued_cmd_num: int
    completed_cmd_num: int
    current_buffer_num: int
    current_buffer_seqno: int
    last_completed_buffer_seqno: int

_task_struct = struct.Struct("<IIIIII")

def read_task_state(f):
    flags, queued_cmd_num, completed_cmd_num, current_buffer_num, current_buffer_seqno, \
        last_completed_buffer_seqno = _task_struct.unpack(f.read(6*4))
    
    return TaskState(flags, queued_cmd_num, completed_cmd_num, current_buffer_num, current_buffer_seqno, \
                     last_completed_buffer_seqno)

_uint32_struct = struct.Struct("<I")
_int32_struct = struct.Struct("<i")

def _read_axes(f,n):
    return np.frombuffer(f.read(n*4),dtype=np.int32).astype(np.float64)

def read_group_state(f, group_info):
    grp_flags = _uint32_struct.unpack(f.read(4))[0]
    n = grp_flags & 0x7

    p_to_rad = group_info.pulse_to_radians

    cmd_pos = np.divide(_read_axes(f,n), p_to_rad)
    fb_pos = np.divide(_read_axes(f,n), p_to_rad)
    cmd_spd = np.divide(_read_axes(f,n), p_to_rad)
    fb_spd = np.divide(_read_axes(f,n), p_to_rad)
    trq = np.divide(_read_axes(f,n), p_to_rad)

    return GroupState(n, grp_flags, cmd_pos, fb_pos, cmd_spd, fb_spd, trq)

class MotionStreamingStateFlags(IntFlag):
    joint_streaming_requested = 0x01
    joint_streaming_active = 0x02
    joint_streaming_error = 0x80

class MotionStreamingState(NamedTuple):
    streaming_flags: int
    streaming_error: int

_streaming_struct = struct.Struct("<II")
def read_motion_streaming_state(f):
    flags, err = _streaming_struct.unpack(f.read(2*4))
    return MotionStreamingState(flags,err)

_common_struct = struct.Struct("<IIIIIHHI")

def read_controller_state(f, controller_info):
    magic, version, clock_s, clock_nsec, seqno, num_grps, num_tasks, flags = _common_struct.unpack(f.read(7*4))

    assert magic == RR_MOTOPLUS_FEEDBACK_MAGIC, f"Invalid feedback magic {magic}"

    t = float(clock_s) + (float(clock_nsec)*1e-9)

    grp_states = []
    for i in range(num_grps):
        grp_states.append(read_group_state(f, controller_info.control_groups[i]))

    task_states = []
    for i in range(num_tasks):
        task_states.append(read_task_state(f))

    streaming_state = read_motion_streaming_state(f)

    return ControllerState(version, t, seqno, flags, grp_states, task_states, streaming_state)

class MotoPlusRRDriverFeedbackClient:
    def __init__(self, controller_info):
        self.s = None
        self._send_task = None
        self._keep_going = True
        self._loop = None
        self._recv_state = None
        self._recv_state_t = 0
        self._controller_info = controller_info
        self._cv = asyncio.Condition()

    def start(self, host, port = RR_MOTOPLUS_FEEDBACK_SERVER_PORT):
        self._loop = asyncio.get_event_loop()
        self._host = host
        self._port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(('',0))

        self._send_task = self._loop.create_task(self._send_task_fn())
        self._recv_task = self._loop.create_task(self._recv_task_fn())

    async def _send_task_fn(self):
        while self._keep_going:
            try:
                magic_bytes = _uint32_struct.pack(RR_MOTOPLUS_FEEDBACK_MAGIC)
                await self._loop.sock_sendto(self.s,magic_bytes, (self._host, self._port))
            except Exception:
                traceback.print_exc()
            await asyncio.sleep(10)

    async def _recv_task_fn(self):
        while self._keep_going:
            try:
                buf, addr = await self._loop.sock_recvfrom(self.s,12800)
                with io.BytesIO(buf) as f:
                    async with self._cv:
                        self._recv_state = read_controller_state(f,self._controller_info)
                        self._recv_state_t = time.perf_counter()
                        self._cv.notify_all()
            except:
                traceback.print_exc()
                await asyncio.sleep(1)

    def last_received_state(self):
        with self._cv:
            return self._recv_state, self._recv_state_t
        
    async def try_receive_state(self, timeout = 0.1, max_age = 0.05):
        t = time.perf_counter()
        async with self._cv:
            if (t - self._recv_state_t) < max_age:
                return True, self._recv_state
            try:
                await asyncio.wait_for(self._cv.wait_for(lambda: (t-self._recv_state_t) < max_age), timeout)
                return True, self._recv_state
            except asyncio.TimeoutError:
                return False, None
            
    def close(self):
        self._keep_going = False
        if self._send_task is not None:
            self._send_task.cancel()
        if self._recv_task is not None:
            self._recv_task.cancel()
        with suppress(Exception):
            self.s.close()
