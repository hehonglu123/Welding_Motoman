import copy
from motoplus_rr_driver_client import MotoPlusRRDriverReverseSocketClient
from typing import NamedTuple, List, Tuple
import numpy as np
import io
import struct
from enum import IntEnum
from motoplus_rr_driver_feedback import read_controller_state, read_task_state

RR_MOTOPLUS_COMMAND_SERVER_PORT=24581
RR_MOTOPLUS_COMMAND_MAGIC=0x43415252

RR_MOTOPLUS_CMDOP_NOOP=0x0
RR_MOTOPLUS_CMDOP_CTRL_META=0x1
RR_MOTOPLUS_CMDOP_CTRLGRP_META=0x2
RR_MOTOPLUS_CMDOP_READ_STATE=0x3
RR_MOTOPLUS_CMDOP_READ_TASK_STATE=0x4
RR_MOTOPLUS_CMDOP_MP_START=0x8
RR_MOTOPLUS_CMDOP_MP_STOP=0x9
RR_MOTOPLUS_CMDOP_MP_PAUSE=0xA
RR_MOTOPLUS_CMDOP_MP_RESUME=0xB
RR_MOTOPLUS_CMDOP_MP_PREEMPT=0xC
RR_MOTOPLUS_CMDOP_SERVO_ON=0x10
RR_MOTOPLUS_CMDOP_SERVO_OFF=0x11
RR_MOTOPLUS_CMDOP_MS_ENABLE=0x1001
RR_MOTOPLUS_CMDOP_MS_DISABLE=0x1002
RR_MOTOPLUS_CMDOP_MS_UPDATE_TARGET=0x1003
RR_MOTOPLUS_CMDOP_MS_CLEAR_ERROR=0x100F
RR_MOTOPLUS_CMDOP_FILE_LOAD=0x2001
RR_MOTOPLUS_CMDOP_FILE_READ=0x2002
RR_MOTOPLUS_CMDOP_REMOTE_JOB_START=0x2100
RR_MOTOPLUS_CMDOP_REMOTE_JOB_HOLD=0x2101

RR_MOTOPLUS_META_CTRLGRP_JOINT_TYPE=0x1
RR_MOTOPLUS_META_CTRLGRP_PULSE_TO_RAD=0x2
RR_MOTOPLUS_META_CTRLGRP_PULSE_TO_M=0x3
RR_MOTOPLUS_META_CTRLGRP_PULSE_LIMITS=0x4
RR_MOTOPLUS_META_CTRLGRP_ANG_LIMITS=0x5
RR_MOTOPLUS_META_CTRLGRP_INC=0x6
RR_MOTOPLUS_META_CTRLGRP_DH=0x7

RR_MOTOPLUS_CMD_ERR_CTRLGRP=0x1
RR_MOTOPLUS_CMD_ERR_INVALID_OP=0x2

class ControllerFileNumber:
    RRJOB = 1
    SYSTEM = 2
    RBCALIB = 3
    TOOL = 4
    UFRAME = 5

class ControllerInfo(NamedTuple):
    version: Tuple = None
    control_group_count: int = 0
    interpolation_period: int = 0
    control_groups: List = []

class ControlGroupInfo(NamedTuple):
    group_number: int = 0
    group_id: int = 0
    axes_count: int = 0
    joint_type: np.array = None
    pulse_to_radians: np.array = None
    pulse_to_meters: np.array = None
    joint_limits_low: np.array = None
    joint_limits_high: np.array = None
    joint_angular_velocity: np.array = None
    max_increment: np.array = None
    dh_parameters: np.array = None

class StreamingMotionTarget(NamedTuple):
    group_number: int = 0
    target: np.array = None
    max_pulse_error: np.array = None

def _unpack_controller_info(res):
    version = (
        (res.param2 >> 16) & 0xFFFF,
        (res.param2 >> 8) & 0xFF,
        (res.param2 & 0xFF)
    )

    return ControllerInfo(version, res.param3, res.param4)

def _pulse_limits_to_radians(pulse_limits_low, pulse_limits_high, pulse_to_rad):
    joint_limits_low1 = np.divide(pulse_limits_low, pulse_to_rad, dtype=np.float64)
    joint_limits_high1 = np.divide(pulse_limits_high, pulse_to_rad, dtype=np.float64)
    joint_limits_low = np.copy(joint_limits_low1)
    joint_limits_high = np.copy(joint_limits_high1)
    swap_i = np.where(pulse_to_rad < 0)
    joint_limits_low[swap_i] = joint_limits_high1[swap_i]
    joint_limits_high[swap_i] = joint_limits_low1[swap_i]

    assert np.all(joint_limits_low < joint_limits_high)

    return joint_limits_low, joint_limits_high

def _unpack_group_info(res, payload):
    grp_num = res.index
    grp_id = res.param2
    num_j = res.param3

    joint_type = None
    pulse_to_rad = None
    pulse_to_m = None
    pulse_limits_low = None
    pulse_limits_high = None
    ang_vel = None
    max_inc = None
    dh_params = None

    header_struct = struct.Struct("<HH")

    f = io.BytesIO(payload)
    while f.tell() < len(payload):
        header_bytes = f.read(4)
        size, entry = header_struct.unpack(header_bytes)
        if entry == RR_MOTOPLUS_META_CTRLGRP_JOINT_TYPE:
            joint_type = np.frombuffer(f.read(size*4), dtype=np.uint32)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_PULSE_TO_RAD:
            pulse_to_rad = np.frombuffer(f.read(size*4), dtype=np.float64)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_PULSE_TO_M:
            pulse_to_m = np.frombuffer(f.read(size*4), dtype=np.float64)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_PULSE_LIMITS:
            pulse_limits_low = np.frombuffer(f.read(size*2), dtype=np.int32)
            pulse_limits_high = np.frombuffer(f.read(size*2), dtype=np.int32)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_ANG_LIMITS:
            ang_vel = np.frombuffer(f.read(size*4), dtype=np.float64)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_INC:
            max_inc = np.frombuffer(f.read(size*4), dtype=np.int32)
        elif entry == RR_MOTOPLUS_META_CTRLGRP_DH:
            dh_params = np.frombuffer(f.read(size*4*4), dtype=np.float32)
        else:
            f.read(size*4)

    return ControlGroupInfo(
        grp_num, 
        grp_id, 
        num_j, 
        joint_type, 
        pulse_to_rad.astype(np.float64),
        pulse_to_m.astype(np.float64),
        # TODO: Support prismatic axis type
        *_pulse_limits_to_radians(pulse_limits_low, pulse_limits_high, pulse_to_rad),
        ang_vel, 
        np.divide(max_inc, pulse_to_rad,dtype=np.float64),
        dh_params
    )
class MotoPlusRRDriverCommandClient(MotoPlusRRDriverReverseSocketClient):

    def __init__(self):
        super().__init__(RR_MOTOPLUS_COMMAND_MAGIC)

    def start(self, host, port = RR_MOTOPLUS_COMMAND_SERVER_PORT):
        return super().start(host, port)
    
    def start_reverse(self, port = RR_MOTOPLUS_COMMAND_SERVER_PORT):
        return super().start_reverse(port)
    
    def _res_check_error(self, res):
        if res.param1 != 0:
            raise Exception(f"Command returned error: {res.param1}")
    
    async def get_controller_info(self):
        
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_CTRL_META, 0)
        self._res_check_error(res)
        
        ctrl_info1 = _unpack_controller_info(res)

        grp_info1 = []

        for i in range(ctrl_info1.control_group_count):
            res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_CTRLGRP_META, i)
            grp_info1.append(_unpack_group_info(res,payload))

        return ControllerInfo(ctrl_info1.version, ctrl_info1.control_group_count, ctrl_info1.interpolation_period, \
                grp_info1)
    
    async def start_motion_program(self, master_grp_no, task_no, buffer_index, ctrl_grps):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MP_START, master_grp_no, task_no, buffer_index, ctrl_grps)
        self._res_check_error(res)

    async def stop_motion_program(self, task_no):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MP_START, 0, task_no)
        self._res_check_error(res)

    async def pause(self, task_no):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MP_PAUSE, 0, task_no)
        self._res_check_error(res)

    async def resume(self, task_no):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MP_RESUME, 0, task_no)
        self._res_check_error(res)

    async def enable(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_SERVO_ON, 0, 0)
        self._res_check_error(res)

    async def disable(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_SERVO_OFF, 0, 0)
        self._res_check_error(res)

    async def read_controller_state(self, controller_info = None):
        if controller_info is None:
            controller_info = await self.get_controller_info()
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_READ_STATE, 0)
        self._res_check_error(res)

        with io.BytesIO(payload) as f:
            return read_controller_state(f, controller_info)
        
    async def read_task_state(self, task_no):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_READ_TASK_STATE, 0, task_no)
        self._res_check_error(res)

        with io.BytesIO(payload) as f:
            return read_task_state(f)

    async def preempt_motion_program(self, task_no, buffer_index, cmd_num):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MP_PREEMPT, 0, task_no, buffer_index, cmd_num)
        self._res_check_error(res)

    async def start_motion_streaming(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MS_ENABLE, 0)
        self._res_check_error(res)

    async def stop_motion_streaming(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MS_DISABLE, 0)
        self._res_check_error(res)

    async def update_motion_streaming_pulse_target(self, target):
        if not isinstance(target, list):
            target = [target]

        req_payload = _pack_streaming_target(target)

        res, res_payload = await self.send_request(RR_MOTOPLUS_CMDOP_MS_UPDATE_TARGET, 0, param1=len(target), payload=req_payload)
        self._res_check_error(res)

    async def motion_streaming_clear_error(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_MS_CLEAR_ERROR, 0)
        self._res_check_error(res)

    async def load_controller_file(self, file_number, contents):
        assert isinstance(contents, bytes) or isinstance(contents, bytearray), "File contents must be bytes or bytearray"
        byte_len = len(contents)
        rem = byte_len % 4
        if rem != 0:
            contents = copy.copy(contents)
            contents += b'\0'*(4-rem)
        payload = np.frombuffer(contents, dtype=np.uint32)

        res, res_payload = await self.send_request(RR_MOTOPLUS_CMDOP_FILE_LOAD, 0, param1=file_number, param2=byte_len, 
                                                   payload=payload)
        self._res_check_error(res)

    async def start_remote_job(self):
        res, payload = await self.send_request(RR_MOTOPLUS_CMDOP_REMOTE_JOB_START, 0)
        self._res_check_error(res)

def _pack_streaming_target(target):
    ret = np.zeros((len(target) * 17,),dtype=np.int32)
    for i in range(len(target)):
        off = i*17
        ret[off] = target[i].group_number
        t = target[i].target
        p = target[i].max_pulse_error
        ret[(off+1):off+1+len(t)] = t
        if p is None:
            ret[(off+9):(off+17)] = 1000
        else:
            assert np.all(p1 >= 0 for p1 in p), "max_pulse_error must be >=0"
            ret[(off+9):off+9+len(p)] = p
    return ret
