import asyncio
import socket
import traceback
from contextlib import suppress
import numpy as np
import struct
from typing import NamedTuple
import time

class Response(NamedTuple):
    payload_size: int = 0
    opcode: int = 0
    index: int = 0
    param1: int = 0
    param2: int = 0
    param3: int = 0
    param4: int = 0

async def async_sock_recv_all(loop, sock, nbytes):
    resp_bytes = b''
    while len(resp_bytes) < nbytes:
        resp_bytes1 = await asyncio.wait_for(loop.sock_recv(sock, nbytes-len(resp_bytes)),5)
        if resp_bytes1 is None or len(resp_bytes1) == 0:
            raise Exception("Socket connection closed")
        resp_bytes += resp_bytes1
    return resp_bytes

class MotoPlusRRDriverReverseSocketClient:
    def __init__(self, magic = 0x43415252):
        self._reverse = False
        self._reverse_port = None
        self._connect_ep = None
        self._connected = False
        self._lock = asyncio.Lock()
        self._cond = asyncio.Condition()
        self._sock = None
        self._run_task = None
        self._request_struct = struct.Struct("<8I")
        self._magic = magic
        self._heartbeat_wait_event = None

    def start(self, host, port):
        self._connect_ep = (host,port)
        loop = asyncio.get_event_loop()
        self._run_task = loop.create_task(self._run())

    def start_reverse(self, port):
        self._reverse = True
        self._reverse_port = port
        loop = asyncio.get_event_loop()
        self._run_task = loop.create_task(self._run())

    def stop(self):
        with suppress(Exception):
            self._run_task.cancel()

    async def _run(self):
        server = None
        if self._reverse:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.bind(('localhost', self._port))
            server.listen(8)
            server.setblocking(False)
        try:

            loop = asyncio.get_event_loop()
            while True:
                sock = None
                try:
                    if self._reverse:
                        sock, _ = await loop.sock_accept(server)
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.setblocking(False)
                        await asyncio.wait_for(loop.sock_connect(sock, self._connect_ep), 5)
                    async with self._cond:
                        self._sock = sock
                        self._cond.notify_all()
                    self._heartbeat_wait_event = asyncio.Event()

                    while self._sock is not None:
                        await self.send_heartbeat()
                        with suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(self._heartbeat_wait_event.wait(), 5)
                except asyncio.CancelledError:
                    return
                except Exception:
                    traceback.print_exc()
                finally:                    
                    async with self._cond:
                        self._sock = None
                        self._cond.notify_all()
                    with suppress(Exception):
                        sock.close()                            

        finally:
            if server is not None:
                server.close() 

    def _verify_payload(self, payload):
        assert isinstance(payload, np.ndarray), "payload must be numpy array"
        assert payload.dtype == np.int32 or payload.dtype == np.uint32, "payload must be int32 or uint32"
        assert len(payload.shape) == 1, "payload must be 1D array"

    async def send_request(self, opcode, index, param1 = None, param2 = None, param3 = None, param4 = None, payload = None):
        loop = asyncio.get_event_loop()

        if param1 is None:
            param1 = 0
        if param2 is None:
            param2 = 0
        if param3 is None:
            param3 = 0
        if param4 is None:
            param4 = 0

        payload_size = 0
        payload_bytes = None
        if payload is not None:
            self._verify_payload(payload)
            payload_size = len(payload)
            payload_bytes = payload.tobytes()

        request_bytes = self._request_struct.pack(self._magic, payload_size, opcode, index, param1, param2, param3, param4)

        async with self._lock:
            if self._sock is None:
                raise Exception("Socket not connected!")
            try:
                await loop.sock_sendall(self._sock, request_bytes)

                if payload_bytes is not None:
                    await loop.sock_sendall(self._sock, payload_bytes)

                resp_bytes = await async_sock_recv_all(loop, self._sock, 32)
            except:
                with suppress(Exception):
                    self._sock.close()
                    async with self._cond:
                        self._sock = None
                        self._cond.notify_all()
                    self._heartbeat_wait_event.set()
                raise
        resp_words = self._request_struct.unpack(resp_bytes)

        payload_bytes = None
        if (resp_words[1] > 0):
            payload_bytes = await async_sock_recv_all(loop, self._sock, resp_words[1]*4)
        return Response(*resp_words[1:]), payload_bytes

    async def send_heartbeat(self):
        await self.send_request(0,0)

    async def wait_ready(self, timeout=-1):
        async with self._cond:
            if timeout < 0:
                await self._cond.wait_for(lambda: self._sock is not None)
            else:
                await asyncio.wait_for(self._cond.wait_for(lambda: self._sock is not None), timeout)


    
