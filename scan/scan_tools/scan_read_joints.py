
import numpy as np
import socket
import struct
import time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('0.0.0.0',11000))

hz=[]
while True:
    now=time.time()
    buf = s.recv(1024)
    data = struct.unpack("<16i",buf)
    print(np.array(data[2:]))
    # hz.append(1/(time.time()-now))
    # print(np.average(hz))