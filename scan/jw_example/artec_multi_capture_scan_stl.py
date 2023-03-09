from RobotRaconteur.Client import *
import time
from contextlib import suppress

c = RRN.ConnectService('rr+tcp://localhost:64238?service=scanner')

N = 100

scan_handles = []

t1 = time.perf_counter()
for i in range(N):
    scan_handles.append(c.capture_deferred(False))
t2 = time.perf_counter()

print(f"Capture took {t2-t1} seconds at a rate of {N/(t2-t1)} fps")

t1 = time.perf_counter()
prepare_gen = c.deferred_capture_prepare(scan_handles)
with suppress(RR.StopIterationException):
    prepare_res = prepare_gen.Next()
    print(prepare_res)
t2 = time.perf_counter()
print(f"Preparing stl took {t2-t1} seconds")

t1 = time.perf_counter()
for i in range(N):
    stl_mesh_bytes = c.getf_deferred_capture(scan_handles[i])

    print("i")
    print(c.getf_deferred_capture_stamps(scan_handles[i]).seconds,'.',c.getf_deferred_capture_stamps(scan_handles[i]).micro_seconds)

    # with open(f"deferred_captured_mesh_{i+1}.stl", "wb") as f:
    #     f.write(stl_mesh_bytes)
t2 = time.perf_counter()

print(f"Saving stl took {t2-t1} seconds")
