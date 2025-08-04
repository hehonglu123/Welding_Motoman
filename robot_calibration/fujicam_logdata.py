import numpy as np
import time, pickle
from copy import deepcopy
from StreamingSend import *
from RobotRaconteur.Client import *

fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
sub=RRN.SubscribeService(fujicam_url)
obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
fuji_scan_wire=sub.SubscribeWire("lineProfile")
sub.ClientConnectFailed += connect_failed


stream_rate = 125.0
RR_robot_sub = RRN.SubscribeService('rr+tcp://localhost:59945?service=robot')
SS=StreamingSend(RR_robot_sub,streaming_rate=stream_rate)

input("Press Enter to start logging Fuji cam data...")

scan_exe = []
weld_js_exe = []
while True:
    try:
        wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
        valid_indices=np.where(wire_packet[1].I_data>1)[0]
        valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>50)[0])
        line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
        scan_exe.append(line_profile)

        weld_js_exe.append(np.append(time.perf_counter(),deepcopy(SS.q_cur))) # log timestamp and robot joints
        time.sleep(1/stream_rate) # wait for the next scan
    except KeyboardInterrupt:
        print("Logging interrupted by user.")
        break
    except Exception as e:
        print("Error logging Fuji cam data:", e)
        break

SS.deinitialize_robot()

data_dir = 'turntable_calibration/'
this_name = 'angle_0_0'

weld_js_exe = np.loadtxt(f'{data_dir}{this_name}_weld_js_exe.csv',delimiter=',')
with open(f'{data_dir}{this_name}_scan_exe.pickle', 'rb') as f:
    scan_exe = pickle.load(f)