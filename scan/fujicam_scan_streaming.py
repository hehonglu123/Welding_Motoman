import numpy as np
import time
from matplotlib import pyplot as plt
from RobotRaconteur.Client import *

fujicam_url = 'rr+tcp://localhost:12181/?service=fujicam'
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
sub=RRN.SubscribeService(fujicam_url)
obj = sub.GetDefaultClientWait(2)		#connect, timeout=2s
fuji_scan_wire=sub.SubscribeWire("lineProfile")
sub.ClientConnectFailed += connect_failed

time.sleep(0.5) # wait for connection


# # test frame rate
# start_time=time.perf_counter()
# for i in range(1000):
#     wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
#     valid_indices=np.where(wire_packet[1].I_data>1)[0]
#     valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>10)[0])
#     line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))
# end_time=time.perf_counter()
# print("Frame rate: "+str(1000/(end_time-start_time))+" Hz")

rate=30
rate=rate*2

print("Click on the plot and press any key to exit")
while True:
    try:
        wire_packet=fuji_scan_wire.TryGetInValue() # log fuji cam scanner data
        valid_indices=np.where(wire_packet[1].I_data>1)[0]
        valid_indices=np.intersect1d(valid_indices,np.where(np.abs(wire_packet[1].Z_data)>10)[0])
        line_profile=np.hstack((wire_packet[1].Y_data[valid_indices].reshape(-1,1),wire_packet[1].Z_data[valid_indices].reshape(-1,1)))

        plt.clf()
        plt.plot(line_profile[:,0], line_profile[:,1], 'o', markersize=2)
        plt.title("FujiCam Line Profile")
        plt.xlabel("Y (mm)")
        plt.ylabel("Z (mm)")
        plt.xlim(-33, 33)
        plt.ylim(60, 140)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid()
        plt.pause(1/rate)
        quit_flag = plt.waitforbuttonpress(1/rate)
        if quit_flag:
            break
    except KeyboardInterrupt:
        plt.close()