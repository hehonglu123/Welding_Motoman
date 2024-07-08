from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys


url='rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS'




########subscription mode
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

sub=RRN.SubscribeService(url)
obj = sub.GetDefaultClientWait(30)		#connect, timeout=30s
flame_reading=sub.SubscribeWire("flame_reading")


sub.ClientConnectFailed += connect_failed

while True:
    try:
        time.sleep(0.1)
        wire_packet=flame_reading.TryGetInValue()
        print(wire_packet[0])
        if wire_packet[0]:
            print(wire_packet[1].Value)
    except:
        traceback.print_exc()
        break
            