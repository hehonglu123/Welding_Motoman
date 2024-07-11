from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys


url='rr+tcp://localhost:12182/?service=FLIR_RR_PROCESS'




########subscription mode
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

sub=RRN.SubscribeService(url)
obj = sub.GetDefaultClientWait(30)		#connect, timeout=30s
ir_process_result=sub.SubscribeWire("ir_process_result")


sub.ClientConnectFailed += connect_failed

while True:
    try:
        time.sleep(0.1)
        wire_packet=ir_process_result.TryGetInValue()
        print(wire_packet[0])
        if wire_packet[0]:
            print(wire_packet[1].flame_reading, wire_packet[1].torch_bottom, wire_packet[1].weld_pool)
    except:
        traceback.print_exc()
        break
            