import sys, glob, pickle, os, traceback, wave
from RobotRaconteur.Client import *
from weldRRSensor import *


########################################################RR Microphone########################################################
microphone = RRN.ConnectService('rr+tcp://192.168.55.20:60828?service=microphone')
########################################################RR FLIR########################################################
flir=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
########################################################RR CURRENT########################################################
current_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')

#############################################################UNIT TEST#############################################################
#microphone
rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=microphone,current_service=None)
rr_sensors = WeldRRSensor(weld_service=None,cam_service=flir,microphone_service=None,current_service=None)
rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=None,current_service=current_sub)
rr_sensors.start_all_sensors()

counts=0
while True:
    time.sleep(60)
    try:
        rr_sensors.stop_all_sensors()
        dir='recorded_data/'+str(counts)+'/'
        os.mkdirs(dir,exist_ok=True)
        rr_sensors.save_all_sensors(dir)
        rr_sensors.clear_all_sensors()
        counts+=1
    except:
        traceback.print_exc()
    finally:
        break