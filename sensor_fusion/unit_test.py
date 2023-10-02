import sys, glob, pickle, os, traceback, wave
from RobotRaconteur.Client import *
from weldRRSensor import *


########################################################RR Microphone########################################################
microphone = RRN.ConnectService('rr+tcp://192.168.55.15:60828?service=microphone')
########################################################RR FLIR########################################################
flir=RRN.ConnectService('rr+tcp://192.168.55.15:60827/?service=camera')
########################################################RR CURRENT########################################################
current_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')

#############################################################UNIT TEST#############################################################
# rr_sensors = WeldRRSensor(weld_service=None,cam_service=flir,microphone_service=None,current_service=None)
rr_sensors = WeldRRSensor(weld_service=None,cam_service=flir,microphone_service=microphone,current_service=None)
# rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=None,current_service=current_sub)
counts=0
micrphone_length_all=[]

while True:
    
    try:
        rr_sensors.start_all_sensors()
        time.sleep(20)
        rr_sensors.stop_all_sensors()
        dir='recorded_data/'+str(counts)+'/'
        os.makedirs(dir,exist_ok=True)
        # print(len(rr_sensors.ir_timestamp),len(rr_sensors.ir_recording))
        print(len(rr_sensors.audio_recording))
        micrphone_length_all.append(len(rr_sensors.audio_recording))
        rr_sensors.save_all_sensors(dir)
        counts+=1
    except:
        traceback.print_exc()
        print(micrphone_length_all)
        break