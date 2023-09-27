import sys, glob, pickle, os, traceback, wave
from RobotRaconteur.Client import *
from weldRRSensor import *
sys.path.append('../toolbox/')
from WeldSend import *
from dx200_motion_program_exec_client import *


robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=15,tool_file_path=config_dir+'torch.csv',\
	pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=R1_marker_dir+'MA2010_'+R1_ph_dataset_date+'_marker_config.yaml',tool_marker_config_file=weldgun_marker_dir+'weldgun_'+R1_ph_dataset_date+'_marker_config.yaml')
primitives=['movel', 'movel']
v=[10,4]
path_q=np.array([[-0.40792327,  0.6501066 ,  0.19265245,  0.20088046, -0.76420849,-0.49824186],
       [-0.40792327,  0.65399247,  0.18877973,  0.20251446, -0.75660721,-0.50049703],
       [-0.44311285,  0.69087143,  0.24214015,  0.18223052, -0.76674788,-0.45101575],
       [-0.44311285,  0.68712597,  0.24610138,  0.18079734, -0.77432793,-0.4490184 ]])
robot_client=MotionProgramExecClient()
ws=WeldSend(robot_client)
########################################################RR Microphone########################################################
microphone = RRN.ConnectService('rr+tcp://192.168.55.15:60828?service=microphone')
########################################################RR FLIR########################################################
flir=RRN.ConnectService('rr+tcp://192.168.55.10:60827/?service=camera')
########################################################RR CURRENT########################################################
current_sub=RRN.SubscribeService('rr+tcp://192.168.55.21:12182?service=Current')

#############################################################UNIT TEST#############################################################
# rr_sensors = WeldRRSensor(weld_service=None,cam_service=flir,microphone_service=None,current_service=None)
rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=microphone,current_service=None)
# rr_sensors = WeldRRSensor(weld_service=None,cam_service=None,microphone_service=None,current_service=current_sub)
counts=0
while True:
    ws.jog_single(robot_weld,path_q[0],1)
    rr_sensors.start_all_sensors()
    ws.weld_segment_single(primitives,robot_weld,path_q[1:-1],v,cond_all=[220],arc=False)
    rr_sensors.stop_all_sensors()
    try:
        dir='recorded_data/'+str(counts)+'/'
        os.makedirs(dir,exist_ok=True)
        # print(len(rr_sensors.ir_timestamp),len(rr_sensors.ir_recording))
        print(len(rr_sensors.audio_recording))
        rr_sensors.save_all_sensors(dir)
        counts+=1
    except:
        traceback.print_exc()
        break