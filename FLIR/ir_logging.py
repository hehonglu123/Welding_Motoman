# Capture and display streaming frames

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, pickle

def packet_received(self, pipe):
    pass

image_consts = None
logging=[]
ts=[]
def main():
    global logging, ts

    now=time.time()
    url='rr+tcp://192.168.55.10:60827/?service=camera'

    c1=RRN.ConnectService(url)
    c1.setf_param("focus_pos", RR.VarValue(int(1400),"int32"))
    c1.setf_param("object_distance", RR.VarValue(0.3,"double"))
    c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
    c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
    c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))

    # c1.setf_param("current_case", RR.VarValue(2,"int32"))
    c1.setf_param("ir_format", RR.VarValue("temperature_linear_100mK","string"))
    # c1.setf_param("ir_format", RR.VarValue("radiometric","string"))

    c1.setf_param("object_emissivity", RR.VarValue(0.7,"double"))    
    
    c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
    c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

    time.sleep(1)


    # print(print(c1.getf_param('atmospheric_temperature').data[0]))
    global image_consts
    image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

    p=c1.frame_stream.Connect(-1)

    #Set the callback for when a new pipe packet is received to the
    #new_frame function
    p.PacketReceivedEvent+=new_frame
    try:
        c1.start_streaming()
    except: pass


    fig = plt.figure(1)
    
    try:
        while True:
            time.sleep(0.001)
    
    finally:
        try:
            p.Close()
        except: pass

        try:
            c1.stop_streaming()
        except: pass

        with open('ir_recording_raw.pickle', 'wb') as file:
            pickle.dump(logging, file)



current_mat = None

def new_frame(pipe_ep):
    global logging

    #Loop to get the newest frame
    while (pipe_ep.Available > 0):
        #Receive the packet
        rr_img=pipe_ep.ReceivePacket()
        if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono8"]:
            # Simple uint8 image
            mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
        elif rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
            data_u16 = np.array(rr_img.data.view(np.uint16))
            mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
        
        # ir_format = rr_img.image_info.extended["ir_format"].data

        # if ir_format == "temperature_linear_10mK":
        #     display_mat = (mat * 0.01) - 273.15    
        # elif ir_format == "temperature_linear_100mK":
        #     display_mat = (mat * 0.1) - 273.15    
        # else:
        #     display_mat = mat

        # #Convert the packet to an image and set the global variable
        # current_mat = display_mat

        logging.append(mat)


if __name__ == "__main__":
    main()