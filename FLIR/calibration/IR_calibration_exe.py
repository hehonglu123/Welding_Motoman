# Capture and display streaming frames

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, traceback


image_consts = None

def main():
    url='rr+tcp://localhost:12182/?service=Temperature'
    sub=RRN.SubscribeService(url)
    temperature_sub=sub.SubscribeWire("temperature")


    url='rr+tcp://localhost:60827/?service=camera'

    c1=RRN.ConnectService(url)
    #pre adjust focus
    c1.setf_param("focus_pos", RR.VarValue(int(1400),"int32"))
    c1.setf_param("object_distance", RR.VarValue(0.3,"double"))
    c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
    c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
    c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
    c1.setf_param("current_case", RR.VarValue(2,"int32"))
    c1.setf_param("ir_format", RR.VarValue("temperature_linear_100mK","string"))
    c1.setf_param("ir_format", RR.VarValue("radiometric","string"))
    c1.setf_param("object_emissivity", RR.VarValue(0.7,"double"))    
    c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
    c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

    global image_consts, ts
    ts=0
    image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

    p=c1.frame_stream.Connect(-1)

    #Set the callback for when a new pipe packet is received to the
    #new_frame function
    p.PacketReceivedEvent+=new_frame
    try:
        c1.start_streaming()
    except: pass


    ir_images=[]
    temperature_reading=[]
    timestamp=[]
    start_time=time.time()
    try:
        while True:
            if current_mat is not None:
                ir_images.append(current_mat)
                temperature_reading.append(temperature_sub.InValue)
                timestamp.append(time.time()-start_time)
                time.sleep(0.1)
    finally:
        try:
            p.Close()
            np.savetxt('recorded_data/temperature_reading.csv',temperature_reading,delimiter=',')
            np.save('recorded_data/ir_images.npy',np.vstack((np.array(timestamp),np.array(ir_images))).T)
                
        except: 
            traceback.print_exc()



current_mat = None

def new_frame(pipe_ep):
    global current_mat, ts

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
        
        ir_format = rr_img.image_info.extended["ir_format"].data

        if ir_format == "temperature_linear_10mK":
            display_mat = (mat * 0.01) - 273.15    
        elif ir_format == "temperature_linear_100mK":
            display_mat = (mat * 0.1) - 273.15    
        else:
            display_mat = mat

        #Convert the packet to an image and set the global variable
        ts=rr_img.image_info.data_header.ts['seconds']+rr_img.image_info.data_header.ts['nanoseconds']*1e-9
        current_mat = display_mat
        
        

if __name__ == "__main__":
    main()