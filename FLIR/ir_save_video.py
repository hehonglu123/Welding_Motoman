# Capture and display streaming frames

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, cv2

def packet_received(self, pipe):
    pass

image_consts = None

def main():
    cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)

    url = 'rr+tcp://192.168.55.10:60827/?service=camera'

    c1 = RRN.ConnectService(url)

    c1.setf_param("focus_pos", RR.VarValue(int(1900),"int32"))
    c1.setf_param("object_distance", RR.VarValue(0.4,"double"))
    c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
    c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
    c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
    c1.setf_param("current_case", RR.VarValue(2,"int32"))
    c1.setf_param("ir_format", RR.VarValue("radiometric","string"))
    c1.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
    c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
    c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))

    global image_consts, ts, current_mat
    ts = 0
    image_consts = RRN.GetConstants('com.robotraconteur.image', c1)

    p = c1.frame_stream.Connect(-1)

    p.PacketReceivedEvent += new_frame
    try:
        c1.start_streaming()
    except:
        pass

    fig = plt.figure(1)

    # Initializing the VideoWriter
    result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'X264'),
                         30, (240,320))

    try:
        while True:
            if current_mat is not None:
                ir_normalized = ((current_mat - np.min(current_mat)) / (np.max(current_mat) - np.min(current_mat))) * 255
                ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
                cv2.imshow("IR Recording", cv2.rotate(ir_bgr, cv2.ROTATE_90_CLOCKWISE))
                # Write frame to video file
                result.write(cv2.rotate(ir_bgr, cv2.ROTATE_90_CLOCKWISE))
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
    finally:
        try:
            p.Close()
            cv2.destroyAllWindows()
            result.release()
        except:
            pass



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