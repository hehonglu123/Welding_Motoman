# Capture and display streaming frames

from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, cv2, traceback

def main():
    recorded_dir='recorded_data/'

    url = 'rr+tcp://192.168.55.10:60827/?service=camera'

    c1 = RRN.ConnectService(url)

    c1.setf_param("focus_pos", RR.VarValue(int(1900),"int32"))
    c1.setf_param("object_distance", RR.VarValue(0.4,"double"))
    c1.setf_param("reflected_temperature", RR.VarValue(291.15,"double"))
    c1.setf_param("atmospheric_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("relative_humidity", RR.VarValue(50,"double"))
    c1.setf_param("ext_optics_temperature", RR.VarValue(293.15,"double"))
    c1.setf_param("ext_optics_transmission", RR.VarValue(0.99,"double"))
    # c1.setf_param("current_case", RR.VarValue(2,"int32"))
    c1.setf_param("ir_format", RR.VarValue("radiometric","string"))
    c1.setf_param("object_emissivity", RR.VarValue(0.13,"double"))
    c1.setf_param("scale_limit_low", RR.VarValue(293.15,"double"))
    c1.setf_param("scale_limit_upper", RR.VarValue(5000,"double"))
    image_consts = RRN.GetConstants('com.robotraconteur.image', c1)
    counts=0
    time.sleep(1)
    while True:
        try:
            rr_img=c1.capture_frame()
            if rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono8"]:
                # Simple uint8 image
                current_mat = rr_img.data.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
            elif rr_img.image_info.encoding == image_consts["ImageEncoding"]["mono16"]:
                data_u16 = np.array(rr_img.data.view(np.uint16))
                current_mat = data_u16.reshape([rr_img.image_info.height, rr_img.image_info.width], order='C')
            #cap
            current_mat[current_mat >10000] = 10000
            ir_normalized = ((current_mat - np.min(current_mat)) / (np.max(current_mat) - np.min(current_mat))) * 255
            ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

            cv2.imwrite(recorded_dir+str(counts)+'.jpg', ir_bgr,[cv2.IMWRITE_JPEG_QUALITY, 100])
            counts+=1
            time.sleep(3)
        except:
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()