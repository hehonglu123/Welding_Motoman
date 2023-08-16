import cv2, time
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *
from robot_def import *

for slice_num in range(1000):
    print(slice_num)
    try:
        # Load the IR recording data from the pickle file
        with open('../../../recorded_data/wall_recording_streaming/slice_%i_0_flir.pickle'%slice_num, 'rb') as file:
            ir_recording = pickle.load(file)
    except:
        continue

    ir_ts=np.loadtxt('../../../recorded_data/wall_recording_streaming/slice_%i_0_flir_ts.csv'%slice_num, delimiter=',')[1:]
    js=np.loadtxt('../../../recorded_data/wall_recording_streaming/slice_%i_0_joint.csv'%slice_num, delimiter=',')
    js[:,0]=js[:,0]-js[0,0]
    ir_ts=ir_ts-ir_ts[0]
    robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
        pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=0)
    ###get average layer height
    p_all=robot.fwd(js[:,1:7]).p_all
    height_avg=np.mean(p_all[:,2])
    #interp x with time
    x_interp=np.interp(ir_ts, js[:,0], p_all[:,0])

    # Create a window to display the images
    # cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)
    # Set the colormap (inferno) and normalization range for the color bar
    cmap = cv2.COLORMAP_INFERNO
    colorbar_min = np.min(ir_recording)
    colorbar_max = np.max(ir_recording)
    wirelength_all=[]
    x_all=[]
    for i in range(len(ir_recording)):
        # print(np.max(ir_recording[i]), np.min(ir_recording[i]))
        now=time.time()
        centroid, bbox=flame_detection(ir_recording[i])
        # print(time.time()-now)

        temp=counts2temp(ir_recording[i].flatten(),6.39661118e+03, 1.40469989e+03, 1.00000008e+00, 8.69393436e+00, 8.40029488e+03,Emiss=0.13).reshape((240,320))
        temp[temp > 1300] = 1300    ##thresholding
        # Normalize the data to [0, 255]
        ir_normalized = ((temp - np.min(temp)) / (np.max(temp) - np.min(temp))) * 255
        
        # ir_normalized = ir_normalized[50:-50, 50:-50]
        ir_normalized=np.clip(ir_normalized, 0, 255)

        # Convert the IR image to BGR format with the inferno colormap
        ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

        # add bounding box
        if centroid is not None:
            # cv2.rectangle(ir_bgr, (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,255,0), thickness=1)   #flame bbox
            bbox_below_size=10
            centroid_below=(int(centroid[0]+bbox[2]/2+bbox_below_size/2),centroid[1])
            cv2.rectangle(ir_bgr, (int(centroid_below[0]-bbox_below_size/2),int(centroid_below[1]-bbox_below_size/2)), (int(centroid_below[0]+bbox_below_size/2),int(centroid_below[1]+bbox_below_size/2)), (0,255,0), thickness=1)   #flame below centroid
            cv2.line(ir_bgr,(139,136),(int(centroid[0]),int(centroid[1])),(0,255,0))
            wirelength_all.append(centroid[0]-139)
            x_all.append(x_interp[min(i,len(ir_ts)-1)])

        cv2.rectangle(ir_bgr, (50,110,95,60), (255,0,0), thickness=1)   #torch bounding box
        
        # Display the IR image
        # cv2.imshow("IR Recording", ir_bgr)

        # # Wait for a specific time (in milliseconds) before displaying the next frame
        # cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

    # Close the window after the loop is completed
    # cv2.destroyAllWindows()

    height_all=height_avg-0.3*np.array(wirelength_all)
    plt.plot(x_all,height_all[:len(x_all)])


plt.show()
