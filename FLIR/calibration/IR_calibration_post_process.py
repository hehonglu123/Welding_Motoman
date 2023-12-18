import numpy as np 
import cv2
import matplotlib.pyplot as plt

ir_images=np.load('ir_images.npy')
temperature_reading=np.loadtxt('temperature_reading.csv',delimiter=',')

ROI=[(0,0),(0,0)]
pixel_count=[]
for i in range(len(ir_images)):
    pixel_count.append(np.average(ir_images[i][ROI[0][1]:ROI[1][1],ROI[0][0]:ROI[1][0]]))
    ir_normalized = ((ir_images[i] - np.min(ir_images[i])) / (np.max(ir_images[i]) - np.min(ir_images[i]))) * 255
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    ###add bounding box
    cv2.rectangle(ir_bgr,ROI[0],ROI[1],(0,255,0),2)
    cv2.imshow("IR Recording", ir_bgr)
    if cv2.waitKey(1) == 27: 
        break  # esc to quit

cv2.destroyAllWindows()
plt.plot(temperature_reading,pixel_count)
plt.xlabel('Temperature Reading')
plt.ylabel('Pixel Count')
plt.title('Pixel Count vs Temperature Reading')
plt.show()

