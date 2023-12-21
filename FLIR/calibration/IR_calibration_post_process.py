import numpy as np 
import cv2
import matplotlib.pyplot as plt
from thermal_couple_conversion import voltage_to_temperature

ir_images=np.load('recorded_data/ir_images.npy')[80:]
temperature_reading=np.loadtxt('recorded_data/temperature_reading.csv',delimiter=',')[80:]

ROI=[(139,139),(149,149)]
pixel_count=[]
for i in range(len(ir_images)):
    pixel_count.append(np.average(ir_images[i][ROI[0][1]:ROI[1][1],ROI[0][0]:ROI[1][0]]))
    ir_normalized = ((ir_images[i] - np.min(ir_images[i])) / (np.max(ir_images[i]) - np.min(ir_images[i]))) * 255
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    temperature_reading[i,1]=voltage_to_temperature(temperature_reading[i,1])
    # ###add bounding box
    # cv2.rectangle(ir_bgr,ROI[0],ROI[1],(0,255,0),2)
    # cv2.imshow("IR Recording", ir_bgr)
    # if cv2.waitKey(1) == 27: 
    #     break  # esc to quit

cv2.destroyAllWindows()

plt.plot(temperature_reading[:,0],pixel_count,label='Pixel Count')
plt.plot(temperature_reading[:,0],temperature_reading[:,1],label='Temperature Reading')
plt.xlabel('Time (s)')
plt.ylabel('Temperature, Pixel Reading')
plt.title('Pixel Temperature vs Time')
plt.legend()
plt.show()


plt.scatter(temperature_reading[:,1],pixel_count)
plt.xlabel('Temperature Reading')
plt.ylabel('Pixel Count')
plt.title('Pixel Count vs Temperature Reading')
plt.show()

