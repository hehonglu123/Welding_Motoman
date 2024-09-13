import numpy as np 
import cv2
import matplotlib.pyplot as plt
from thermal_couple_conversion import voltage_to_temperature

room_temp=23
ir_images=np.load('../../../recorded_data/IR_calibration/ir_images.npy')
temperature_reading=np.loadtxt('../../../recorded_data/IR_calibration/temperature_reading.csv',delimiter=',')
start_idx=130   #index of thermal couple heated up to peak

temperature_bias=room_temp-voltage_to_temperature(np.average(temperature_reading[:10,1]))
print('Temperature Bias: ',temperature_bias)

ROI=[(139,139),(149,149)]
pixel_count=[]
for i in range(start_idx,len(ir_images)):
    pixel_count.append(np.average(ir_images[i][ROI[0][1]:ROI[1][1],ROI[0][0]:ROI[1][0]]))
    ir_normalized = ((ir_images[i] - np.min(ir_images[i])) / (np.max(ir_images[i]) - np.min(ir_images[i]))) * 255
    ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    temperature_reading[i,1]=voltage_to_temperature(temperature_reading[i,1])+temperature_bias
    # ###add bounding box
    # cv2.rectangle(ir_bgr,ROI[0],ROI[1],(0,255,0),2)
    # cv2.imshow("IR Recording", ir_bgr)
    # if cv2.waitKey(1) == 27: 
    #     break  # esc to quit

cv2.destroyAllWindows()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(temperature_reading[start_idx:,0], pixel_count, 'r-', label='Pixel Counts')
ax2.plot(temperature_reading[start_idx:,0], temperature_reading[start_idx:,1], 'y-', label='Temperature Reading')


ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pixel Counts', color='r')
ax2.set_ylabel('Temperature (C)', color='y')
plt.title('Pixel Temperature vs. Time')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)

plt.show()

plt.scatter(pixel_count,temperature_reading[start_idx:,1])
plt.ylabel('Temperature Reading (C)')
plt.xlabel('Pixel Count')
plt.title('Temperature Reading vs. Pixel Count')
plt.show()


solid_index=np.where(temperature_reading[start_idx:,1]<550)[0][0]+start_idx
polyfit=np.polyfit(pixel_count[solid_index-start_idx:],temperature_reading[solid_index:,1],deg=40)
predicted_temp=np.poly1d(polyfit)(pixel_count[solid_index-start_idx:])
plt.plot(pixel_count[solid_index-start_idx:],temperature_reading[solid_index:,1],label='Actual Temperature')
plt.plot(pixel_count[solid_index-start_idx:],predicted_temp,label='Predicted Temperature')
plt.legend()
plt.xlabel('Pixel Count')
plt.ylabel('Temperature (C)')
plt.title('Fit Temperature vs. Actual Temperature')
plt.show()  

np.save('ER4043_IR_calibration.npy',polyfit)