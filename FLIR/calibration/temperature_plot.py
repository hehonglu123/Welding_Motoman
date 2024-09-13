import numpy as np 
import matplotlib.pyplot as plt
from thermal_couple_conversion import voltage_to_temperature

room_temp=20

temperature_reading=np.loadtxt('recorded_data/temperature_reading.csv',delimiter=',')
temperature_bias=room_temp-voltage_to_temperature(np.average(temperature_reading[:10,1]))

temperature_converted=[]
for i in range(len(temperature_reading)):
    temperature_converted.append(voltage_to_temperature(temperature_reading[i,1])+temperature_bias)

plt.plot(temperature_reading[:,0],temperature_converted)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.title('Thermal Couple Temperature vs. Time')
plt.show()