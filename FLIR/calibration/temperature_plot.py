import numpy as np 
import matplotlib.pyplot as plt
from thermal_couple_conversion import voltage_to_temperature


temperature_reading=np.loadtxt('recorded_data/temperature_reading.csv',delimiter=',')

temperature_converted=[]
for i in range(len(temperature_reading)):
    temperature_converted.append(voltage_to_temperature(temperature_reading[i,1]))

plt.plot(temperature_reading[:,0],temperature_converted)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (C)')
plt.title('Temperature vs Time')
plt.show()