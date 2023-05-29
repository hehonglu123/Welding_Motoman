import numpy as np
import matplotlib.pyplot as plt

# Define the parameters
magnitude = 0.3*2
frequency = 0.1
osc_center=-0.3
# Define the time range
t = np.linspace(0, 100, 1000)

# Generate the triangle wave
triangle = osc_center + magnitude * (2 * np.abs(frequency * t % 1 - 0.5) - 0.5)

# Plot the triangle wave
plt.plot(t, triangle)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Centered Periodic Triangle Wave')
plt.grid(True)
plt.show()
