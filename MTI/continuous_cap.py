# Read a frame from the scanner and plot

from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time

c = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")

c.setExposureTime("25")
time.sleep(0.5)

now=time.time()
while True:
    frame = c.Capture()
    print(1/(time.time()-now))
    now=time/time()

plt.figure()
plt.plot(frame.X_data, frame.Z_data, "x")
plt.title("XY Scatter Plot")
plt.figure()
plt.plot(frame.X_data, frame.I_data, "x")
plt.title("XI Scatter Plot")

plt.show()