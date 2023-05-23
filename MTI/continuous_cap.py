# Read a frame from the scanner and plot

from RobotRaconteur.Client import *
import matplotlib.pyplot as plt
import time

c = RRN.ConnectService("rr+tcp://192.168.55.10:60830/?service=MTI2D")

c.setExposureTime("25")
time.sleep(0.5)
fig = plt.figure(1)


now=time.time()
while True:
    frame = c.lineProfile
    plt.plot(frame.X_data, frame.Z_data, "x")

    plt.pause(0.1)
    plt.clf()
