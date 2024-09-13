from RobotRaconteur.Client import *
import time
import matplotlib.pyplot as plt
import numpy as np

rate = RRN.CreateRate(1000)

t1 = time.perf_counter()
t = []
for i in range(1000):
    # rate.Sleep()
    time.sleep(0.001)
    t.append(time.perf_counter())
t2 = time.perf_counter()

print("Time for 125 sleeps: " + str(t2-t1))

plt.plot(np.array(t)-t[0], "x")
plt.show()