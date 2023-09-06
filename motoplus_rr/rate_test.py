from RobotRaconteur.Client import *
import time
import matplotlib.pyplot as plt
import numpy as np

rate = RRN.CreateRate(125)

t1 = time.perf_counter()
t = []
for i in range(125):
    rate.Sleep()
    t.append(time.perf_counter())
t2 = time.perf_counter()

print("Time for 125 sleeps: " + str(t2-t1))

plt.plot(np.array(t)-t[0], "x")
plt.show()