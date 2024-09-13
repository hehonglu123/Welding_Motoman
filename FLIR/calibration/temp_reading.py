from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import time, traceback
from thermal_couple_conversion import voltage_to_temperature

url='rr+tcp://localhost:12182/?service=Temperature'
sub=RRN.SubscribeService(url)
temperature_sub=sub.SubscribeWire("temperature")

time.sleep(1)
while True:
    print(voltage_to_temperature(temperature_sub.InValue))