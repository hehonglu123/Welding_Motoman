from RobotRaconteur.Client import *
import time
import numpy as np

timestamp=[]
voltage=[]
current=[]
feedrate=[]
energy=[]

now=time.time()
def wire_cb(sub, value, ts):
    global timestamp, voltage, current, feedrate, energy, now
    
    print(time.time()-now)
    now=time.time()
    timestamp.append(value.ts['microseconds'][0])
    voltage.append(value.welding_voltage)
    current.append(value.welding_current)
    feedrate.append(value.wire_speed)
    energy.append(value.welding_energy)


sub=RRN.SubscribeService('rr+tcp://192.168.55.10:60823?service=welder')
obj = sub.GetDefaultClientWait(3)      #connect, timeout=30s
welder_state_sub=sub.SubscribeWire("welder_state")


welder_state_sub.WireValueChanged += wire_cb
input('press enter to quit')


np.savetxt('weld_log.csv',np.array([timestamp,voltage,current,feedrate,energy]).T,delimiter=',', header='timestamp,voltage,current,feedrate,energy',comments='')


