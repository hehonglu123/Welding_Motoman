from RobotRaconteur.Client import *
import time
import numpy as np

c = RRN.ConnectService('rr+tcp://192.168.55.10:60823?service=welder')

consts = RRN.GetConstants("experimental.fronius", c)
flags_const = consts["WelderStateFlags"]
hflags_const = consts["WelderStateHighFlags"]

timestamp=[]
voltage=[]
current=[]
feedrate=[]
energy=[]

while True:
    try:
        state, _ = c.welder_state.PeekInValue()

        flags_str = []

        flags = state.welder_state_flags
        hflags = state.welder_state_flags >> 32

        for s, f in flags_const.items():
            if flags & f:
                flags_str.append(s)

        for s, f in hflags_const.items():
            if hflags & f:
                flags_str.append(s)

        print(f"flags: {', '.join(flags_str)}")

        print(f"welding_process: {state.welding_process}")
        print(f"main_error: {state.main_error}")
        print(f"warning: {state.warning}")
        print(f"welding_voltage: {state.welding_voltage}")
        print(f"welding_current: {state.welding_current}")
        print(f"wire_speed: {state.wire_speed}")
        print(f"welding_energy: {state.welding_energy}")
        print("")
        print("")

        timestamp.append(state.ts['microseconds'][0])
        # timestamp.append(time.time())
        voltage.append(state.welding_voltage)
        current.append(state.welding_current)
        feedrate.append(state.wire_speed)
        energy.append(state.welding_energy)

        time.sleep(0.1)

    except KeyboardInterrupt:
        break

np.savetxt('weld_log.csv',np.array([timestamp,voltage,current,feedrate,energy]).T,delimiter=',', header='timestamp,voltage,current,feedrate,energy',comments='')


