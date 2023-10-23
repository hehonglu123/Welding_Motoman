# Problem Track

## Driver Problem

### Robot Streaming(Motoplus) Non-smooth Motion (Unsolved)
Power cycle; use Ubuntu.

### Robot Streaming(Motoplus) Triggers Pulse Difference Check (Unsolved)
Linear continuous joint position command triggers 5° pulse difference check, likely due to loss of data.

### M1k Current Empty Reading (Temporary Solution)
Unplug and plug back the USB connection, known for disconnection.

### Microphone Miss Recording (Solved)
Inconsistent recording duration when streaming audio data, use localhost for reliable streaming recording, otherwise use duration recording function.


### IR Static Frame (Solved)
It takes 5~10s for FLIR to change calibration mode (low/high exposure/temperature), so it's possible to get static frames.

### MTI Disconnection (Temporary Solution)
1. Add `try catch` in the python script
2. Restart RR driver when catch disconnect
3. The robot should rescan the last printed layer




## Robot Problem
### Robot Joint Overload Error (Temporary Solution)
Long INFORM JOB causes joint 3(U) overload error on teachpendant. Try power cycle the controller.

### Robot Long-run TPS Error (Temporary Solution)
Long INFORM JOB causes TPS error on teachpendant. Try power cycle the controller.

### Robot INFOM Job Unterminated Motor (Jinhan only)


## Hardware Problem
### IR Protection Lens Crack (Unsolved)
Protection lens cracked

### Robot 2 Accidental Move (Unsolved)
Stablize Robot2 Base.

### Station Accidental Move (Unsolved)
Drill holes on ground to secure D500B.

### Coupon Thickness Diminish (Unsolved)
Coupons get thinner every time after machined, causing kinematics accuracy issue (TCP relative frame) 

### Current Clamp Battery Draining (Temporary Solution)
Turn off current clamp after every use.
Connect with power supply.