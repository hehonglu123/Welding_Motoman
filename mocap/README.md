# Calibrating PH Parameters

We are using the product of exponential to calculate the forward kinematics (FK) of robots. The product of exponential requires the PH parameters of the robot. H represents the rotating axis of each joint, and P is the vector from the origin of frames from joint to joint.

The nominal PH parameters are obtained from the manual provided by the robot vendors, which is how they were initially designed. However, these parameters may undergo changes due to factors such as backlash of motors, manufacturing deviations, thermal expansion, etc. This can lead to inaccuracies in the estimation of the tool center point (TCP) using FK.

Therefore, the goal is to estimate/calibrate the PH parameters with the help of a motion capture system.

## Motion Capture System

The algorithms and codes are designed to be independent of the specific motion capture system being used. Currently, we are using Optitrack.

### Optitrack

#### RR Driver
Please refer to [this repository](https://github.com/eric565648/optitrack_mocap_robotraconteur_driver) for the RR driver.

#### Mocap Listener

The file `MocapPoseListener.py` is a useful tool that can collect data points in the background as an RR client. It further converts the data points into a specified frame for the user. Please refer to the code for examples.

## Run the Algorithm

### Data Collection

Run the following command:

```
python3 mocap_calib_PH_datacollect.py
```

### Estimation

Run the following command:

```
python3 mocap_calib_PH_offline.py
```


### Numerical Gradient Descent

Run the following command:

```
python3 mocap_calib_PH_grad.py
```

### 
