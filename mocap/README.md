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

## Testing Data

There are sets of files available for training and testing purposes. Each set consists of two files: one containing the robot joint angles, and the other containing the Cartesian pose of the tool's rigid body in the base frame. The poses in these files are aligned, meaning that the data in the same row of the files represent the same pose. It is important to note that both files within a set should have the same number of rows.

The data collection process involves the robot traversing the reachable space

An example robot file. Column 1~7 are `q1` to `q6`. The unit of the joint angles is `radians`.
```
0.000000000000000000e+00,-9.599317085782494985e-01,-1.221738972348442642e+00,-1.706319033052640030e-05,-1.780513462534675552e-05,0.000000000000000000e+00
-1.746135931663858956e-02,-9.599304220014058808e-01,-1.221738972348442642e+00,-1.706319033052640030e-05,0.000000000000000000e+00,0.000000000000000000e+00
1.744834787154416963e-02,-9.599304220014058808e-01,-1.221728015343047868e+00,-1.706319033052640030e-05,1.669231371126258330e-05,0.000000000000000000e+00
...
```

An example of mocap file. Column 1~3 are `xyz` and Column 4~7 are quaternion `qw qx qy qz`.
```
8.264911262933443368e+02,-8.910976219197795700e+00,2.594488438267676429e+02,4.184298376874036607e-01,5.526446085205515280e-01,5.708252190399274451e-01,4.400670141172571825e-01
8.262231066728221549e+02,-2.328624425902405903e+01,2.594304449890893807e+02,4.222374158245049691e-01,5.575911105062921758e-01,5.659711341421594932e-01,4.364451780923254698e-01
8.265213406561595093e+02,5.418167856058365928e+00,2.594302387735352795e+02,4.146614032239314129e-01,5.475640659619190886e-01,5.758450192228546927e-01,4.434321009803432467e-01
...
```

#### Availabe Dataset
- Testing Dataset 1 [robot_q](https://github.com/hehonglu123/Welding_Motoman/blob/devel-eric/mocap/kinematic_raw_data/test0516/robot_q_align.csv) [mocap_T](https://github.com/hehonglu123/Welding_Motoman/blob/devel-eric/mocap/kinematic_raw_data/test0516/mocap_T_align.csv)

## Numerical Gradient Descent

### Data

The dataset is represented in the same format as the 'Testing Data'.

During the data collection process, the robot traverses various configurations of `q2 q3`. Within each configuration, the robot samples `N` points around that configuration to ensure a rich dataset and prevent situations such as rolling off into null space during calibration. As a result, rows 1 to N represent the same 'q2 q3' configuration, rows N+1 to 2N represent another identical configuration, and so on.

#### Availabe Dataset
- Dataset 1 [robot_q](https://github.com/hehonglu123/Welding_Motoman/blob/devel-eric/mocap/PH_grad_data/test0516_R1/train_data_robot_q_align.csv) [mocap_T](https://github.com/hehonglu123/Welding_Motoman/blob/devel-eric/mocap/PH_grad_data/test0516_R1/train_data_mocap_T_align.csv). Neighbor N=7.

### Run Calibration

Run the following command:

```
python3 mocap_calib_PH_grad.py
```

### Data Collection

Run the following command:

```
python3 mocap_calib_PH_grad_datacollect.py
```

## Rotating Joints Method

### Run Estimation

Run the following command:

```
python3 mocap_calib_PH_rotate.py
```

### Data Collection

Run the following command:

```
python3 mocap_calib_PH_rotate_datacollect.py
```