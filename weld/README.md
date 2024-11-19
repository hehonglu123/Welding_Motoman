# Weld

This folder includes all codes performing welding motion. 

## Closed-loop Scan-n-Print

### Pre-request

* Data curve
* Robot definition files
* Welding and scanning joint space trajectory [here](https://github.com/arminstitute/ARM-TEC-22-01-F-07/tree/main/redundancy_resolution)

### Hardware requirements

* Motoman robots
* Fronius welder
* MTI Scanner
* Flir thermal camera
* Microphone

### Execution

* Wall geometry scan-n-print
```
python310.exe weld_wall_scan.py
```

* Complex geometry scan-n-print
```
python310.exe weld_geometry_scan.py
```

### Helper function

These codes/algorithms are used in the main execution codes. (Go to folder ```scan/``` for scanning codes.)

* ```weld_dh2v.py``` The code computes deposition height given the torch speed or vice versa for three metals used in the project.
* ```weldCorrectionStrategy.py``` The code includes scan-n-print correction strategies given the current scanned results and the target deposition height.
* ```weldRRsensor.py``` A compact function to include all sensor interfaces.
* ```WeldScan.py``` A compact function to execute a robot trajectory given the joint space trajectory and breakpoints.
* ```scanPathGen.py``` Given the weld curve, generate cartesian scanning path and the corresponding joint space trajecotries.
* ```scanProcess.py``` Given the scanned points and the corresponding robot joints, generates the 3D point clouds and deposition height.