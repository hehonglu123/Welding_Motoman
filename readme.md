# Convergent Manufacturing for WAAM Process
<img src="images/ER4043_blade_ir_lapse.gif" alt="Alt Text" width="100" height="auto">
<img src="images/ER4043_bell_ir_lapse.gif" alt="Alt Text" width="100"  height="auto">

## Dependencies
`python311 -m pip install -r requirements.txt`

<img src="images/architecture_red.png" alt="Alt Text" width="400" height="auto">

## Usage
### INFORM 
* [DX200 Driver](https://github.com/hehonglu123/dx200_motion_progam_exec):   Create Motoman INFORM code (*.JBI) and upload it to the robot controller
* Helper Functions:   `toolbox/WeldSend.py`
### MotoPlus Streaming with RobotRaconteur
* MotoPlus File (.out): Loaded into DX200 Motoplus Functions, started automatically 
* MotoPlus RR Driver: Running on separate Ubuntu computer (ubuntu-motoman@ubuntu-motoman (192.168.55.15))
`./run_RR_robot.bat` or `python311 -m motoman_robotraconteur_driver --robot-info-file=../config/rpi_waam_testbed.yml --robot-ip-address=192.168.1.31`
* [Fronius RR Driver](https://github.com/johnwason/fronius_robotraconteur_driver): controls the Fronius welding parameter separately on a raspberry pi (fronius-pi@fronius-pi (192.168.55.21)) with wired E-stop 
     `./run_fronius_control.bat` or `python310 -m fronius_robotraconteur_driver --welder-ip=192.168.1.51 --welder-info=../config/fronius_tps_500i_default_config.yml`
* Helper Functions: `toolbox/StreamingSend.py`

## Slicing
Non-planar slicing from CAD with uniform deposition rate. WAAM requires torch deposition along gravity direction, so the slicing needs to be support free and along the surface tangent direction.

### Baseline
We used NX to extract curves on the edges and `curve offset` function to push curve along surface tangent direction by a fixed amount.

<img src="images/surface_cad.jpg" alt="Alt Text" width="200" height="auto">
<img src="images/nx_slicing.png" alt="Alt Text" width="200" height="auto">

After all slicing created in NX, we used NX-open to sample each curve as Cartesian points. Curve normal at each point is generated from perpendicular vector pointing toward previous curve.
The sliced layers are then further tested for WAAM process successfully.

### Automated Slicer
Similar idea as baseline but the process is automated with `numpy-stl`, with user specified layer height. We used first order approximation to offset curves along surface tangent direciton and projected back to the surface.

<img src="images/slicing.png" alt="Alt Text" width="200" height="auto">



## Motion Planning

Joint space redundancy resolution with gravity constraints.

Motion primitive planning.

<img src="images/welding.jpg" alt="Alt Text" width="200" height="auto">



## WAAM Process

Welding parameter tunning.

<img src="images/ss_welding.jpg" alt="Alt Text" width="200" height="auto">
<img src="images/ss_blade.jpg" alt="Alt Text" width="200" height="auto">

### Sensor Monitoring


## Product Inspection

<img src="images/artec.png" alt="Alt Text" width="50" height="auto">

```stl

solid cube_corner
  facet normal 0.0 -1.0 0.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 0.0 0.0 1.0
    endloop
  endfacet
  facet normal 0.0 0.0 -1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 0.0 1.0 0.0
      vertex 1.0 0.0 0.0
    endloop
  endfacet
  facet normal -1.0 0.0 0.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 0.0 0.0 1.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
  facet normal 0.577 0.577 0.577
    outer loop
      vertex 1.0 0.0 0.0
      vertex 0.0 1.0 0.0
      vertex 0.0 0.0 1.0
    endloop
  endfacet
endsolid

```

<img src="images/scanned_blade.jpg" alt="Alt Text" width="200" height="auto">
<img src="images/scan_eval.jpg" alt="Alt Text" width="200" height="auto">
