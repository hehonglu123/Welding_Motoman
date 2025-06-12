import numpy as np
from matplotlib import pyplot as plt

inch2mm = 25.4  # conversion factor from inches to mm
mm2inch = 1/inch2mm  # conversion factor from mm to inches

def welding_profile_generate_smooth(feedrate_nom, VPD, cross_section, lam_max):
    
    feedrate_nom = int(round(feedrate_nom/10)*10) # round to 10, make sure it is a multiple of 10
    feedrate_min = feedrate_nom-5
    feedrate_max = feedrate_nom+5
    vel_nom = cross_section*inch2mm*feedrate_nom/VPD # get nominal velocity
    vel_min = cross_section*inch2mm*feedrate_min/VPD
    vel_max = cross_section*inch2mm*feedrate_max/VPD

    vel_profile = np.linspace(vel_min, vel_max, np.round(lam_max/vel_nom).astype(int)) # velocity profile around nominal velocity
    feedrate_profile = [feedrate_nom]*len(vel_profile) # feedrate profile is constant
    return vel_profile, feedrate_profile

lam_max = 110 # maximum length of the weld in mm
layer_feedrate = 100
layer_nom_vel = 10 # mm/s
cross_section = 1.14

VPD = cross_section*inch2mm*layer_feedrate/layer_nom_vel # volume per distance (mm^3/mm)

feedrate_layers = np.arange(50,201,10).astype(int) # inch/min
feedrate_layers = feedrate_layers[::-1] # always start from the highest feedrate (highest velocity)

VPD_start = VPD * 0.5 # starting VPD for the first layer

feedrate_samples_all = []
vel_samples_all = []
for vpd_sample_ratio in np.arange(0,9,1):
    print(f'VPD sample ratio: {vpd_sample_ratio}')
    feedrate_samples=[]
    vel_samples=[]
    this_VPD = VPD_start * (np.sqrt(np.sqrt(2))**vpd_sample_ratio)
    for fdr in feedrate_layers:
        vel_profile, feedrate_profile = welding_profile_generate_smooth(fdr, this_VPD, cross_section, lam_max)
        feedrate_samples.extend(feedrate_profile)
        vel_samples.extend(vel_profile)
    plt.scatter(vel_samples, feedrate_samples, s=2)
plt.xlabel('Velocity (mm/s)')
plt.ylabel('Feedrate (inch/min)')
plt.title('Welding Profile Generation')
plt.grid()
plt.ylim(40, 210)
plt.show()