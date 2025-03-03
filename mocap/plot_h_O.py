import numpy as np
# draw 3D with matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from general_robotics_toolbox import *

h_nom = np.array([0.0, 0.0, 1])
k1 = np.array([1, 0, 0])
k2 = np.array([0, 1, 0])
h_act = rot(k1, np.radians(-12))@rot(k2, np.radians(12)) @ h_nom
shift_O = np.array([0.4, 0.6, 0])
Oi_prime = np.array([-h_act[0]+shift_O[0]+h_act[0]/3, -h_act[1]+shift_O[1]+h_act[1]/3, -h_act[2]+shift_O[2]+h_act[2]/3])
O0 = np.array([-0.5,-0.5,-1])*0.8

# plot h k1 k2
head_scale = 0.1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# draw a plane at z=0
plan_start=-0.5
plan_end=1.5
x = np.linspace(plan_start, plan_end, 50)
y = np.linspace(plan_start, plan_end, 50)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)
ax.plot_surface(x, y, z, alpha=0.5, color='gray')

ax.quiver(-h_nom[0], -h_nom[1], -h_nom[2], h_nom[0], h_nom[1], h_nom[2], color='lightcoral', length=2, arrow_length_ratio = head_scale)
ax.quiver(-h_act[0]+shift_O[0], -h_act[1]+shift_O[1], -h_act[2]+shift_O[2], h_act[0], h_act[1], h_act[2], color='red', length=2, arrow_length_ratio = head_scale)
ax.quiver(0, 0, 0, k1[0], k1[1], k1[2], length=1, color='lime',arrow_length_ratio = head_scale*2)
ax.quiver(0, 0, 0, k2[0], k2[1], k2[2], length=1, color='forestgreen', arrow_length_ratio = head_scale*2)
ax.scatter(0,0,0, color='tab:blue', s=100)
ax.scatter(shift_O[0],shift_O[1],shift_O[2], color='tab:blue', s=100)
ax.scatter(Oi_prime[0],Oi_prime[1],Oi_prime[2], color='lightskyblue', s=100)
ax.scatter(O0[0],O0[1],O0[2], color='tab:blue', s=100)
ax.quiver(O0[0],O0[1],O0[2], -O0[0], -O0[1], -O0[2], color='dodgerblue', length=1, arrow_length_ratio = head_scale)
ax.quiver(O0[0],O0[1],O0[2], shift_O[0]-O0[0], shift_O[1]-O0[1], shift_O[2]-O0[2], color='dodgerblue', length=1, arrow_length_ratio = head_scale)
ax.quiver(O0[0],O0[1],O0[2], Oi_prime[0]-O0[0], Oi_prime[1]-O0[1], Oi_prime[2]-O0[2], color='lightskyblue', length=1, arrow_length_ratio = head_scale)

ax.set_xlim([plan_start, plan_end])
ax.set_ylim([plan_start, plan_end])
ax.set_zlim([-1, 1])
# without grid and tixes
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.show()