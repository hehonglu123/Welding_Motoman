import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


z_offset=4
line_width=4
# Define the range and resolution of the surface
z = np.linspace(0, 100, 100)
theta = np.linspace(0, 2*np.pi, 100)
Z, Theta = np.meshgrid(z, theta)

# Calculate the XY values based on the circle equation
X = 10*(Z+z_offset)**(1/4) * np.cos(Theta)
Y = 10*(Z+z_offset)**(1/4) * np.sin(Theta)
radius_outter = 10*z_offset**(1/4)
# Z-=z_offset

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_box_aspect([1, 1, 1])  # Adjust the values as needed

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot with Circular XY')

num_circle=int(np.ceil(radius_outter/line_width))
radii=np.linspace(line_width/2,radius_outter,num_circle)
for r in radii:
    theta = np.linspace(0, 2 * np.pi, 100)
    XX = r * np.cos(theta)
    YY = r * np.sin(theta)
    ZZ = np.zeros_like(XX)
    ax.plot(XX, YY, ZZ, label='Circle', color='red')

# Show the plot
plt.show()