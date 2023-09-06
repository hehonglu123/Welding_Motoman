import matplotlib.pyplot as plt
import numpy as np

def parametrize_hexagon_perimeter(t, a):
    x = np.zeros_like(t)
    y = np.zeros_like(t)

    # Parametric equations for each side of the hexagon
    mask_1 = (t >= 0) & (t < np.pi/3)
    mask_2 = (t >= np.pi/3) & (t < 2*np.pi/3)
    mask_3 = (t >= 2*np.pi/3) & (t < np.pi)
    mask_4 = (t >= np.pi) & (t < 4*np.pi/3)
    mask_5 = (t >= 4*np.pi/3) & (t < 5*np.pi/3)
    mask_6 = (t >= 5*np.pi/3) & (t <= 2*np.pi)

    x[mask_1] = a * t[mask_1]
    y[mask_1] = a * np.sqrt(3)

    x[mask_2] = a * (t[mask_2] - np.pi/3)
    y[mask_2] = -a * (t[mask_2] - np.pi/3) * np.sqrt(3)

    x[mask_3] = -a * (t[mask_3] - np.pi)
    y[mask_3] = -a * np.sqrt(3)

    x[mask_4] = -a * (t[mask_4] - 4*np.pi/3)
    y[mask_4] = a * (t[mask_4] - 4*np.pi/3) * np.sqrt(3)

    x[mask_5] = a * (t[mask_5] - 5*np.pi/3)
    y[mask_5] = a * np.sqrt(3)

    x[mask_6] = a * (t[mask_6] - 2*np.pi)
    y[mask_6] = -a * (t[mask_6] - 2*np.pi) * np.sqrt(3)

    return x, y

# Side length of the hexagon
a = 1.0

# Generate points on the hexagon perimeter for t ranging from 0 to 2*pi
num_points = 1000
t_values = np.linspace(0, 2*np.pi, num_points)
x_values, y_values = parametrize_hexagon_perimeter(t_values, a)

# Plot the hexagon perimeter
plt.figure()
plt.plot(x_values, y_values, 'b-', label='Regular Hexagon')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regular Hexagon Perimeter')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
