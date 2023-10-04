import numpy as np
import matplotlib.pyplot as plt

coeff1=100
coeff2=40
final_radius=40
total_height=60
def y_of_x(x, A=final_radius, sigma=30):
    # Gaussian function defined for x >= 0
    if x >= 0:
        return A * np.exp(-(coeff1-x-coeff2)**2 / (2*sigma**2))
    else:
        return None

# Vectorize the function for array input
y_of_x_vectorized = np.vectorize(y_of_x)

# Generate x values
x_values = np.linspace(0, total_height, 400)
y_values = y_of_x_vectorized(x_values)

# Plot
plt.plot(x_values, y_values, label='y = 60 * exp(-x^2 / (2*30^2))')
plt.title('Stretched Flipped Gaussian')
plt.xlabel('x')
plt.ylabel('y')

plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()
