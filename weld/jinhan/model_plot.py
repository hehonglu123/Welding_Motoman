import numpy as np
import matplotlib.pyplot as plt

# Define the models
models = {
    "100ipm": [-0.62015, 1.84913],
    "140ipm": [-0.5068, 1.643],
    "160ipm": [-0.4619, 1.647],
    "180ipm": [-0.3713, 1.506],
    "190ipm": [-0.5784, 1.999],
    "200ipm": [-0.5776, 2.007],
    "210ipm": [-0.5702, 1.990],
    "220ipm": [-0.5699, 1.985],
    "230ipm": [-0.5374, 1.848]
}

# Generate the data
logV_full = np.linspace(1.5, 3.5, 400)
logV_partial = np.linspace(1.5, 2.6, 400)

plt.figure(figsize=(10, 6))

# Plot each model
for label, (a, b) in models.items():
    if label in ["100ipm","140ipm", "160ipm", "180ipm"]:
        logV = logV_partial
    else:
        logV = logV_full

    logDh = a * logV + b
    plt.plot(logV, logDh, label=label)

# Customize the plot
plt.xlabel("log(V) (log(mm/s))")
plt.ylabel("log(Δh) (log(mm))")
plt.title("Al 4043 Deposition Model")
plt.xlim(1.5, 3.5)
plt.ylim(0, 1.2)
# Set finer grid
ax = plt.gca()
ax.xaxis.set_ticks(np.arange(1.5, 3.6, 0.5))  # Change 0.1 to any value for desired grid size on x-axis
ax.yaxis.set_ticks(np.arange(0, 1.3, 0.25))   # Change 0.05 to any value for desired grid size on y-axis

plt.grid(True)
plt.legend()


# Display the plot
plt.tight_layout()
plt.show()
