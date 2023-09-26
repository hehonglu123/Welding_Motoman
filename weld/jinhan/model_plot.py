import numpy as np
import matplotlib.pyplot as plt

# Define the models
models = {
    # #ER 4043
    # "100ipm": [-0.62015, 1.84913],
    # "140ipm": [-0.5068, 1.643],
    # "160ipm": [-0.4619, 1.647],
    # "180ipm": [-0.3713, 1.506],
    # "190ipm": [-0.5784, 1.999],
    # "200ipm": [-0.5776, 2.007],
    # "210ipm": [-0.5702, 1.990],
    # "220ipm": [-0.5699, 1.985],
    # "230ipm": [-0.5374, 1.848],

    #ER 70S-6
    "100ipm": [-0.31828998,  0.68503243],
    "110ipm": [-0.27499103,  0.63182495],
    "120ipm": [-0.31950134,  0.69261567],
    "130ipm": [-0.31630631,  0.70834374],
    "140ipm": [-0.31654673,  0.74122273],
    "150ipm": [-0.31903634,  0.78416199],
    "160ipm": [-0.31421562,  0.81825734],
    "170ipm": [-0.22845064,  0.77327933],
    "180ipm": [-0.20186512,  0.78154183],
    "190ipm": [-0.2810107 ,  0.88758474],
    # "200ipm": [-0.29245818,  0.99450003],
    # "210ipm": [-0.5702, 1.990],
    # "220ipm": [-0.5699, 1.985],
    "230ipm": [-0.29245818,  0.99450003],
    "240ipm": [-0.31196034,  1.03601865],
    "250ipm": [-0.27141449,  1.03156706]

    # #316L
    # "100ipm": [-0.62015, 1.84913],
    # "110ipm": [-0.62015, 1.84913],
    # "120ipm": [-0.62015, 1.84913],
    # "130ipm": [-0.62015, 1.84913],
    # "140ipm": [-0.5068, 1.643],
    # "150ipm": [-0.62015, 1.84913],
    # "160ipm": [-0.4619, 1.647],
    # "170ipm": [-0.62015, 1.84913],
    # "180ipm": [-0.3713, 1.506],
    # "190ipm": [-0.5784, 1.999],
    # "200ipm": [-0.5776, 2.007],
    # "210ipm": [-0.5702, 1.990],
    # "220ipm": [-0.5699, 1.985],
    # "230ipm": [-0.5374, 1.848],
    # "240ipm": [-0.62015, 1.84913],
    # "250ipm": [-0.62015, 1.84913]
}

# Generate the data
logV_full = np.linspace(1.5, 3.5, 400)
logV_partial = np.linspace(1.5, 2.6, 400)

plt.figure(figsize=(10, 6))

# Plot each model
for label, (a, b) in models.items():
    # if label in ["100ipm","140ipm", "160ipm", "180ipm"]:
    if label in []:
        logV = logV_partial
    else:
        logV = logV_full

    logDh = a * logV + b
    plt.plot(logV, logDh, label=label)


# Customize the plot
plt.xlabel("log(V) (log(mm/s))", fontsize=20)
plt.ylabel("log(Δh) (log(mm))", fontsize=20)
plt.title("ER 70S-6 Deposition Model", fontsize=20)
# plt.xlim(1.5, 3.5)
# plt.ylim(0, 1.2)
# Set finer grid
ax = plt.gca()
# ax.xaxis.set_ticks(np.arange(1.5, 3.6, 0.5))  # Change 0.1 to any value for desired grid size on x-axis
# ax.yaxis.set_ticks(np.arange(0, 1.3, 0.25))   # Change 0.05 to any value for desired grid size on y-axis

# Adjust grid line thickness
ax.xaxis.grid(True, linewidth=1.8)  # Increase linewidth for thicker grid lines
ax.yaxis.grid(True, linewidth=1.8)  # Increase linewidth for thicker grid lines


plt.grid(True)
# Customize legend
legend = plt.legend(fontsize=9, loc="upper right")  # Adjust fontsize as needed and specify location
frame = legend.get_frame()
frame.set_linewidth(1.0)  # Adjust legend box border thickness

# Adjust tick label fontsize
ax.tick_params(axis="both", labelsize=18)  # Adjust fontsize as needed

# Display the plot
plt.tight_layout()
plt.show()
