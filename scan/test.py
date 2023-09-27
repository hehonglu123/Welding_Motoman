import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Create a dummy ScalarMappable with the desired colormap
norm = Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])  # This is required as it's a dummy instance.

# Create a figure
fig, ax = plt.subplots(figsize=(2, 4))

# Display only the colorbar
cbar = fig.colorbar(sm, ax=ax)

plt.show()