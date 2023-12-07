import matplotlib.pyplot as plt
import numpy as np

matrix = np.random.rand(10, 10)
fig, ax = plt.subplots()
im = ax.matshow(matrix, cmap='RdBu', vmin=-1.2, vmax=1.2, interpolation='none')
cbar = fig.colorbar(im, ax=ax, extend='both')
cbar.minorticks_on()
for i in range(10):
    for j in range(10):
        c = matrix[j,i]
        if c == 0:
            ax.text(i, j, str(c), va='center', ha='center', color='black')
        else:
            ax.text(i, j, str(c), va='center', ha='center')
plt.show()