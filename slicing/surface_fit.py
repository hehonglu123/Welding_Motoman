from stl import mesh
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the STL file
filename = '../data/blade0.1/surface.stl'
your_mesh = mesh.Mesh.from_file(filename)

# Extract all vertices
vertices = np.zeros((len(your_mesh), 3, 3))
for i, facet in enumerate(your_mesh.vectors):
    vertices[i] = facet
# Flatten the vertices array and remove duplicates
unique_vertices = np.unique(vertices.reshape(-1, 3), axis=0)
x=unique_vertices[:,0]
y=unique_vertices[:,1]
z=unique_vertices[:,2]

# Polynomial regression
degree = 50
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
X = np.column_stack((x, y))
poly_model.fit(X, z)

# Predicting the fitted curved plane
zz = poly_model.predict(np.column_stack((x, y)))

# Plot the original points and the fitted curved plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, c='r', marker='o', label='Original points')
surf = ax.plot_trisurf(x,y,zz, linewidth=0, antialiased=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()