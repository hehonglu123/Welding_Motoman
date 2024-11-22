import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import cvxpy as cp

# Generate random points
points = np.random.rand(30, 2)

# Compute the minimal area polygon (not necessarily convex)
def compute_minimal_area_polygon(points):
    n = len(points)
    x = cp.Variable(n, boolean=True)
    
    # Objective: Minimize the number of points in the polygon
    objective = cp.Minimize(cp.sum(x))
    
    # Constraints to ensure the selected points form a valid polygon
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-3:
                constraints.append(x[i] + x[j] <= 1)
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    # Get the selected points for the polygon
    selected_points = points[np.where(x.value > 0.5)[0]]
    return selected_points

polygon_points = compute_minimal_area_polygon(points)

print(polygon_points)

# Function to check if a point is inside the polygon
def is_point_in_polygon(point, polygon_points):
    delaunay = Delaunay(polygon_points)
    return delaunay.find_simplex(point) >= 0

# Example point to test
new_point = np.array([0.5, 0.5])

# Check if the new point is inside the polygon
inside = is_point_in_polygon(new_point, polygon_points)
print(f"The point {new_point} is inside the polygon: {inside}")

# Plot the points, the minimal area polygon, and the new point
plt.plot(points[:, 0], points[:, 1], 'o')
plt.plot(polygon_points[:, 0], polygon_points[:, 1], 'k-')
plt.fill(polygon_points[:, 0], polygon_points[:, 1], edgecolor='k', fill=False)

# Plot the new point in red
plt.plot(new_point[0], new_point[1], 'ro' if inside else 'bo')
plt.title("Minimal Area Polygon and Point Check")
plt.show()
