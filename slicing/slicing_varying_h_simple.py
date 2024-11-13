import glob
from copy import deepcopy
import pickle
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

def is_point_in_triangle(p, a, b, c):
    # Unpack points
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c

    # Compute vectors
    v0 = [cx - ax, cy - ay]
    v1 = [bx - ax, by - ay]
    v2 = [px - ax, py - ay]

    # Compute dot products
    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False  # Triangle is degenerate
    inv_denom = 1 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if point is inside the triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)

layer_heights = np.arange(0, 30.1, 0.1)
fork_triangle1 = np.array([[0,15],[24,15],[12,30]])
fork_triangle2 = np.array([[48,15],[72,15],[60,30]])

for h in layer_heights:
    sliced_sections = [[],[]]
    for x in range(0,72.1,0.1):
        if h<=15:
            sliced_sections[0].append(np.array([x,0,h,0,0,-1]))
        else:
            point = np.array([x,h])
            # check if the points is inside triangles
            if is_point_in_triangle(point, fork_triangle1[0], fork_triangle1[1], fork_triangle1[2]):
                sliced_sections[0].append(np.array([x,0,h,0,0,-1]))
            elif is_point_in_triangle(point, fork_triangle2[0], fork_triangle2[1], fork_triangle2[2]):
                sliced_sections[1].append(np.array([x,0,h,0,0,-1]))
            else:
                pass
    for i, section in enumerate(sliced_sections):
        if len(section)>0:
            section = np.array(section)
            np.savetxt('../fork/curve_sliced/slice_%d_%d.csv'%(h,i),section,delimiter=',')
            