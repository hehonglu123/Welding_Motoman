import cv2
import pickle, sys
import numpy as np
sys.path.append('../../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# Load the IR recording data from the pickle file
data_dir='../../../recorded_data/ER316L/wallbf_100ipm_v10_100ipm_v10/'
# data_dir='../../../recorded_data/wall_weld_test/4043_150ipm_2024_06_18_11_16_32/layer_4/'


with open(data_dir+'/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)
ir_ts=np.loadtxt(data_dir+'/ir_stamps.csv', delimiter=',')

#load template
template = cv2.imread('torch_template.png',0)


# Create a window to display the images
cv2.namedWindow("IR Recording", cv2.WINDOW_NORMAL)

# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO

frame=639
ir_image = np.rot90(ir_recording[frame], k=-1)
######################################################################################################################
threshold=max(1.5e4,0.5*np.max(ir_image))
print(threshold)
thresholded_img=(ir_image>threshold).astype(np.uint8)
nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
#find the largest connected area
# The fourth column of the stats matrix contains the area (in pixels) of each connected component
areas = stats[:, 4]
areas[0] = 0    # Exclude the background component (label 0) from the search

# Find the index of the component with the largest area
largest_component_index = np.argmax(areas)




######################################################################################################################
ir_normalized = ((ir_image - np.min(ir_image)) / (np.max(ir_image) - np.min(ir_image))) * 255
ir_normalized=np.clip(ir_normalized, 0, 255)

# Convert the IR image to BGR format with the inferno colormap
ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)

#make all the pixels of the largest connected component green
ir_bgr[labels == largest_component_index] = [0, 255, 0]
pixel_coordinates = np.flip(np.array(np.where(labels == largest_component_index)).T,axis=1)


#find the closest point in the region that's closest to the template
template_upper_corner=torch_detect(ir_image,template)
# add bounding box of the torch
cv2.rectangle(ir_bgr, template_upper_corner, (template_upper_corner[0] + template.shape[1], template_upper_corner[1] + template.shape[0]), (0,255,0), 2)

template_bottom_center=template_upper_corner+np.array([template.shape[1]//2,template.shape[0]])
ir_bgr[template_bottom_center[1],template_bottom_center[0]]=[0,0,255]

#convert the shape of pixel_coordinates to convex hull
hull = cv2.convexHull(pixel_coordinates)
#visualize the hull
# for i in range(len(hull)):
#     ir_bgr[hull[i][0][1],hull[i][0][0]]=[255,0,0]

#find the point on the hull that is shortest to the template bottom center
poly = Polygon([tuple(point[0]) for point in hull])
point = Point(template_bottom_center[0],template_bottom_center[1])
# The points are returned in the same order as the input geometries:
p1, p2 = nearest_points(poly, point)
print(p1.wkt)
torch_tip_coordinate=np.array([p1.x,p1.y]).astype(int)
# torch_tip_idx=np.argmin(np.linalg.norm(pixel_coordinates-np.flip(template_bottom_center),axis=1))
 
#make torch_tip_idx white
ir_bgr[torch_tip_coordinate[1],torch_tip_coordinate[0]]=[255,0,0]

# Display the IR image
cv2.imshow("IR Recording", ir_bgr)
cv2.waitKey(0)