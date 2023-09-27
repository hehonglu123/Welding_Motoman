import cv2
import pickle, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../toolbox/')
from flir_toolbox import *

def normalize2cv(frame):
    ir_normalized = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame))) * 255
    return ir_normalized.astype(np.uint8)


# Load the IR recording data from the pickle file
with open('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_2/ir_recording.pickle', 'rb') as file:
    ir_recording = pickle.load(file)

ir_ts=np.loadtxt('../../../recorded_data/316L_model_120ipm_2023_09_25_19_56_43/layer_2/ir_stamps.csv', delimiter=',')


# Set the colormap (inferno) and normalization range for the color bar
cmap = cv2.COLORMAP_INFERNO
colorbar_min = np.min(ir_recording)
colorbar_max = np.max(ir_recording)
fig = plt.figure(1)
cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'XVID'),
                         30, (320,240))
for i in range(0,len(ir_recording)):
    # ir_colormap = cv2.applyColorMap(normalize2cv(ir_recording[i]), cv2.COLORMAP_INFERNO)
    # Display the image using matplotlib
    plt.imshow(ir_recording[i], cmap='inferno')

    # Add contour lines
    # You can adjust the number of contour levels or specify particular values
    contour_levels = 5  # or you could provide a list like [50, 100, 150, 200]
    # contour_levels=[100,200,300]
    contour_paths = plt.contour(normalize2cv(ir_recording[i]), levels=contour_levels, colors='black', origin='lower')
    # plt.axis('off')
    # plt.pause(0.001)
    # plt.clf()

    # Get the isolines as a list of numpy arrays
    isolines = []
    for _, collection in enumerate(contour_paths.collections):
        for path in collection.get_paths():
            try:
                vertices = path.to_polygons()[0]
            except IndexError:
                continue
            vertices = vertices.astype(int)  # Convert vertices to integers

            # Convert matplotlib path to OpenCV contour
            contour = vertices.reshape((-1, 1, 2))

            # Add the contour to the list
            isolines.append(contour)
            
    ir_contour = cv2.applyColorMap(normalize2cv(ir_recording[i]), cv2.COLORMAP_INFERNO)
    for idx, isoline in enumerate(isolines):
        # Draw the isoline on the ir_colormap image
        cv2.drawContours(ir_contour, [isoline], -1, (0,255,0), 1)
    # Display or save
    # cv2.imshow('Contours', ir_contour)
    # cv2.waitKey(int(1000*(ir_ts[i+1]-ir_ts[i])))

    result.write(ir_contour)
result.release()

cv2.destroyAllWindows()

