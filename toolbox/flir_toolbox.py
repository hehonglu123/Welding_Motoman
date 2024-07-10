import numpy as np
import cv2, copy
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points
import matplotlib.pyplot as plt



##############################################FLIR COUNTS TO TEMPERATURE CONVERSION##############################################
####FROM https://flir.custhelp.com/app/answers/detail/a_id/3321/~/the-measurement-formula
def counts2temp(data_counts,R,B,F,J1,J0,Emiss):
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp


def counts2temp_4learning(data_counts,R,B,F,J1,J0):
    Emiss=data_counts[-1]
    data_counts=data_counts[:-1]
    # reflected energy
    TRefl = 18

    # atmospheric attenuation
    TAtmC = 20
    TAtm = TAtmC + 273.15
    Tau = 0.99 #transmission

    # external optics
    TExtOptics = 20
    TransmissionExtOptics = 1
  
    K1 = 1 / (Tau * Emiss * TransmissionExtOptics)
        
    # Pseudo radiance of the reflected environment
    r1 = ((1-Emiss)/Emiss) * (R/(np.exp(B/TRefl)-F))
    # Pseudo radiance of the atmosphere
    r2 = ((1 - Tau)/(Emiss * Tau)) * (R/(np.exp(B/TAtm)-F)) 
    # Pseudo radiance of the external optics
    r3 = ((1-TransmissionExtOptics) / (Emiss * Tau * TransmissionExtOptics)) * (R/(np.exp(B/TExtOptics)-F))
            
    K2 = r1 + r2 + r3
    
    data_obj_signal = (data_counts - J0)/J1
    data_temp = (B / np.log(R/((K1 * data_obj_signal) - K2) + F)) -273.15
    
    return data_temp


def flame_detection(raw_img,threshold=1.2e4,area_threshold=10):
    ###flame detection by raw counts thresholding and connected components labeling
    #centroids: x,y
    #bbox: x,y,w,h
    thresholded_img=(raw_img>threshold).astype(np.uint8)
    if np.max(thresholded_img)==0:
        return None, None

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)

    valid_indices=np.where(stats[:, cv2.CC_STAT_AREA] > area_threshold)[0][1:]  ###threshold connected area
    if len(valid_indices)==0:
        return None, None
    
    average_pixel_values = [np.mean(raw_img[labels == label]) for label in valid_indices]   ###sorting

    valid_index=valid_indices[np.argmax(average_pixel_values)]      ###get the area with largest average brightness value

    # Extract the centroid and bounding box of the largest component
    centroid = centroids[valid_index]
    bbox = stats[valid_index, :-1]

    return centroid, bbox


def flame_detection_no_arc(raw_img,torch_template,threshold=1.5e4,area_threshold=10,template_center_offset=2):
    ###welding point detection without flame
    #centroids: x,y
    #bbox: x,y,w,h

    ###adaptively increase the threshold to 50% of the maximum pixel value
    threshold=max(threshold,0.5*np.max(raw_img))


    thresholded_img=(raw_img>threshold).astype(np.uint8)
    if np.max(thresholded_img)==0:      #if no pixel above threshold, means not welding 
        print('no hotspot detected')
        return None, None


    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
    #find the largest connected area
    areas = stats[:, 4]
    areas[0] = 0    # Exclude the background component (label 0) from the search

    if np.max(areas)<area_threshold:    #if no hot spot larger than area_threshold, return None
        print('hotspot too small')
        return None, None
    
    # Find the index of the component with the largest area
    largest_component_index = np.argmax(areas)
    pixel_coordinates = np.flip(np.array(np.where(labels == largest_component_index)).T,axis=1)

    ## Torch template detection
    template_upper_corner=torch_detect(raw_img,torch_template)
    if template_upper_corner is None:   #if no torch detected, return None
        print('torch not found')
        return None, None
    template_bottom_center=template_upper_corner+np.array([torch_template.shape[1]//2+template_center_offset,torch_template.shape[0]])
    hull = cv2.convexHull(pixel_coordinates)

    ##############################################display for debugging#########################################################
    # ir_normalized = ((raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))) * 255
    # ir_normalized=np.clip(ir_normalized, 0, 255)
    # # Convert the IR image to BGR format with the inferno colormap
    # ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    # cv2.rectangle(ir_bgr, template_upper_corner, (template_upper_corner[0] + torch_template.shape[1], template_upper_corner[1] + torch_template.shape[0]), (0,255,0), 2)
    # #change convex hull vertices to green
    # for i in range(len(hull)):
    #     cv2.circle(ir_bgr, tuple(hull[i][0]), 1, (0,255,0), thickness=2)

    # cv2.imshow('ir_bgr',ir_bgr)
    # cv2.waitKey(0)
    ##############################################display for debugging END#########################################################

    poly = Polygon([tuple(point[0]) for point in hull])
    point = Point(template_bottom_center[0],template_bottom_center[1])
    # The points are returned in the same order as the input geometries:
    weld_pool, p2 = nearest_points(poly, point)
    centroid=np.array([weld_pool.x,weld_pool.y]).astype(int)

    #create 5x5 bbox around the centroid
    bbox=np.array([centroid[0]-2,centroid[1]-2,5,5])


    return centroid, bbox



def flame_detection_yolo(raw_img,yolo_model,threshold=1.5e4,area_threshold=10,percentage_threshold=0.6):
    ###welding point detection without flame
    #centroids: [x,y], top pixel coordinate of the weldpool (intersection between wire and piece)
    #bbox: x,y,w,h

    ###adaptively increase the threshold to 60% of the maximum pixel value
    threshold=max(threshold,percentage_threshold*np.max(raw_img))


    thresholded_img=(raw_img>threshold).astype(np.uint8)
    if np.max(thresholded_img)==0:      #if no pixel above threshold, means not welding 
        print('no hotspot detected')
        return None, None


    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)
    #find the largest connected area
    areas = stats[:, 4]
    areas[0] = 0    # Exclude the background component (label 0) from the search

    if np.max(areas)<area_threshold:    #if no hot spot larger than area_threshold, return None
        print('hotspot too small')
        return None, None
    
    # Find the index of the component with the largest area
    largest_component_index = np.argmax(areas)
    pixel_coordinates = np.flip(np.array(np.where(labels == largest_component_index)).T,axis=1)

    ## Torch detection
    torch_centroid, torch_bbox=torch_detect_yolo(raw_img,yolo_model)
    if torch_centroid is None:   #if no torch detected, return None
        print('torch not found')
        return None, None, None, None
    template_bottom_center=torch_bbox[:2]+np.array([torch_bbox[2]/2,torch_bbox[3]])
    hull = cv2.convexHull(pixel_coordinates)

    poly = Polygon([tuple(point[0]) for point in hull])
    point = Point(template_bottom_center[0],template_bottom_center[1])
    
    ###find the intersection between the line and the hull
    downward_line = LineString([point, (template_bottom_center[0], 320)])
    # Find the intersection between the line and the polygon's hull
    intersection = poly.exterior.intersection(downward_line)
    if not intersection.is_empty:
        if intersection.geom_type == 'MultiPoint':
            # Convert MultiPoint to a list of points and find the one with the lowest y-value
            weld_pool = min(intersection.geoms, key=lambda p: p.y)
        else:
            weld_pool = intersection
    else:
        ### find the closest point on the hull
        weld_pool, _ = nearest_points(poly, point)



    centroid=np.array([weld_pool.x,weld_pool.y]).astype(int)
    #create 5x5 bbox around the centroid
    bbox=np.array([centroid[0]-2,centroid[1]-2,5,5])
    

    # ##############################################display for debugging#########################################################
    # #plot out the convex hull and the template bottom center
    # # plt.plot(*poly.exterior.xy)
    # # plt.scatter(*point.xy, c='r')
    # # plt.plot(*downward_line.xy)
    # # plt.show()
    # ir_normalized = ((raw_img - np.min(raw_img)) / (np.max(raw_img) - np.min(raw_img))) * 255
    # ir_normalized=np.clip(ir_normalized, 0, 255)
    # # Convert the IR image to BGR format with the inferno colormap
    # ir_bgr = cv2.applyColorMap(ir_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    # cv2.rectangle(ir_bgr, tuple(bbox[:2]), tuple(bbox[:2]+bbox[2:]), (0,255,0), 2)
    # #change convex hull vertices to green
    # for i in range(len(hull)):
    #     cv2.circle(ir_bgr, tuple(hull[i][0]), 1, (255,0,0), thickness=2)
    # #make template bottom center red
    # cv2.circle(ir_bgr, tuple(map(int, template_bottom_center)), 1, (0,0,255), thickness=2)


    # cv2.imshow('ir_bgr',ir_bgr)
    # cv2.waitKey(0)
    # ##############################################display for debugging END#########################################################


    return centroid, bbox, torch_centroid, torch_bbox

def weld_detection(raw_img,threshold=1.2e4,area_threshold=10):
    ###flame detection by raw counts thresholding and connected components labeling
    #centroids: x,y
    #bbox: x,y,w,h
    #pixels: row,col coordinates

    threshold=max(threshold,np.max(raw_img)*0.9)
    thresholded_img=(raw_img>threshold).astype(np.uint8)
    if np.max(thresholded_img)==0:
        # print('max counts below threshold: ',np.max(raw_img))
        return None, None, None

    nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_img, connectivity=4)

    valid_indices=np.where(stats[:, cv2.CC_STAT_AREA] > area_threshold)[0][1:]  ###threshold connected area
    if len(valid_indices)==0:
        # print('no connected components')
        return None, None, None
    
    average_pixel_values = [np.mean(raw_img[labels == label]) for label in valid_indices]   ###sorting

    valid_index=valid_indices[np.argmax(average_pixel_values)]      ###get the area with largest average brightness value

    #list of pixel coordinates
    pixels=np.where(labels==valid_index)
    # Extract the centroid and bounding box of the largest component
    centroid = centroids[valid_index]
    bbox = stats[valid_index, :-1]

    return centroid, bbox, pixels

def torch_detect(ir_image,template,template_threshold=0.3,pixel_threshold=1e4):
    ###template matching for torch, return the upper left corner of the matched region
    #threshold and normalize ir image
    ir_torch_tracking=ir_image.copy()
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255

    # run edge detection
    edges = cv2.Canny(ir_torch_tracking_normalized.astype(np.uint8), threshold1=20, threshold2=50)
    # bolden all edges
    edges=cv2.dilate(edges,None,iterations=1)

    # cv2.imshow('ir_torch_tracking',ir_torch_tracking_normalized.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    

    ###template matching with normalized image
    res = cv2.matchTemplate(edges,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    if max_val<template_threshold:
        return None
    
    return max_loc

def torch_detect_yolo(ir_image,yolo_model,pixel_threshold=1e4):
    ir_torch_tracking=copy.deepcopy(ir_image)
    ir_torch_tracking[ir_torch_tracking>pixel_threshold]=pixel_threshold
    ir_torch_tracking_normalized = ((ir_torch_tracking - np.min(ir_torch_tracking)) / (np.max(ir_torch_tracking) - np.min(ir_torch_tracking))) * 255
    ir_torch_tracking_normalized = ir_torch_tracking_normalized.astype(np.uint8)
    ir_torch_tracking = cv2.cvtColor(ir_torch_tracking_normalized, cv2.COLOR_GRAY2BGR)

    #run yolo
    result= yolo_model.predict(ir_torch_tracking,verbose=False)[0]
    if result.boxes.cls.cpu().numpy()==0:
        bbox = result.boxes.cpu().xyxy[0].numpy()\
        #change bbox to opencv format
        bbox[2]=bbox[2]-bbox[0]
        bbox[3]=bbox[3]-bbox[1]
        centroid = np.array([(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2])
        return centroid, bbox.astype(int)
    else:
        return None, None

def get_pixel_value(ir_image,coord,window_size):
    ###get top 1/4 average pixel value within the window
    window = ir_image[coord[1]-window_size//2:coord[1]+window_size//2+1,coord[0]-window_size//2:coord[0]+window_size//2+1]
    pixel_avg = np.mean(window)
    mask = (window > 3*pixel_avg/4) # filter out background 
    pixel_avg = np.mean(window[mask])
    return pixel_avg