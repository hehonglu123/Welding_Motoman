from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
import sys, copy, traceback
from scipy.spatial import ConvexHull

sys.path.append('../toolbox')
from utils import *
from lambda_calc import *
from error_check import *
from toolbox_circular_fit import *

def check_boundary(p,stl_pc):
    ###find closest 50 points
    num_points=50
    indices=np.argsort(np.linalg.norm(stl_pc-p,axis=1))[:num_points]
    distance=np.linalg.norm(stl_pc[indices]-p,axis=1)

    threshold=1
    if np.min(distance)>10*threshold:
        return False
    
    if np.min(distance)>threshold:
        ###find a plane
        normal, centroid=fit_plane(stl_pc[indices])

        ###find projected 2d points
        projection = np.append(stl_pc[indices],[p],axis=0) - centroid
        projection = projection - np.outer(np.dot(projection, normal), normal)

        projection_xy = rodrigues_rot(projection, normal, [0,0,1])[:,:-1]
        ###convexhull checking
        hull = ConvexHull(projection_xy[:-1])
        hull_path = Path( projection_xy[:-1][hull.vertices] )

        # Check if the point is inside the convex hull
        if hull_path.contains_point(projection_xy[-1]):
            return True
        else:
            return False
    
    return True

def find_point_on_boundary(p1,p2,stl_pc):
    ###find a point between p1 (inside) and p2 (outside) on the stl boundary
    num_points=50
    indices1=np.argsort(np.linalg.norm(stl_pc-p1,axis=1))[:num_points]
    indices2=np.argsort(np.linalg.norm(stl_pc-p2,axis=1))[:num_points]
    indices=np.unique(np.concatenate((indices1,indices2),0))

    ###find a plane
    normal, centroid=fit_plane(stl_pc[indices])

    ###find projected 2d points
    projection = np.append(stl_pc[indices],[p1,p2],axis=0) - centroid
    projection = projection - np.outer(np.dot(projection, normal), normal)

    projection_xy = rodrigues_rot(projection, normal, [0,0,1])[:,:-1]
    ###convexhull checking
    line = Line(projection_xy[-1], projection_xy[-2])
    hull = ConvexHull(projection_xy[:-2])
    for simplex in hull.simplices:
        p1_hull = Point(projection_xy[simplex[0], 0], projection_xy[simplex[0], 1])
        p2_hull = Point(projection_xy[simplex[1], 0], projection_xy[simplex[1], 1])
        hull_edge = Line(p1_hull, p2_hull)
        intersection_point = line.intersection(hull_edge)
        if intersection_point:
            intersection_points.append(intersection_point)
    print(len(intersection_points))
    distance=np.linalg.norm(intersection_points[0]-projection_xy[-1])
    return p1+distance*(p2-p1)/np.linalg.norm(p2-p1)


def extract_bottom_edge(stl_file):
    # Load the STL mesh
    model_mesh = mesh.Mesh.from_file(stl_file)

    # Find the minimum z-coordinate
    max_z = np.max(model_mesh.vectors[:,:,2])

    # Set a threshold for identifying the bottom edge
    threshold = 1e-6

    # Extract the bottom triangles
    bottom_triangles = model_mesh.vectors[np.abs(model_mesh.vectors[:,:,2] - max_z) < threshold]

    # Extract the bottom edge vertices
    bottom_edge_vertices = np.unique(bottom_triangles.reshape(-1, 3), axis=0)

    return bottom_edge_vertices

def calculate_surface_normal(p1,p2,p3):
    vector1 = p2 - p1
    vector2 = p3 - p1
    surface_normal = np.cross(vector1, vector2)
    surface_normal /= np.linalg.norm(surface_normal)  # Normalize the normal vector
    return surface_normal


def project_point_onto_plane(point, normal, centroid):
    # Compute the vector from the centroid to the point
    v = point - centroid

    # Compute the dot product of v and the normal vector
    dot_product = np.dot(v, normal)

    # Compute the projection of the point onto the plane
    projected_point = point - dot_product * normal

    return projected_point

def project_point_on_stl(point,stl_pc):     ###project a point on stl surface
    indices=np.argsort(np.linalg.norm(stl_pc-point,axis=1))[:50]
    normal,centroid=fit_plane(stl_pc[indices])
    return project_point_onto_plane(point,normal,centroid)


def slicing_uniform(stl_pc,z,threshold = 1e-6):
    bottom_edge_vertices = stl_pc[np.where(np.abs(stl_pc[:,2] - z) <threshold)[0]]

    return bottom_edge_vertices

def smooth_curve(curve):
    lam=calc_lam_cs(curve)
    polyfit=np.polyfit(lam,curve,deg=20)

    return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam))).T

def get_curve_normal(curve,stl_pc,direction):
    ###provide the curve and complete stl point cloud, a rough normal direction
    curve_normal=[] 
    for i in range(len(curve)):
        tangent=curve[min(len(curve)-1,i+1)]-curve[max(0,i-1)]
        indices=np.argsort(np.linalg.norm(stl_pc-curve[i],axis=1))[:10]
        surf_norm, _=fit_plane(stl_pc[indices])
        true_norm=np.cross(surf_norm,tangent)
        if true_norm@direction<0:
            true_norm=-true_norm
        curve_normal.append(true_norm/np.linalg.norm(true_norm))
    
    return np.array(curve_normal)

def get_curve_normal_from_curves(curve,curve_prev):
    curve_normal=[]
    for i in range(len(curve)):
        indices=np.argsort(np.linalg.norm(curve_prev-curve[i],axis=1))[:2]
        u = curve_prev[indices[0]] - curve[i]
        v = curve_prev[indices[1]] - curve_prev[indices[0]]
        proj_v_u = np.dot(u, v) / np.dot(v, v) * v
        w = u - proj_v_u
        curve_normal.append(- w / np.linalg.norm(w))

    return np.array(curve_normal)


def slice_next_layer(curve,stl_pc,curve_normal,slice_height):

    slice_next=[]
    for i in range(len(curve)):
        p_plus=project_point_on_stl(curve[i]+slice_height*curve_normal[i],stl_pc)
        slice_next.append(p_plus)

    return np.array(slice_next)

def find_point_on_boundary(p,vec,stl_pc):
    return

def fit_to_length(curve,stl_pc,resolution=0.5):

    ###shrink start first
    start_idx=0
    start_shrinked=False
    while not check_boundary(curve[start_idx],stl_pc):
        start_shrinked=True
        start_idx+=1
        if start_idx>len(curve)-1:
            return []
        
    if start_shrinked:
        boundary_distance=np.linalg.norm(curve[start_idx-1]-curve[start_idx])
    else:
        boundary_distance=5
    curve=curve[start_idx:]

    if len(curve)<2:
        return curve
    
    ###extend start second
    vec=(curve[0]-curve[1])/np.linalg.norm(curve[0]-curve[1])
    distance_added=0
    while check_boundary(curve[0],stl_pc) and distance_added < boundary_distance:
        next_p=project_point_on_stl(curve[0]+resolution*vec,stl_pc)
        curve=np.insert(curve,0,copy.deepcopy(next_p),axis=0)
        distance_added+=resolution
    curve=curve[1:]
    

    ###shrink end first
    end_idx=len(curve)-1
    end_shrinked=False
    while not check_boundary(curve[end_idx],stl_pc):
        end_shrinked=True
        end_idx-=1
        if end_idx<1:
            return []
        
    if end_shrinked:
        boundary_distance=np.linalg.norm(curve[end_idx+1]-curve[end_idx])
    else:
        boundary_distance=5
    curve=curve[:end_idx+1]

    if len(curve)<2:
        return curve

    ###extend end second
    vec=(curve[-1]-curve[-2])/np.linalg.norm(curve[-1]-curve[-2])
    distance_added=0
    while check_boundary(curve[-1],stl_pc) and distance_added < boundary_distance:
        next_p=project_point_on_stl(curve[-1]+resolution*vec,stl_pc)
        curve=np.append(curve,[copy.deepcopy(next_p)],axis=0)
        distance_added+=resolution
    curve=curve[:-1]

    return curve

def split_slices(curve,stl_pc):
    indices=[]
    continuous_count=0
    continuous_threshold=1      ###more than x continuous points not on stl means a gap 
    continuous_threshold2=2     ###splited curve must contain more than x points
    for i in range(len(curve)):
        if not check_boundary(curve[i],stl_pc):
            if i-1 in indices:
                continuous_count+=1
            else:
                continuous_count=0
            indices.append(i)
        else:
            ###filter small gaps
            if continuous_count<continuous_threshold and i-1 in indices:
                indices=indices[:-(continuous_count+1)]

    ###point distance thresholding
    distances = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    threshold=10*np.average(distances)
    # Find outlier indices
    indices=np.hstack((np.array(indices),np.where(distances > threshold)[0])).astype(int)
    ###split curve
    sub_curves=np.split(curve, indices)



    sub_curves=[sub_curve for sub_curve in sub_curves if len(sub_curve) > continuous_threshold2]
    # for sub_curve in sub_curves:
    #     print(len(sub_curve))
    return sub_curves

def slice_stl(bottom_curve,stl_pc,direction,slice_height):
    direction=np.array([0,0,-1])
    slice_all=[[bottom_curve]]
    layer_num=0
    while True:
        print(layer_num, 'th layer')
        if len(slice_all[-1])==0:
            slice_all=slice_all[:-1]
            break
        slice_ith_layer=[]
        for x in range(len(slice_all[-1])):
            ###push curve 1 layer up
            try:
                curve_normal=get_curve_normal_from_curves(slice_all[-1][x],np.concatenate(slice_all[-2],axis=0))
            except:
                print('USING SURF NORM @ %ith layer'%layer_num)
                curve_normal=get_curve_normal(slice_all[-1][x],stl_pc,direction)

            curve_next=slice_next_layer(slice_all[-1][x],stl_pc,curve_normal,slice_height)

            if x==0 or x==len(slice_all[-1])-1: ###only extend or shrink if first or last segment for now
                curve_next=fit_to_length(curve_next,stl_pc)

            if len(curve_next)==0:   
                if len(slice_all[-1])<=1:   ###end condition
                    return slice_all
                continue

            ###split the curve based on projection error
            sub_curves_next=split_slices(curve_next,stl_pc)

            for j in range(len(sub_curves_next)):
                sub_curves_next[j]=smooth_curve(sub_curves_next[j])
            
            slice_ith_layer.extend(sub_curves_next)

        layer_num+=1
        slice_all.append(slice_ith_layer)

    
    return slice_all

def smooth_normal(curve_normal,n=15):
    curve_normal_new=copy.deepcopy(curve_normal)
    for i in range(len(curve_normal)):
        curve_normal_new[i]=np.average(curve_normal[max(0,i-n):min(len(curve_normal),i+n)],axis=0)
        curve_normal_new[i]=curve_normal_new[i]/np.linalg.norm(curve_normal_new[i])
    return curve_normal_new

def post_process(slice_all,point_distance=0.5):       ###postprocess the sliced layers into equally spaced points and attach curve normal
    slice_all_new=[]
    curve_normal_all=[]

    lam=calc_lam_cs(slice_all[0][0])
    polyfit=np.polyfit(lam,slice_all[0][0],deg=47)
    lam=np.linspace(0,lam[-1],num=int(lam[-1]/point_distance))
    slice_all_new.append([np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam))).T])
    curve_normal_all.append([np.array([[0,0,1]]*len(slice_all_new[0][0]))])

    slice_prev=slice_all_new[0][0]
    for i in range(1,len(slice_all)):
        slice_ith_layer=[]
        normal_ith_layer=[]
        for x in range(len(slice_all[i])):
            lam=calc_lam_cs(slice_all[i][x])
            polyfit=np.polyfit(lam,slice_all[i][x],deg=47)
            lam=np.linspace(0,lam[-1],num=int(np.ceil(lam[-1]/point_distance)))
            curve=np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam))).T
            slice_ith_layer.append(curve)
            normal=-get_curve_normal_from_curves(curve,slice_prev)

            normal_ith_layer.append(smooth_normal(normal))

        slice_prev=np.concatenate(slice_ith_layer,axis=0)
        slice_all_new.append(slice_ith_layer)
        curve_normal_all.append(normal_ith_layer)
        if len(slice_prev)==1:
            break       ###if previous layer only contains 1 point given point_distance, then quit

    return slice_all_new, curve_normal_all


def main():
    # Load the STL file
    filename = '../data/blade0.1/surface.stl'
    your_mesh = mesh.Mesh.from_file(filename)
    # Get the number of facets in the STL file
    num_facets = len(your_mesh)

    slice_height=0.1

    # Extract all vertices
    vertices = np.zeros((num_facets, 3, 3))
    for i, facet in enumerate(your_mesh.vectors):
        vertices[i] = facet
    # Flatten the vertices array and remove duplicates
    stl_pc = np.unique(vertices.reshape(-1, 3), axis=0)
    stl_pc *= 25.4      ##convert to mm

    bottom_edge = slicing_uniform(stl_pc,z = np.max(stl_pc[:,2]))

    slice_all=slice_stl(bottom_edge,stl_pc,np.array([0,0,-1]),slice_height=slice_height)
    slice_all,curve_normal_all=post_process(slice_all,point_distance=1)
   

    # Plot the original points and the fitted curved plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis_step=1

    for i in range(len(slice_all)):
        for x in range(len(slice_all[i])):
            if len(slice_all[i][x])==0:
                break

            ax.plot3D(slice_all[i][x][::vis_step,0],slice_all[i][x][::vis_step,1],slice_all[i][x][::vis_step,2],'r.-')
            # np.savetxt('slicing_result/slice%i_%i.csv'%(i,x),slice_all[i][x],delimiter=',')
            np.savetxt('slicing_result/slice%i_%i.csv'%(i,x),np.hstack((slice_all[i][x],curve_normal_all[i][x])),delimiter=',')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('STL %fmm Slicing'%slice_height)
    plt.show()

if __name__ == "__main__":
    main()