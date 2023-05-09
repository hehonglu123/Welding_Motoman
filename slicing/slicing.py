from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
import sys, copy
from scipy.spatial import ConvexHull
sys.path.append('../toolbox')
from lambda_calc import *
from error_check import *
from toolbox_circular_fit import *

def check_boundary(p,stl_pc):
    ###find closest 50 points
    num_points=50
    indices=np.argsort(np.linalg.norm(stl_pc-p,axis=1))[:num_points]
    distance=np.linalg.norm(stl_pc[indices]-p,axis=1)

    threshold=1
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

def fit_plane(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)

    # Center the points by subtracting the centroid
    centered_points = points - centroid

    # Calculate the SVD of the centered points
    u, s, vh = np.linalg.svd(centered_points)

    # The normal vector of the plane is the last column of vh
    normal = vh[-1]

    return normal, centroid

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
    lam=np.insert(np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1)),0,0)
    polyfit=np.polyfit(lam,curve,deg=47)

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

def split_slice1(curve):    
    # Calculate distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    threshold=5*np.average(distances)

    # Find outlier indices
    outlier_indices = np.where(distances > threshold)[0]
    print('num outlier indices: ',len(outlier_indices))
    if len(outlier_indices)>0:

        return [curve[:outlier_indices[0]], curve[outlier_indices[-1]+1:]]

    else:
        return [curve]

def find_point_on_boundary(p,vec,stl_pc):
    return
def fit_to_length(curve,stl_pc,threshold=1): 
    
    if check_boundary(curve[0],stl_pc):
        ###extend to fit on boundary
        next_p=project_point_on_stl(2*curve[0]-curve[1],stl_pc)        

        while check_boundary(next_p,stl_pc) and np.linalg.norm(next_p-curve[0])<threshold:
            curve=np.insert(curve,0,copy.deepcopy(next_p),axis=0)
            next_p=project_point_on_stl(2*curve[0]-curve[1],stl_pc)
        start_idx=0

    else:
        ###shrink to fit on boundary
        i=1
        while not check_boundary(curve[i],stl_pc): 
            i+=1
            if i>len(curve)-1:
                return []
        start_idx=i

    # print('end point check')
    if check_boundary(curve[-1],stl_pc):
        ###extend to fit on boundary
        next_p=project_point_on_stl(2*curve[-1]-curve[-2],stl_pc)
        while check_boundary(next_p,stl_pc) and np.linalg.norm(next_p-curve[-1])<threshold:
            curve=np.append(curve,[copy.deepcopy(next_p)],axis=0)
            next_p=project_point_on_stl(2*curve[-1]-curve[-2],stl_pc)

        end_idx=len(curve)-1
    else:
        ###shrink to fit on boundary
        i=len(curve)-1
        while not check_boundary(curve[i],stl_pc): 
            i-=1
            if i<1:
                return []
        end_idx=i
    
    if start_idx>=end_idx:
        return []
    return curve[start_idx:end_idx+1]

def split_slices(curve,stl_pc):
    indices=[]
    continuous_count=0
    continuous_threshold=1      ###more than x continuous points not on stl means a gap 
    continuous_threshold2=2     ###curve must contain more than x points
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

def slice(bottom_curve,stl_pc,direction,slice_height):
    direction=np.array([0,0,-1])
    slice_all=[[bottom_curve]]
    layer_num=0
    while True:
        print(layer_num, 'th layer')
        if len(slice_all[-1])==0:
            break
        slice_ith_layer=[]
        for x in range(len(slice_all[-1])):
            ###push curve 1 layer up
            try:
                ###Fix curve normal from unaligned section
                if len(slice_all[-2])!=len(slice_all[-1]):
                    curve_normal=get_curve_normal(slice_all[-1][x],stl_pc,direction)
                else:
                    curve_normal=get_curve_normal_from_curves(slice_all[-1][x],slice_all[-2][x])
            except:
                print('USING SURF NORM @ %ith layer'%layer_num)
                curve_normal=get_curve_normal(slice_all[-1][x],stl_pc,direction)

            curve_next=slice_next_layer(slice_all[-1][x],stl_pc,curve_normal,slice_height)

            if x==0 or x==len(slice_all[-1])-1: ###only extend or shrink if first or last segment for now, need to address later
                curve_next=fit_to_length(curve_next,stl_pc)

            if len(curve_next)==0:   
                print('here',len(slice_all[-1]))   
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

def main():
    # Load the STL file
    filename = '../data/blade0.1/surface.stl'
    your_mesh = mesh.Mesh.from_file(filename)
    # Get the number of facets in the STL file
    num_facets = len(your_mesh)

    # Extract all vertices
    vertices = np.zeros((num_facets, 3, 3))
    for i, facet in enumerate(your_mesh.vectors):
        vertices[i] = facet
    # Flatten the vertices array and remove duplicates
    stl_pc = np.unique(vertices.reshape(-1, 3), axis=0)
    stl_pc *= 25.4      ##convert to mm

    bottom_edge = slicing_uniform(stl_pc,z = np.max(stl_pc[:,2]))
    curve_normal=get_curve_normal(bottom_edge,stl_pc,np.array([0,0,-1]))

    slice_all=slice(bottom_edge,stl_pc,np.array([0,0,-1]),slice_height=0.1)

    # Plot the original points and the fitted curved plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis_step=5

    # ax.plot3D(bottom_edge[::vis_step,0],bottom_edge[::vis_step,1],bottom_edge[::vis_step,2],'r.-')
    # ax.quiver(bottom_edge[::vis_step,0],bottom_edge[::vis_step,1],bottom_edge[::vis_step,2],curve_normal[::vis_step,0],curve_normal[::vis_step,1],curve_normal[::vis_step,2],length=0.1, normalize=True)
    # ax.scatter(stl_pc[:,0], stl_pc[:,1], stl_pc[:,2], c='b', marker='o', label='Original points')

    for i in range(len(slice_all)):
        for x in range(len(slice_all[i])):
            ax.plot3D(slice_all[i][x][::vis_step,0],slice_all[i][x][::vis_step,1],slice_all[i][x][::vis_step,2],'r.-')
            np.savetxt('slicing_result/slice_%i_%i.csv'%(i,x),slice_all[i][x],delimiter=',')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('STL first X Layer Slicing')
    plt.show()

if __name__ == "__main__":
    main()