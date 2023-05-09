import numpy as np
import traceback, sys
from scipy.optimize import minimize
sys.path.append('../toolbox')
from utils import *

def circle_from_3point(p_start,p_end,p_mid):
    v1=np.cross(p_mid-p_start,p_end-p_mid)
    v2=np.cross(p_mid-p_start,v1)
    v3=np.cross(p_end-p_mid,v1)
    p4=(p_start+p_mid)/2
    p5=(p_end+p_mid)/2
    A=np.vstack((v2,v3)).T
    b=p5-p4
    res=np.linalg.lstsq(A,b)[0]
    ###verify center aligned
    # print(p4+res[0]*v2)
    # print(p5-res[1]*v3)
    center=p4+res[0]*v2

    radius=np.linalg.norm(center-p_start)

    return center, radius
    
def arc_from_3point(p_start,p_end,p_mid,N=100):
    v1=np.cross(p_mid-p_start,p_end-p_mid)
    v1=v1/np.linalg.norm(v1)
    center, radius=circle_from_3point(p_start,p_end,p_mid)
    u=p_start-center
    v=p_end-center
    theta = angle_between(u, v, v1)
    l = np.linspace(0, theta, N)

    arc=generate_circle_by_vectors(l, center, radius, v1, u)
    return arc


def generate_circle_by_vectors(t, C, r, n, u):
    n = n/np.linalg.norm(n)
    u = u/np.linalg.norm(u)
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle

def fit_circle_2d(x, y, p=[], p2=[]):
    if len(p)==0:
        A = np.array([x, y, np.ones(len(x))]).T
        b = x**2 + y**2
        
        # Solve by method of least squares
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        
        # Get circle parameters from solution c
        xc = c[0]/2
        yc = c[1]/2
        r = np.sqrt(c[2] + xc**2 + yc**2)
        return xc, yc, r
    elif len(p2)==0:

        ###rewrite lstsq to fit point p on circle
        A = np.array([x-p[0], y-p[1]]).T
        b = x**2 + y**2 - p[0]**2 - p[1]**2
        
        # Solve by method of least squares
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        
        # Get circle parameters from solution c
        xc = c[0]/2
        yc = c[1]/2
        r=np.linalg.norm(p[:-1]-np.array([xc,yc]))
        return xc, yc, r
    else:
        A_x=(p[0]+p2[0])/2
        A_y=(p[1]+p2[1])/2
        vT=np.array([p[1]-p2[1],p2[0]-p[0]])
        vT=vT/np.linalg.norm(vT)
        A=np.array([vT[0]*(x-p[0])+vT[1]*(y-p[1])]).T
        b=x**2 + y**2 - p[0]**2 - p[1]**2-2*A_x*x+2*A_x*p[0]-2*A_y*y+2*A_y*p[1]
        d = np.linalg.lstsq(A,b,rcond=None)[0][0]
        xc=A_x + d*vT[0]
        yc=A_y + d*vT[1]
        r = abs(d)

        return xc, yc, r

def fit_circle_2d_w_slope(x,curve,p):
    #x:0-2 direction, 3 radius
    center=p-x[-1]*x[:3]
    return np.linalg.norm(x[-1]**2-np.linalg.norm(curve-center,axis=1))         ###min{ || (x-x_c)^2+(y-y_c)^2+(z-z_c)^2 - r^2 ||  }

def fit_circle_2d_w_slope2(x,curve,p,r_dir):
    #given direction already
    center=p-x*r_dir
    return np.linalg.norm(x**2-np.linalg.norm(curve-center,axis=1)**2)         ###min{ || (x-x_c)^2+(y-y_c)^2+(z-z_c)^2 - r^2 ||  }   
def vec_proj_plane(u,n):
    ###u: vector in 3D
    ###n: plane normal
    proj_of_u_on_n =u - (np.dot(u, n)/n**2)*n
    
    return proj_of_u_on_n
#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   curve_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(curve, n0, n1):

    if np.all(n0==n1):
        return curve
    
    # If curve is only 1d array (coords of single point), fix it to be matrix
    if curve.ndim == 1:
        curve = curve[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    curve_rot = np.zeros((len(curve),3))
    for i in range(len(curve)):
        curve_rot[i] = curve[i]*np.cos(theta) + np.cross(k,curve[i])*np.sin(theta) + k*np.dot(k,curve[i])*(1-np.cos(theta))

    return curve_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))

    
#-------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
#-------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = np.abs(limits[:,0] - limits[:,1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * np.max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

    ###fit curve with slope and 1 p constraint
def circle_fit_w_slope1(curve,p,slope):
    ###fit on a plane first
    curve_mean = curve.mean(axis=0)
    curve_centered = curve - curve_mean
    p_centered = p - curve_mean

    ###constraint fitting
    ###rewrite lstsq to fit point p on plane
    A = np.array([curve_centered[:,0]-p_centered[0]*curve_centered[:,2]/p_centered[2], curve_centered[:,1]-p_centered[1]*curve_centered[:,2]/p_centered[2]]).T
    b = np.ones(len(curve))-curve_centered[:,2]/p_centered[2]
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    normal=np.array([c[0],c[1],(1-c[0]*p_centered[0]-c[1]*p_centered[1])/p_centered[2]])

    ###make sure constraint point is on plane
    # print(np.dot(normal,p_centered))
    ###normalize plane normal
    normal=normal/np.linalg.norm(normal)

    ###find the line direction where the center of the circle reside
    r_dir=np.cross(slope,normal)
    r_dir=r_dir/np.linalg.norm(r_dir)

    circle_plane_normal=np.cross(r_dir,slope)
    circle_plane_normal=circle_plane_normal/np.linalg.norm(circle_plane_normal)

    res = minimize(fit_circle_2d_w_slope2, [5000], method='SLSQP',tol=1e-10, args=(curve,p,r_dir,))
    # print('radius: ',res.x)
    r=abs(res.x)
    C=p-res.x*r_dir
    end_vec=vec_proj_plane(curve[-1]-C,circle_plane_normal)

    ###get 3D circular arc
    u = p - C
    if np.linalg.norm(p-curve[0])<np.linalg.norm(p-curve[-1]):
        v=curve[-1] - C
    else:
        v=curve[0] - C
    theta = angle_between(u, v, circle_plane_normal)

    l = np.linspace(0, theta, len(curve)+1)
    curve_fitarc = generate_circle_by_vectors(l, C, r, circle_plane_normal, u)[1:]
    l = np.linspace(0, 2*np.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, circle_plane_normal, u)

    return curve_fitarc, curve_fitcircle

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def circle_fit_w_2slope(curve,p,p2,slope1,slope2):
    ###fit a circle with 2 points constraint and 2 slope constraints
    #output include p1 & p2, size=len(curve)+2
    ###fit a plane with 2 point constraint first

    ###constraint fitting
    ###rewrite lstsq to fit point p on plane
    A = np.array([curve[:,0]-p[0]*curve[:,2]/p[2]-(p2[0]-p[0]*p2[2]/p[2])*(curve[:,1]-p[1]*curve[:,2]/p[2])/(p2[1]-p[1]*p2[2]/p[2])]).T
    b = np.ones(len(curve))-curve[:,2]/p[2]-(1-p2[2]/p[2])*(curve[:,1]-p[1]*curve[:,2]/p[2])/(p2[1]-p[1]*p2[2]/p[2])
    if np.linalg.norm(b)==0:        ###corner case, when all z have same value
        normal=np.array([0,0,1])
    else:

        c = np.linalg.lstsq(A,b,rcond=None)[0]
        A_out=c[0]
        B_out=(1-p2[2]/p[2]-A_out*(p2[0]-p[0]*p2[2]/p[2]))/(p2[1]-p[1]*p2[2]/p[2])
        normal=np.array([A_out,B_out,(1-A_out*p[0]-B_out*p[1])/p[2]])

    ###make sure constraint point is on plane
    # print(np.dot(normal,p))
    # print(np.dot(normal,p2))
    ###normalize plane normal
    normal=normal/np.linalg.norm(normal)

    curve_xy = rodrigues_rot(curve, normal, [0,0,1])
    p_temp1 = rodrigues_rot(p, normal, [0,0,1]).flatten()[:-1]
    p_temp2 = rodrigues_rot(p2, normal, [0,0,1]).flatten()[:-1]
    slope_temp1 = rodrigues_rot(slope1, normal, [0,0,1]).flatten()[:-1]
    slope_temp2 = rodrigues_rot(slope2, normal, [0,0,1]).flatten()[:-1]

    

    p_mid=(p_temp1+p_temp2)/2
    slope_mid=-(p_temp2[0]-p_temp1[0])/(p_temp2[1]-p_temp1[1])

    xc1, yc1 = get_intersect(p_temp1,p_temp1+np.array([-slope_temp1[1],slope_temp1[0]]),p_mid,p_mid+np.array([1,slope_mid]))
    xc2, yc2 = get_intersect(p_temp2,p_temp2-np.array([-slope_temp2[1],slope_temp2[0]]),p_mid,p_mid+np.array([1,slope_mid]))

    xc=(xc1+xc2)/2
    yc=(yc1+yc2)/2


    r=np.linalg.norm(p_temp1-np.array([xc,yc]))

    ###convert to 3D coordinates
    C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + np.array([0,0,np.average(curve[:,-1])])
    C = C.flatten()


    ###get 3D circular arc
    ###always start from constraint p
    u=p-C
    v=p2-C

   
    

    theta = angle_between(u, v, normal)

    l = np.linspace(0, theta, len(curve)+2)
    curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)
    l = np.linspace(0, 2*np.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, normal, u)

    return curve_fitarc, curve_fitcircle




def circle_fit(curve,p=[],p2=[]):
    ###curve: 3D point train_data
    ###p:   constraint point of the arc
    ########################################
    ###return curve_fit: 3D point train_data

    if len(p)==0:   #no constraint
        ###fit on a plane first
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        U,s,V = np.linalg.svd(curve_centered)
        # Normal vector of fitting plane is given by 3rd column in V
        # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
        normal = V[2,:]

        curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])
        xc, yc, r = fit_circle_2d(curve_xy[:,0], curve_xy[:,1])

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        u = curve[0] - C
        v = curve[-1] - C

        theta = angle_between(u, v, normal)
        l = np.linspace(0, theta, len(curve))
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)
 

    elif len(p2)==0:    #single point constraint
        ###fit on a plane first
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        p_centered = p - curve_mean

        ###constraint fitting
        ###rewrite lstsq to fit point p on plane
        A = np.array([curve_centered[:,0]-p_centered[0]*curve_centered[:,2]/p_centered[2], curve_centered[:,1]-p_centered[1]*curve_centered[:,2]/p_centered[2]]).T
        b = np.ones(len(curve))-curve_centered[:,2]/p_centered[2]
        c = np.linalg.lstsq(A,b,rcond=None)[0]
        normal=np.array([c[0],c[1],(1-c[0]*p_centered[0]-c[1]*p_centered[1])/p_centered[2]])

        ###make sure constraint point is on plane
        # print(np.dot(normal,p_centered))
        ###normalize plane normal
        normal=normal/np.linalg.norm(normal)

        curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])
        p_temp = rodrigues_rot(p_centered, normal, [0,0,1])
        p_temp = p_temp.flatten()


        xc, yc, r = fit_circle_2d(curve_xy[:,0], curve_xy[:,1],p_temp)

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        ###always start from constraint p
        u=p-C
        if np.linalg.norm(p-curve[0])<np.linalg.norm(p-curve[-1]):
            v=curve[-1] - C
        else:
            v=curve[0] - C

        theta = angle_between(u, v, normal)
        l = np.linspace(0, theta, len(curve)+1)
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)[1:]

    else:
        ###fit a plane with 2 point constraint
        curve_mean = curve.mean(axis=0)
        curve_centered = curve - curve_mean
        p_centered = p - curve_mean
        p2_centered= p2 - curve_mean

        ###constraint fitting
        ###rewrite lstsq to fit point p on plane
        A = np.array([curve_centered[:,0]-p_centered[0]*curve_centered[:,2]/p_centered[2]-(p2_centered[0]-p_centered[0]*p2_centered[2]/p_centered[2])*(curve_centered[:,1]-p_centered[1]*curve_centered[:,2]/p_centered[2])/(p2_centered[1]-p_centered[1]*p2_centered[2]/p_centered[2])]).T
        b = np.ones(len(curve))-curve_centered[:,2]/p_centered[2]-(1-p2_centered[2]/p_centered[2])*(curve_centered[:,1]-p_centered[1]*curve_centered[:,2]/p_centered[2])/(p2_centered[1]-p_centered[1]*p2_centered[2]/p_centered[2])

        c = np.linalg.lstsq(A,b,rcond=None)[0]
        A_out=c[0]
        B_out=(1-p2_centered[2]/p_centered[2]-A_out*(p2_centered[0]-p_centered[0]*p2_centered[2]/p_centered[2]))/(p2_centered[1]-p_centered[1]*p2_centered[2]/p_centered[2])
        normal=np.array([A_out,B_out,(1-A_out*p_centered[0]-B_out*p_centered[1])/p_centered[2]])

        ###make sure constraint point is on plane
        # print(np.dot(normal,p_centered))
        # print(np.dot(normal,p2_centered))
        ###normalize plane normal
        normal=normal/np.linalg.norm(normal)

        curve_xy = rodrigues_rot(curve_centered, normal, [0,0,1])
        p_temp1 = rodrigues_rot(p_centered, normal, [0,0,1]).flatten()
        p_temp2 = rodrigues_rot(p2_centered, normal, [0,0,1]).flatten()


        xc, yc, r = fit_circle_2d(curve_xy[:,0], curve_xy[:,1],p_temp1,p_temp2)

        ###convert to 3D coordinates
        C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + curve_mean
        C = C.flatten()
        ###get 3D circular arc
        ###always start from constraint p
        u=p-C
        v=p2-C

        theta = angle_between(u, v, normal)
        l = np.linspace(0, theta, len(curve)+1)
        curve_fitarc = generate_circle_by_vectors(l, C, r, normal, u)[1:]

   
    

    
    l = np.linspace(0, 2*np.pi, 1000)
    curve_fitcircle = generate_circle_by_vectors(l, C, r, normal, u)

    return curve_fitarc, curve_fitcircle

def seg_3dfit(seg2fit,p=[]):

    curve_fitarc,curve_fit_circle=circle_fit(seg2fit,p)
    error=[]
    ###check error
    for i in range(len(curve_fitarc)):
        error_temp=np.linalg.norm(seg2fit-curve_fitarc[i],axis=1)
        idx=np.argmin(error_temp)
        error.append(error_temp[idx])
    return curve_fitarc,np.max(error)


def stepwise_3dfitting(curve,breakpoints):
    if len(breakpoints)==1:
        print("num of breakpoints must be greater than 2")
        return
    fit=[]
    for i in range(len(breakpoints)-1):
        seg2fit=curve[breakpoints[i]:breakpoints[i+1]]

        try:
            if i==0:

                curve_fitarc,curve_fit_circle=circle_fit(seg2fit)
                fit.append(curve_fitarc)

            else:
                curve_fitarc,curve_fit_circle=circle_fit(seg2fit,p=fit[-1][-1])
                fit.append(curve_fitarc)
        except:
            traceback.print_exc()
            print(breakpoints)

            

    return np.array(fit).reshape(-1,3)



