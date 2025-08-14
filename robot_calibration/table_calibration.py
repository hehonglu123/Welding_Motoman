from copy import deepcopy
import sys
from robotics_utils import *
from motoman_def  import *
from matplotlib import pyplot as plt
from dx200_motion_program_exec_client import *
from general_robotics_toolbox import *
import numpy as np
from numpy import linalg, linspace,cos, sin, array, pi, dot, cross, newaxis, ones, diag, sqrt
import time
import yaml

def closest_points_on_lines(C1, n1, C2, n2, eps=1e-12):
    """
    Compute the closest points between two infinite 3D lines:
        L1(s) = C1 + s * n1
        L2(t) = C2 + t * n2
    Returns:
        P1, P2, s, t, dist
    where P1 on L1 and P2 on L2 minimize ||P1 - P2||.
    Handles skew, intersecting, parallel, and coincident lines.
    """
    C1 = np.asarray(C1, dtype=float).reshape(3)
    n1 = np.asarray(n1, dtype=float).reshape(3)
    C2 = np.asarray(C2, dtype=float).reshape(3)
    n2 = np.asarray(n2, dtype=float).reshape(3)

    if np.linalg.norm(n1) < eps or np.linalg.norm(n2) < eps:
        raise ValueError("Direction vectors n1 and n2 must be non-zero.")

    r0 = C1 - C2
    a = np.dot(n1, n1)
    b = np.dot(n1, n2)
    c = np.dot(n2, n2)
    d = np.dot(n1, r0)
    e = np.dot(n2, r0)
    denom = a * c - b * b

    if abs(denom) > eps:
        # Skew or intersecting lines (generic case)
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        P1 = C1 + s * n1
        P2 = C2 + t * n2
    else:
        # Parallel (or nearly). Use a unit direction to resolve.
        u = n1 / np.linalg.norm(n1)
        v = C2 - C1
        v_perp = v - np.dot(v, u) * u  # component of C2-C1 perpendicular to the lines

        if np.linalg.norm(v_perp) < eps:
            # Coincident lines (same infinite line): infinite solutions.
            # Return a consistent pair: project C1 onto L2.
            t = np.dot(C1 - C2, n2) / np.dot(n2, n2)
            s = 0.0
            P1 = C1
            P2 = C2 + t * n2
        else:
            # Strictly parallel, distinct lines.
            # Choose P1 on L1 at C1 (s=0), pick P2 on L2 so segment P1P2 is perpendicular.
            k = np.dot(n2, u)  # equals ±||n2||
            if abs(k) < eps:
                # Shouldn't happen if truly parallel, but guard anyway
                k = np.sign(k) * np.linalg.norm(n2) if abs(k) > 0 else np.linalg.norm(n2)
            s = 0.0
            t = - np.dot(v, u) / k  # ensure (P1 - P2)·u = 0
            P1 = C1 + s * n1
            P2 = C2 + t * n2

    dist = np.linalg.norm(P1 - P2)
    return P1, P2, s, t, dist
def _plane_basis_from_normal(normal: np.ndarray):
    """
    Given a 3D normal vector, return two orthonormal vectors (u, v)
    spanning the plane perpendicular to that normal, plus the unit normal.
    """
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal must be non-zero")
    n = n / n_norm

    # Pick a vector 'a' that is not parallel to n
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    # u is perpendicular to both n and a; v completes the right-handed basis
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)  # already unit length because n and u are unit & orthogonal
    return u, v, n

def sample_circle3d(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    n_samples: int,
    where: str = "disk",         # "disk" for filled circle (uniform area), "circumference" for boundary
    inner_radius: float = 0.0,   # set >0 for an annulus (disk mode only)
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Sample points in/on a 3D circle defined by its center and normal.

    Parameters
    ----------
    center : array-like, shape (3,)
        3D coordinates of the circle center.
    normal : array-like, shape (3,)
        3D normal vector of the circle's plane (need not be unit length).
    radius : float
        Outer radius of the circle.
    n_samples : int
        Number of points to sample.
    where : {"disk", "circumference"}
        - "disk": sample uniformly over area (filled circle).
        - "circumference": sample uniformly along the boundary.
    inner_radius : float
        For "disk" mode only: inner radius to create an annulus (default 0).
    rng : np.random.Generator
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    pts : ndarray, shape (n_samples, 3)
        Sampled 3D points.
    """
    center = np.asarray(center, dtype=float).reshape(3)
    u, v, _ = _plane_basis_from_normal(np.asarray(normal, dtype=float))

    if radius <= 0:
        raise ValueError("radius must be positive")
    if where not in {"disk", "circumference"}:
        raise ValueError("where must be 'disk' or 'circumference'")

    rng = rng or np.random.default_rng()

    if where == "circumference":
        # Uniform along boundary
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        r = np.full(n_samples, radius)
    else:
        # Uniform over area (or annulus if inner_radius > 0)
        if inner_radius < 0 or inner_radius >= radius:
            raise ValueError("inner_radius must satisfy 0 <= inner_radius < radius")
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        # Correct area-uniform sampling: r^2 is uniform in [r_in^2, r_out^2]
        r2 = inner_radius**2 + (radius**2 - inner_radius**2) * rng.uniform(0.0, 1.0, size=n_samples)
        r = np.sqrt(r2)

    # 2D coordinates in the circle's plane
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Lift to 3D using the plane basis
    pts = center + np.outer(x, u) + np.outer(y, v)
    return pts
def fit_circle_2d(x, y, w=[]):
    
    A = array([x, y, ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = diag(w)
        A = dot(W,A)
        b = dot(W,b)
    
    # Solve by method of least squares
    c = linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r
def fitting_3dcircle(P):

    #-------------------------------------------------------------------------------
    # (1) Fitting plane by SVD for the mean-centered data
    # Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
    #-------------------------------------------------------------------------------
    P_mean = np.mean(P,axis=0)
    P_centered = P - P_mean
    U,s,V = linalg.svd(P_centered)

    # Normal vector of fitting plane is given by 3rd column in V
    # Note linalg.svd returns V^T, so we need to select 3rd row from V^T
    normal = V[2,:]
    d = -dot(P_mean, normal)  # d = -<p,n>

    #-------------------------------------------------------------------------------
    # (2) Project points to coords X-Y in 2D plane
    #-------------------------------------------------------------------------------
    P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

    #-------------------------------------------------------------------------------
    # (3) Fit circle in new 2D coords
    #-------------------------------------------------------------------------------
    xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])

    #--- Generate circle points in 2D
    t = linspace(0, 2*pi, 100)
    xx = xc + r*cos(t)
    yy = yc + r*sin(t)

    #-------------------------------------------------------------------------------
    # (4) Transform circle center back to 3D coords
    #-------------------------------------------------------------------------------
    C = rodrigues_rot(array([xc,yc,0]), [0,0,1], normal) + P_mean
    C = C.flatten()

    return C,normal

def get_residual_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.sqrt(np.mean(errors**2))
def get_mean_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.mean(errors)
def get_std_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.std(errors)
def get_max_error(p_array):
    p_array = np.array(p_array)
    p_mean = np.mean(p_array, axis=0)
    errors = np.linalg.norm(p_array - p_mean, axis=1)
    return np.max(errors)

config_dir='../config/'

ph_dataset_date='0801'
test_dataset_date='0801'
robot_marker_dir=config_dir+'MA2010_marker_config/'
tool_marker_dir=config_dir+'weldgun_marker_config/'
robot_1=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',\
                    tool_file_path=config_dir+'torch.csv',d=15,\
                    #  tool_file_path='',d=0,\
                    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
                    base_marker_config_file=robot_marker_dir+'MA2010_'+ph_dataset_date+'_marker_config.yaml',\
                    tool_marker_config_file=tool_marker_dir+'weldgun_'+ph_dataset_date+'_marker_config.yaml')

positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')


origin_p_tool = deepcopy(robot_1.p_tool)
origin_R_tool = deepcopy(robot_1.R_tool)
origin_P_R1 = deepcopy(robot_1.robot.P)
origin_H_R1 = deepcopy(robot_1.robot.H)

tool_calib_joints = np.radians(np.loadtxt('tool_joint_angles.csv',delimiter=','))

robot_1.p_tool = np.zeros(3)
robot_1.R_tool = np.eye(3)
robot_1.robot.p_tool = np.zeros(3)
robot_1.robot.R_tool = np.eye(3)
###
num_js = len(tool_calib_joints)
robot_Ts=[]
robot_ps = []
for i in range(num_js):
    q=tool_calib_joints[i][:6]
    robot_T=robot_1.fwd(q)
    robot_Ts.append(H_from_RT(robot_T.R,robot_T.p))
    robot_ps.append(robot_T.p)
# print("Residual tool position:", get_residual_error(robot_ps))
# print("==============")

A=[]
b=[]

# num_js=7
for i in range(num_js-1):
    A.extend(robot_Ts[i][:3,:3]-robot_Ts[i+1][:3,:3])
    b.extend(robot_Ts[i+1][:3,-1]-robot_Ts[i][:3,-1])
    # b.extend(np.zeros(3))  # assume no translation change
# print("A",A, "b",b)

p_tool=np.linalg.pinv(A)@b
# print(p_tool)
# find null space of A
# p_tool=np.linalg.lstsq(A, b, rcond=None)[0]

# use the calibrated tool position
robot_1.p_tool = deepcopy(p_tool)
robot_1.R_tool = deepcopy(origin_R_tool)
robot_1.robot.p_tool = deepcopy(p_tool)
robot_1.robot.R_tool = deepcopy(origin_R_tool)
robot_ps = []
for i in range(num_js):
    q=tool_calib_joints[i][:6]
    robot_T=robot_1.fwd(q)
    robot_ps.append(robot_T.p)
    robot_Ts.append(H_from_RT(robot_T.R,robot_T.p))
print("Residual tool position after calibration:", get_residual_error(robot_ps))

# load table joints
table_calib_joints = np.radians(np.loadtxt('table_angles.csv',delimiter=','))

all_points = []
for joint_angles in table_calib_joints:
    robot_T = robot_1.fwd(joint_angles[:6])
    all_points.append(robot_T.p)
all_points = np.array(all_points)

first_set = [0,1,2,3,4]
positioner_zero_joints = table_calib_joints[0][-2:]
print("Positioner zero joints:", positioner_zero_joints)
second_set = [0,5,6]

# plot 3D points
# all_points = np.array(all_points)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(all_points[first_set,0], all_points[first_set,1], all_points[first_set,2])
# ax.scatter(all_points[second_set,0], all_points[second_set,1], all_points[second_set,2], c='r')
# plt.show()

C1, n1 = fitting_3dcircle(all_points[first_set])
r1 = np.mean(np.linalg.norm(C1 - all_points[first_set], axis=1))
C1_sample = sample_circle3d(C1, n1, r1, 10000, where='circumference')
C1_sample = np.array(C1_sample)
C2, n2 = fitting_3dcircle(all_points[second_set])
r2 = np.mean(np.linalg.norm(C2 - all_points[second_set], axis=1))
C2_sample = sample_circle3d(C2, n2, r2, 10000, where='circumference')
C2_sample = np.array(C2_sample)

error_first = []
for p_idx in first_set:
    error_first.append(np.linalg.norm(all_points[p_idx] - C1_sample, axis=1).min())
print('Error for first set:', error_first)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(all_points[first_set,0], all_points[first_set,1], all_points[first_set,2])
# ax.scatter(C1_sample[:,0], C1_sample[:,1], C1_sample[:,2])
# ax.scatter(all_points[second_set,0], all_points[second_set,1], all_points[second_set,2])
# ax.scatter(C2_sample[:,0], C2_sample[:,1], C2_sample[:,2])
# plt.show()

# find closest points on C1 n1 to C2 n2
P1, P2, s, t, dist = closest_points_on_lines(C1, n1, C2, n2)
print("C1, C2", C1, C2)
print("n1, n2", n1, n2)
print("Closest points on C1 and C2:", P1, P2)
print("Distance between closest points:", dist)

# show positioner base position
print("Positioner base position:", positioner.base_H)

# new base pose
y_axis = -n2
y_axis = y_axis / np.linalg.norm(y_axis)
# rotate n1 around y-axis by positioner_zero_joints[0][1]
rotate_axis_2 = rot(y_axis, positioner_zero_joints[0]) @ deepcopy(n1)
z_axis = rotate_axis_2 - np.dot(rotate_axis_2, y_axis) * y_axis
z_axis = z_axis / np.linalg.norm(z_axis)
x_axis = np.cross(y_axis, z_axis)
# center_p = (P1 + P2) / 2 - 380*z_axis  # move 380mm along z-axis
center_p = P2 - 380*z_axis  # move 380mm along z-axis
P2_P1 = P1-P2
print("P2_P1:", P2_P1)

new_R = np.column_stack((x_axis, y_axis, z_axis))
new_base_H = H_from_RT(new_R, center_p)
print("New base pose:", new_base_H)

rotation_axis_2_ST = new_R.T@rotate_axis_2
P2_P1_ST = new_R.T@P2_P1
print("P2_P1_ST:", P2_P1_ST)
print(positioner.robot.P.T)
print(positioner.robot.H.T)
positioner.robot.P[:,1] = P2_P1_ST+150*rotation_axis_2_ST
positioner.robot.H[:,1] = rotation_axis_2_ST
print(positioner.robot.P.T)
print(positioner.robot.H.T)

positioner.base_H = new_base_H

test_joints = np.radians(np.loadtxt('test_angles.csv', delimiter=','))

t2 = positioner.fwd(test_joints[-2:], world=True)
t1 = robot_1.fwd(test_joints[:6])
print(t1)
t1_t2 = t2.inv() * t1

test_joints_2 = deepcopy(test_joints)
print(np.degrees(test_joints_2)[-2:])
test_joints_2[-1] = test_joints_2[-1] + np.pi # 180 degree rotation
print(np.degrees(test_joints_2)[-2:])

t2_new = positioner.fwd(test_joints_2[-2:], world=True)
t1_new = t2_new*t1_t2
print(t1_new)
robot1_joints = robot_1.inv(t1_new.p, t1.R, last_joints=test_joints_2[:6])[0]

print("Positioner new joints:", np.degrees(test_joints_2[-2:]))
print("Robot new joints:", np.degrees(robot1_joints))

input("Press Enter to execute the motion program...")
robot_client=MotionProgramExecClient()
mp = MotionProgram(ROBOT_CHOICE='ST1', pulse2deg=positioner.pulse2deg)
mp.MoveJ(np.degrees(test_joints_2[-2:]),40,0)
robot_client.execute_motion_program(mp)

mp = MotionProgram(ROBOT_CHOICE='RB1', pulse2deg=robot_1.pulse2deg)
mp.MoveJ(np.degrees(robot1_joints),5,0)
robot_stamps,curve_exe, job_line,job_step = robot_client.execute_motion_program(mp)