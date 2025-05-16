import time, os, copy, sys, yaml, pathlib
import traceback
from copy import deepcopy
import numpy as np
import datetime
from motoman_def import *
from lambda_calc import *
import open3d as o3d
from robotics_utils import *
from general_robotics_toolbox import *
from matplotlib import pyplot as plt

def main():
    ############## Robot definition ##############
    config_dir='../../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    # get fujicam standoff distance
    # Move the fujicam frame along the z-axis with the distance of the standoff distance
    # will locate the frame onto 
    # the plane perpendicular to the weldgun axis and passing through the weldgun TCP
    T_weldgun = robot_weld.fwd(np.zeros(6))
    T_scanner = robot_scan.fwd(np.zeros(6))
    fujicam_standoff_d = np.dot((T_weldgun.p-T_scanner.p),T_weldgun.R[:3,2])/np.dot(T_scanner.R[:3,2],T_weldgun.R[:3,2])
    robot_scan_motion=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=fujicam_standoff_d,tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    robot_thermal=robot_obj('MA1440_A0',def_path=config_dir+'MA1440_A0_robot_default_config.yml',tool_file_path=config_dir+'flir.csv',\
	                        pulse2deg_file_path=config_dir+'MA1440_A0_pulse2deg_real.csv',base_transformation_file=config_dir+'MA1440_pose.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_default_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
        base_transformation_file=config_dir+'D500B_pose.csv',pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'D500B_marker_config/D500B_marker_config.yaml',tool_marker_config_file=config_dir+'positioner_tcp_marker_config/positioner_tcp_marker_config.yaml')
    r_weld_z = robot_weld.fwd(np.zeros(6))
    r_scan_z = robot_scan_motion.fwd(np.zeros(6))
    T_weld_scan = r_weld_z.inv()*r_scan_z
    dist_weld_scan = np.linalg.norm(T_weld_scan.p)
    ##############################################

    base_positioner = deepcopy(positioner.base_H)
    print("base positioner", base_positioner)
    # move in the x direction for -200 mm
    base_positioner[0,3] -= 200
    # move in the y direction for 780 mm
    base_positioner[1,3] += 780
    base_center_pose = deepcopy(base_positioner)
    positioner.base_H = deepcopy(base_center_pose) # This is the center of the workspace
    print("base positioner", base_positioner)

    # parameters
    workspace_width = 3000
    workspace_length = 2000

    ##############################################

    table_joints = np.radians([-15,0])

    print("robot weld zero config", robot_weld.fwd(np.zeros(6)))

    ### create a path points xyz using a circle with radius, z=0
    radius_p_sol = {}
    radius_p_world = {}
    radius_vs_height = []
    last_best_z = 1240
    for radius_circle in range(100,1100,100):
        # radius_circle = 650 # mm
        print("radius circle", radius_circle)
        num_points = 180
        circle_points = np.zeros((num_points,3))
        for i in range(num_points):
            theta = 2*np.pi*i/num_points
            circle_points[i,0] = radius_circle*np.cos(theta)
            circle_points[i,1] = radius_circle*np.sin(theta)
            circle_points[i,2] = 0

        ### check IK results with z=0 until no solution
        z_height = last_best_z
        dz_search = 10
        while True:
            print("z height", z_height)
            q_sol_qll = []
            # p_world_all = []
            for y_rail_diff in np.append(np.arange(-(workspace_width/2-radius_circle), workspace_width/2-radius_circle, 20), workspace_width/2-radius_circle):
                # print("y rail diff", y_rail_diff)
                positioner.base_H = deepcopy(base_center_pose)
                positioner.base_H[1,3] += y_rail_diff
                T_table = positioner.fwd(table_joints,world=True)
                for p in circle_points:
                    p[2] = z_height
                    p_world = T_table.R@p + T_table.p

                    # p_world_all.append(p_world)

                    Rz = np.array([0,0,-1])
                    Rx = np.append(-p_world[:2],0)
                    Rx = Rx/np.linalg.norm(Rx)
                    Ry = np.cross(Rz, Rx)
                    R = np.array([Rx, Ry, Rz]).T
                    try:
                        q_all = robot_weld.inv(p_world,R,last_joints=np.zeros(6))
                        q_sol_qll.append(q_all[0])
                        break # if we found a solution, break the loop
                        # print("q sol", np.round(np.degrees(q_all[0])))
                        # print("p sol", np.round(p_world))
                        # print("p sol table",np.round(p))
                        # input("Press enter to continue")
                    except ValueError:
                        # traceback.print_exc()
                        pass
                if len(q_sol_qll) > 0: # if we found a solution, break the loop
                    break
            # p_world_all = np.array(p_world_all)
            # plt.scatter(p_world_all[:,0], p_world_all[:,1])
            # plt.axis('equal')
            # plt.title("p world all")
            # plt.show()
            if len(q_sol_qll) == 0:
                z_height -= dz_search
                break
            z_height += dz_search
        print("Max z height", z_height)
        ### check IK results at z=z_height
        q_sol_qll = []
        p_sol_all = []
        p_world_all = []
        # for y_rail_diff in np.append(np.arange(-(workspace_width/2-radius_circle), workspace_width/2-radius_circle, 20), workspace_width/2-radius_circle):
        #     positioner.base_H = deepcopy(base_center_pose)
        #     positioner.base_H[1,3] += y_rail_diff
        #     T_table = positioner.fwd(table_joints,world=True)
        #     for p in circle_points:
        #         p[2] = z_height
        #         p_world = T_table.R@p + T_table.p
        #         p_world_all.append(p_world)

        #         Rz = np.array([0,0,-1])
        #         Rx = np.append(-p_world[:2],0)
        #         Rx = Rx/np.linalg.norm(Rx)
        #         Ry = np.cross(Rz, Rx)
        #         R = np.array([Rx, Ry, Rz]).T
        #         try:
        #             q_all = robot_weld.inv(p_world,R,last_joints=np.zeros(6))
        #             q_sol_qll.append(q_all[0])
        #             p_sol_all.append(p_world)
        #         except ValueError:
        #             # traceback.print_exc()
        #             pass
        # radius_p_sol[radius_circle] = np.array(p_sol_all)
        # radius_p_world[radius_circle] = np.array(p_world_all)
                
        radius_vs_height.append([radius_circle, z_height])
        last_best_z = z_height
    plt.scatter(np.array(radius_vs_height)[:,0], np.array(radius_vs_height)[:,1])
    # color the space below the line
    plt.fill_between(np.array(radius_vs_height)[:,0], 0, np.array(radius_vs_height)[:,1], alpha=0.2)
    plt.xlabel("Radius (mm)")
    plt.ylabel("Height (mm)")
    plt.title("Radius vs Height")
    plt.grid()
    plt.show()

    # # draw animation of radius vs p_sol_all
    # while not KeyboardInterrupt:
    #     for radius_circle in radius_p_sol:
    #         p_sol_all = radius_p_sol[radius_circle]
    #         p_world_all = radius_p_world[radius_circle]
    #         # plot p sol all in xy plane
    #         # p_sol_all = np.array(p_sol_all)
    #         # p_world_all = np.array(p_world_all)
    #         plt.scatter(p_world_all[:,0], p_world_all[:,1])
    #         plt.scatter(p_sol_all[:,0], p_sol_all[:,1])
    #         # make axes equal
    #         plt.axis('equal')
    #         plt.title("p sol all "+str(radius_circle))
    #         plt.pause(0.3)
    #         plt.clf()

if __name__ == "__main__":
    main()