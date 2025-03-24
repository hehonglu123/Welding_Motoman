import time, os, copy, sys, yaml, inspect
import glob
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from motoman_def import *
from robotics_utils import *
from flir_toolbox import *
from ultralytics import YOLO
sys.path.append('../../scan/scan_process/')
sys.path.append('../../scan/scan_tools/')
from scan_utils import *
from scanProcess import *

def main():

    ############## Robot definition ##############
    config_dir='../../config/'
    robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
        base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')
    robot_scan=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',tool_file_path=config_dir+'fujicam.csv',\
        pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv')
    positioner=positioner_obj('D500B',def_path=config_dir+'D500B_robot_extended_config.yml',tool_file_path=config_dir+'positioner_tcp.csv',\
		pulse2deg_file_path=config_dir+'D500B_pulse2deg_real.csv',base_transformation_file=config_dir+'D500B_pose.csv')

    # positioner_joints = np.radians([-15,180])

    ################## Read geometry data ##################
    data_dir = '../../data/wall_weld_test/'

    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/', 'weld_fujiscan_2025_02_26_17_39_17/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/', 'weld_fujiscan_2025_02_26_16_24_21/']
    logdata_dir_all = ['weld_fujicontrol_2025_03_12_18_27_33/']
    # logdata_dir_all = ['weld_fujiscan_2025_02_26_18_08_18/']

    run_code_again_flag = False # For scanner leading case, need to generate all profile height before actually get dh.
    create_transform = False
    for logdata_dir_name in logdata_dir_all:
        print('Processing:',logdata_dir_name)

        ## determine if the scanner is leading or lagging
        scanner_lagging= False
        if 'scan' in logdata_dir_name:
            scanner_lagging= True
        
        logdata_dir = data_dir+logdata_dir_name
        
        with open(logdata_dir+'weld_meta_data.yml', 'r') as f:
            meta_data = yaml.safe_load(f)

        last_profile_height = None
        # build layers from bottom to top by layers
        all_pcd_transform = []
        all_profile_height = []
        if create_transform or scanner_lagging:
            Transz0_H_odd = None
            Transz0_H_even = None
            Transicp_H_odd2even = None
        else:
            Transz0_H_odd = np.loadtxt(logdata_dir+'Transz0_H_odd.csv',delimiter=',')
            Transz0_H_even = np.loadtxt(logdata_dir+'Transz0_H_even.csv',delimiter=',')
            Transicp_H_odd2even = np.loadtxt(logdata_dir+'Trans_icp_odd2even.csv',delimiter=',')
            Transz0_H_odd = Transz0_H_odd @ Transicp_H_odd2even
        for weld_parts in ['base','layer']:
        # for weld_parts in ['layer']:
            if weld_parts == 'base':
                total_layers_name = glob.glob(logdata_dir+'baselayer*')
            else:
                total_layers_name = glob.glob(logdata_dir+'layer*')
            # get printed layer number
            layer_nums = []
            for layer_name in total_layers_name:
                this_layer = layer_name.split('\\')[-1]
                this_layer = this_layer.split('r')[-1]
                layer_nums.append(int(this_layer))
            layer_nums = np.sort(layer_nums)

            # for layer_n in [layer_nums[-1],layer_nums[-2]]:
            for layer_n_id, layer_n in enumerate(layer_nums):
                # read layer curve data
                if weld_parts == 'base':
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/baselayer{layer_n}_0.csv',delimiter=',')
                else:
                    curve = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_n}_0.csv',delimiter=',')

                # read logged data
                if weld_parts == 'base':
                    layer_name = 'baselayer'+str(layer_n)
                    next_layer_name = 'baselayer'+str(layer_nums[layer_n_id+1]) if layer_n_id+1 < len(layer_nums) else 'layer9'
                else:
                    layer_name = 'layer'+str(layer_n)
                    next_layer_name = 'layer'+str(layer_nums[layer_n_id+1]) if layer_n_id+1 < len(layer_nums) else ''
                print('Processing layer:',layer_name)
                this_layer_dir = logdata_dir+layer_name+'/'
                next_layer_dir = logdata_dir+next_layer_name+'/'
                rob_js_exe = np.loadtxt(this_layer_dir+'weld_js_exe.csv',delimiter=',')
                # get js at index 1~6 and 13 14
                rob_js_exe = rob_js_exe[:,[0,1,2,3,4,5,6,13,14]]
                robot_stamps = rob_js_exe[:,0]
                with open(this_layer_dir+'scan_exe.pickle', 'rb') as f:
                    scan_exe = pickle.load(f)
                
                assert len(rob_js_exe) == len(scan_exe), 'Weld joint and scan data length mismatched'

                ############### get welding commands #####################
                weld_cmd = np.loadtxt(this_layer_dir+'weld_cmd.csv',delimiter=',')

                ############### get welding js ####################
                weld_split_id = np.argmax(np.diff(robot_stamps))
                scan_js_exe = deepcopy(rob_js_exe)
                if scanner_lagging:
                    weld_js_exe = rob_js_exe[:weld_split_id+1,:]
                else:
                    weld_js_exe = rob_js_exe[weld_split_id+1:,:]

                ############### get welding status ##############
                print("Getting welding status...")
                welding_status = np.loadtxt(this_layer_dir+'welding.csv',delimiter=',',skiprows=1)

                ############### get thermal readings ##############
                print("Getting thermal readings...")
                try:
                    thermal_reading = np.loadtxt(this_layer_dir+'thermal.csv',delimiter=',')
                except FileNotFoundError:
                    with open(this_layer_dir+'ir_recording.pickle', 'rb') as f:
                        ir_exe = pickle.load(f)
                    ir_stamp = np.loadtxt(this_layer_dir+'ir_stamps.csv',delimiter=',')
                    horizontal_offset=0
                    vertical_offset=3
                    ir_pixel_window_size=7
                    flame_centroid_history=[]
                    thermal_reading = []
                    thermal_stamp = []
                    for (ir_image_raw,stamp) in zip(ir_exe,ir_stamp):
                        ir_image = np.rot90(ir_image_raw, k=-1)
                        # centroid, bbox, torch_centroid, torch_bbox=weld_detection_aluminum(ir_image,torch_model,percentage_threshold=0.8)
                        # centroid, bbox, torch_centroid, torch_bbox=weld_detection_steel(ir_image,torch_model,tip_wire_model)
                        # find max pixel value in ir_image
                        centroid = np.unravel_index(np.argmax(ir_image, axis=None), ir_image.shape)
                        # if ir_image[centroid] < 1e4:
                        #     continue
                        # draw bbox and centroid on ir_image
                        if centroid is not None:
                            ###weighted history filter
                            if len(flame_centroid_history) > 30:
                                flame_centroid_history.pop(0)
                                # Calculate the weight for the previous history values
                                previous_weight = 0.8 / len(flame_centroid_history)
                                centroid = 0.2 * centroid + np.sum(np.array(flame_centroid_history) * previous_weight, axis=0)
                                flame_centroid_history.append(centroid)

                            #find average pixel value 
                            pixel_coord = (int(centroid[0]) + horizontal_offset, int(centroid[1]) + vertical_offset)
                            pixel_coord = pixel_coord[::-1]
                            flame_reading=get_pixel_value(ir_image,pixel_coord,ir_pixel_window_size)
                            thermal_reading.append(flame_reading)
                            thermal_stamp.append(stamp)
                            # print(flame_reading, centroid)
                            # show image
                        # plt.imshow(ir_image, cmap='inferno', aspect='auto')
                        # plt.colorbar(format='%.2f')
                        # plt.pause(0.1)
                        # plt.clf()
                    # save thermal readings
                    thermal_reading = np.vstack((thermal_stamp,thermal_reading)).T
                    np.savetxt(this_layer_dir+'thermal.csv',thermal_reading,delimiter=',')

                ################ get speed ##############
                print("Getting speed...")
                try:
                    weld_relative_exe = np.loadtxt(this_layer_dir+'weld_relative_exe.csv',delimiter=',')
                    weld_relative_v_exe = np.loadtxt(this_layer_dir+'weld_relative_v_exe.csv',delimiter=',')
                except FileNotFoundError:
                    weld_relative_exe = []
                    weld_relative_v_exe = []
                    for i in range(len(weld_js_exe)):
                        t1_world = robot_weld.fwd(weld_js_exe[i,1:7])
                        t2_world = positioner.fwd(weld_js_exe[i,7:],world=True)
                        t1_t2 = t2_world.inv()*t1_world
                        weld_relative_exe.append(t1_t2.p)
                    weld_relative_exe = np.array(weld_relative_exe)
                    weld_relative_v_exe=np.linalg.norm(np.diff(weld_relative_exe,axis=0),2,1)/np.diff(weld_js_exe[:,0])
                    weld_relative_v_exe=np.append(weld_relative_v_exe[0],weld_relative_v_exe)
                    weld_relative_v_exe=moving_average(weld_relative_v_exe,padding=True)
                    weld_relative_v_exe=moving_average(weld_relative_v_exe,padding=True)
                    np.savetxt(this_layer_dir+'weld_relative_exe.csv',weld_relative_exe,delimiter=',')
                    np.savetxt(this_layer_dir+'weld_relative_v_exe.csv',weld_relative_v_exe,delimiter=',')
                #############################################
                
                ############### get height and width ##############
                print("Getting height and width...")
                try:
                    profile_height = np.loadtxt(this_layer_dir+'profile_height.csv',delimiter=',')
                    profile_width = np.loadtxt(this_layer_dir+'profile_width.csv',delimiter=',')
                    all_profile_height.append(profile_height)
                    # pcd = o3d.io.read_point_cloud(this_layer_dir+'pcd.pcd')
                    pcd_denoise = o3d.io.read_point_cloud(this_layer_dir+'pcd_denoise.pcd')
                    all_pcd_transform.append(pcd_denoise)

                except FileNotFoundError:
                    # processing the scans
                    scan_process = ScanProcess(robot_scan,positioner)

                    # Single scan 2D reconstruction
                    try:
                        with open(this_layer_dir+'scan_exe_noise_remove.pickle', 'rb') as f:
                            scan_exe_noise_remove = pickle.load(f)
                    except FileNotFoundError:
                        scan_exe_noise_remove = []
                        duration_list = []
                        for (weld_js,scan) in zip(scan_js_exe,scan_exe):
                            st = time.time()
                            scan_noise_remove = scan_process.scan2dDenoise(deepcopy(scan).T,crop_min=[-40,30],crop_max=[40,200])
                            scan_exe_noise_remove.append(scan_noise_remove)
                            duration_list.append(time.time()-st)
                        # plt.plot(duration_list)
                        # plt.show()
                        # print("Average single scan 2D reconstruction time:",np.mean(duration_list))
                        # print("Max single scan 2D reconstruction time:",np.max(duration_list))
                        with open(this_layer_dir+'scan_exe_noise_remove.pickle', 'wb') as f:
                            pickle.dump(scan_exe_noise_remove, f)

                    # whole layer 3D reconstruction
                    pcd=None
                    # pcd = scan_process.pcd_register_mti(scan_exe,scan_js_exe[:,:6],robot_stamps,static_positioner_q=positioner_joints,flip=True,scanner='fuji')
                    pcd = scan_process.pcd_register_mti(scan_exe_noise_remove,scan_js_exe[:,1:],robot_stamps,flip=True,scanner='fuji')
                    # visualize_pcd([pcd])
                    # move pcd_noise_preremoved in y direction
                    # pcd_noise_preremoved = pcd_noise_preremoved.translate((0,200,0))
                    # visualize_pcd([pcd,pcd_noise_preremoved])

                    # cropping the point cloud
                    curve_planned_z = np.mean(curve[:,2])
                    curve_x_end = np.min(curve[:,0])
                    curve_x_start = np.max(curve[:,0])
                    curve_y = np.mean(curve[:,1])
                    if scanner_lagging:
                        z_height_start=curve_planned_z+0.1
                    else:
                        z_height_start=curve_planned_z-5
                        if layer_n == layer_nums[-1] and weld_parts == 'layer':
                            curve_prev = np.loadtxt(data_dir+f'curve_sliced_relative/slice{layer_nums[layer_n_id-1]}_0.csv',delimiter=',')
                            curve_prev_z = np.mean(curve_prev[:,2])
                            z_height_start = curve_prev_z
                    # z_height_start = 0
                    # print(z_height_start)
                    # print(curve_y)
                    crop_extend_x=10
                    crop_extend_z=20
                    crop_min=(curve_x_end-crop_extend_x,curve_y-30,-30)
                    crop_max=(curve_x_start+crop_extend_x,curve_y+30,z_height_start+crop_extend_z)
                    crop_h_min=(curve_x_end-crop_extend_x,curve_y-20,-30)
                    crop_h_max=(curve_x_start+crop_extend_x,curve_y+20,z_height_start+crop_extend_z)
                    # profile_height_noise, profile_width_noise,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    pcd = scan_process.pcd_noise_remove(pcd,min_bound=crop_min,max_bound=crop_max,outlier_remove=False,cluster_based_outlier_remove=False)
                    pcd_denoise = scan_process.pcd_noise_remove(pcd,crop_flag=False,nb_neighbors=40,std_ratio=1.5,min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=100)
                    # Transz0_H = None
                    Transz0_H = deepcopy(Transz0_H_even) if layer_n_id % 2 == 0 else deepcopy(Transz0_H_odd)
                    profile_height, _,Transz0_H = scan_process.pcd2height(deepcopy(pcd_denoise),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    _, profile_width,_ = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H,return_width=True)
                    
                    if create_transform:
                        if layer_n_id % 2 == 0:
                            Transz0_H_even = deepcopy(Transz0_H)
                            np.savetxt(logdata_dir+'Transz0_H_even.csv',Transz0_H_even,delimiter=',')
                        else:
                            Transz0_H_odd = deepcopy(Transz0_H)
                            if layer_n_id == 1 and weld_parts == 'base':
                                Transz0_H_odd[0,-1] -= 4
                                Transz0_H_odd[1,-1] += 1
                            np.savetxt(logdata_dir+'Transz0_H_odd.csv',Transz0_H_odd,delimiter=',')
                        pcd_transform = deepcopy(pcd_denoise)
                        pcd_transform.transform(Transz0_H)
                        all_pcd_transform.append(pcd_transform)
                        if 'control' in logdata_dir_name and layer_n_id == 1 and weld_parts == 'layer':
                            print("Transforming pcd using icp")
                            threshold = 1
                            for pcd_i in [-2,-1]:
                                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1e5,-1e5,0),max_bound=(1e5,1e5,1e5))
                                all_pcd_transform[pcd_i]=all_pcd_transform[pcd_i].crop(bbox)
                            reg_p2p = o3d.pipelines.registration.registration_icp(
                                        all_pcd_transform[-1], all_pcd_transform[-2], threshold, np.eye(4),
                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
                            np.savetxt(logdata_dir+'Trans_icp_odd2even.csv',reg_p2p.transformation,delimiter=',')
                    else:
                        pcd_denoise_trans = deepcopy(pcd_denoise)
                        pcd_denoise_trans.transform(Transz0_H)
                        all_pcd_transform.append(pcd_denoise_trans)

                    # visualize_pcd([pcd_denoise_trans])
                    # if len(all_pcd_transform) != 0:
                    #     cmap = plt.get_cmap('jet')
                    #     color = cmap(np.linspace(0, 1, len(all_pcd_transform)))
                    #     for i in range(len(all_pcd_transform)):
                    #         all_pcd_transform[i].paint_uniform_color(color[i][:3])
                    #     visualize_pcd(all_pcd_transform)

                    # apply 1D smoother to profile_width
                    profile_width[:,1] = np.convolve(profile_width[:,1], np.ones(5)/5, mode='same')
                    # profile_width_noise[:,1] = np.convolve(profile_width_noise[:,1], np.ones(5)/5, mode='same')

                    # save processed profile height and point cloud
                    np.savetxt(this_layer_dir+'profile_height.csv',profile_height,delimiter=',')
                    np.savetxt(this_layer_dir+'profile_width.csv',profile_width,delimiter=',')
                    o3d.io.write_point_cloud(this_layer_dir+'pcd.pcd',pcd)
                    o3d.io.write_point_cloud(this_layer_dir+'pcd_denoise.pcd',pcd_denoise)
                    #############################################

                ################ combine everything in one array ##############
                if not scanner_lagging:
                    if layer_n_id == len(layer_nums)-1 and weld_parts == 'layer':
                        # if the scanner is leading, the last layer is not welding anything.
                        break
                    try:
                        next_scan_height = np.loadtxt(next_layer_dir+'profile_height.csv',delimiter=',')
                        next_scan_width = np.loadtxt(next_layer_dir+'profile_width.csv',delimiter=',')
                    except FileNotFoundError:
                        next_scan_height = deepcopy(profile_height)
                        next_scan_width = deepcopy(profile_width)
                        run_code_again_flag = True

                profile_welding = []
                for js_id,x in enumerate(weld_relative_exe[:,0]):
                # for x_id, x in enumerate(profile_height[:,0]):
                    # find closest x in weld_relative_exe
                    # js_id = np.argmin(np.abs(weld_relative_exe[:,0]-x))
                    if np.min(np.abs(profile_height[:,0]-x)) > 0.3:
                        continue

                    # time at the same x
                    this_t = weld_js_exe[js_id,0]
                    # weld command right before this time
                    cmd_idx = np.where(weld_cmd[:,0]>=this_t)[0]
                    if len(cmd_idx) == 0:
                        cmd_idx = 0
                    else:
                        cmd_idx = cmd_idx[0]
                    this_cmd_v = weld_cmd[cmd_idx,2]
                    this_cmd_fr = weld_cmd[cmd_idx,3]
                    # velocity at the same x
                    this_v = weld_relative_v_exe[js_id]
                    # height and width at the same x
                    if scanner_lagging:
                        this_height = profile_height[np.argmin(np.abs(profile_height[:,0]-x)),1]
                        if last_profile_height is not None:
                            last_height = last_profile_height[np.argmin(np.abs(last_profile_height[:,0]-x)),1]
                        else:
                            last_height = 0
                    else:
                        this_height = next_scan_height[np.argmin(np.abs(next_scan_height[:,0]-x)),1]
                        last_height = profile_height[np.argmin(np.abs(profile_height[:,0]-x)),1]
                    this_dh = this_height - last_height                    
                    this_width = profile_width[np.argmin(np.abs(profile_width[:,0]-x)),1] if scanner_lagging else next_scan_width[np.argmin(np.abs(next_scan_width[:,0]-x)),1]
                    
                    # torch height
                    torch_height = weld_relative_exe[js_id,2] - last_height
                    # welding status at time t
                    welding_status_idx=np.where(welding_status[:,0]>=this_t)[0]
                    welding_status_idx = -1 if len(welding_status_idx) == 0 else welding_status_idx[0]
                    ratio=(this_t-welding_status[:,0][welding_status_idx-1])/(welding_status[:,0][welding_status_idx]-welding_status[:,0][welding_status_idx-1])
                    this_welding_status=welding_status[:,1:][welding_status_idx-1]*(1-ratio)+welding_status[:,1:][welding_status_idx]*ratio

                    # thermal reading at time t
                    thermal_reading_idx = np.where(thermal_reading[:,0]>=this_t)[0]
                    thermal_reading_idx = -1 if len(thermal_reading_idx) == 0 else thermal_reading_idx[0]
                    ratio=(this_t-thermal_reading[:,0][thermal_reading_idx-1])/(thermal_reading[:,0][thermal_reading_idx]-thermal_reading[:,0][thermal_reading_idx-1])
                    this_thermal_reading=thermal_reading[:,1][thermal_reading_idx-1]*(1-ratio)+thermal_reading[:,1][thermal_reading_idx]*ratio

                    this_welding_profile = np.array([this_t,x,this_cmd_v,this_cmd_fr,this_height,this_dh,torch_height,this_width,this_v,this_thermal_reading])
                    this_welding_profile = np.append(this_welding_profile,this_welding_status)
                    profile_welding.append(this_welding_profile)
                
                profile_welding = np.array(profile_welding)
                # save profile welding with header
                header = 'time,x,cmd_v,cmd_feedrate,height,dheight,torch_height,width,v,thermal,voltage,current,feedrate,energy'
                np.savetxt(this_layer_dir+'profile_welding.csv',profile_welding,delimiter=',',header=header)
                last_profile_height = profile_height

                print("Finished processing layer:",layer_name)

    

        fig, ax = plt.subplots()
        ax.set_title('Profile height')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        for i in range(len(all_profile_height)):
            ax.plot(all_profile_height[i][:,0],all_profile_height[i][:,1],label='Layer '+str(i))
        # ax.legend()
        plt.show()

        if len(all_pcd_transform) != 0:
            cmap = plt.get_cmap('tab10')
            for i in range(len(all_pcd_transform)):
                all_pcd_transform[i].paint_uniform_color(cmap(i%10)[:3])
            visualize_pcd(all_pcd_transform)
        

    if run_code_again_flag:
        print("********** You need to run the code again **********")

if __name__ == '__main__':
    main()