from general_robotics_toolbox import *
import numpy as np
from weld_dh2v import *
from copy import deepcopy
import sys
sys.path.append('../toolbox/')
sys.path.append('../scan/scan_tools/')
sys.path.append('../scan/scan_plan/')
sys.path.append('../scan/scan_process/')
from robot_def import *
from scan_utils import *
from scan_continuous import *
from scanPathGen import *
from scanProcess import *
from WeldSend import *
from weldRRSensor import *
from dx200_motion_program_exec_client import *

zero_config=np.zeros(6)

class WeldScan(object):
    def __init__(self,robot_weld,robot_scan,positioner,weldsend,weldRR,mti_client,\
                to_start_s=6,to_home_s=8) -> None:
        
        # robots
        self.robot_weld=robot_weld
        self.robot_scan=robot_scan
        self.positioner=positioner
        # weld send tool
        self.ws=weldsend
        # sensor data collection
        self.wrr=weldRR
        # mti sensor
        self.mti_client=mti_client
        self.to_start_s=to_start_s
        self.to_home_s=to_home_s
        
        # 2. Scanning parameters
        ### scan parameters
        self.scan_speed=5 # scanning speed (mm/sec)
        scan_stand_off_d = 95 ## mm
        Rz_angle = np.radians(0) # point direction w.r.t welds
        Ry_angle = np.radians(0) # rotate in y a bit
        bounds_theta = np.radians(1) ## circular motion at start and end
        extension = 10 ## extension before and after (mm)
        ### scanning path module
        self.spg = ScanPathGen(self.robot_scan,self.positioner,scan_stand_off_d,Rz_angle,Ry_angle,bounds_theta,extension)
        self.all_scan_angle = np.radians([0]) ## scan angle
        self.scan_table=np.radians([-15,200]) ## init table
        self.mti_Rpath = np.array([[ -1.,0.,0.],   
                    [ 0.,1.,0.],
                    [0.,0.,-1.]])
    
    def robot_weld_scan(self,curve,curve_scan,rob_v,ipm_mode,T_R1Base_S1TCP,\
                        robot_weld_mid,robot_weld_home,positioner_weld_q,\
                        robot_scan_mid,robot_scan_home,positioner_scan_q,\
                        arc_on=False,Transz0_H=None,draw_dh=False,skip_weld=False,wait_signal=True):
        
        assert len(curve)-1==len(rob_v), "rob_v must have length equals len(curve)-1"
        rob_v=np.append(rob_v[0],rob_v)
        
        ###################### welding ###########################
        ipm_job_num = int(ipm_mode/10+200)
        
        ## fix torch R because only the welding robot moves
        torch_Rz=[0,0,-1]
        torch_Rx=[-1,0,0]
        torch_Ry=[0,1,0]
        torch_R=np.array([torch_Rx,torch_Ry,torch_Rz]).T
        
        # find curve in R1 frame
        path_T=[]
        cp_i=0
        for cp in curve:
            this_p = T_R1Base_S1TCP[:3,:3]@cp[:3]+T_R1Base_S1TCP[:3,-1]
            this_R = torch_R
            path_T.append(Transform(this_R,this_p))
            cp_i+=1
        # add path collision avoidance
        path_T.insert(0,Transform(path_T[0].R,path_T[0].p+np.array([0,0,50])))
        path_T.append(Transform(path_T[-1].R,path_T[-1].p+np.array([0,0,50])))
        # get path q
        path_q = []
        for tcp_T in path_T:
            path_q.append(self.robot_weld.inv(tcp_T.p,tcp_T.R,zero_config)[0])
        # get path velocity and primitives
        primitives=['movel']*len(rob_v)

        # to welding start position
        r1_start_path=[path_q[0],path_q[1]] if robot_weld_mid is None else [robot_weld_mid,path_q[0],path_q[1]]
        self.ws.jog_dual(self.robot_weld,self.positioner,r1_start_path,positioner_weld_q,self.to_start_s)
        
        # start welding and logging sensor information
        if not skip_weld:
            if wait_signal:
                input("start weld")
            self.wrr.start_all_sensors()
            weld_stamps,weld_js_exe,_,_=self.ws.weld_segment_single(primitives,self.robot_weld,path_q[1:-1],rob_v,cond_all=[ipm_job_num],arc=arc_on)
            self.wrr.stop_all_sensors()
        else:
            weld_stamps=np.zeros(10)
            weld_js_exe=np.zeros((10,14))
        
        # robot to home
        if wait_signal:
            input("weld home")
        r1_home_path=[path_q[-1],robot_weld_home] if robot_weld_mid is None else [path_q[-1],robot_weld_mid,robot_weld_home]
        self.ws.jog_single(self.robot_weld,r1_home_path,self.to_home_s)
        #######################################################
        
        ###################### scanning ##############################
        # generate scan path
        scan_p,scan_R,q_out1,q_out2=self.spg.gen_scan_path([curve_scan],[0],self.all_scan_angle,\
                        solve_js_method=0,q_init_table=positioner_scan_q,R_path=self.mti_Rpath,scan_path_dir=None)
        # generate motion program
        q_bp1,q_bp2,s1_all,s2_all=self.spg.gen_motion_program(q_out1,q_out2,scan_p,self.scan_speed,init_sync_move=0)

        # scanning motion
        while True:
            # to scanning start position
            r2_start_path=[q_bp1[0][0]] if robot_scan_mid is None else [robot_scan_mid,q_bp1[0][0]]
            self.ws.jog_dual(self.robot_scan,self.positioner,r2_start_path,q_bp2[0][0],self.to_start_s)

            if wait_signal:
                input("start scan")
            ## motion start
            mp = MotionProgram(ROBOT_CHOICE='RB2',ROBOT_CHOICE2='ST1',pulse2deg=self.robot_scan.pulse2deg,pulse2deg_2=self.positioner.pulse2deg)
            # routine motion
            for path_i in range(1,len(q_bp1)-1):
                target2=['MOVJ',np.degrees(q_bp2[path_i][0]),s2_all[path_i]]
                mp.MoveL(np.degrees(q_bp1[path_i][0]), s1_all[path_i], target2=target2)
            target2=['MOVJ',np.degrees(q_bp2[-1][0]),s2_all[-1]]
            mp.MoveL(np.degrees(q_bp1[-1][0]), s1_all[-1], 0, target2=target2)

            mti_break_flag=False
            self.ws.client.execute_motion_program_nonblocking(mp)
            ###streaming
            self.ws.client.StartStreaming()
            start_time=time.time()
            state_flag=0
            joint_recording=[]
            scan_stamps=[]
            mti_recording=None
            mti_recording=[]
            while True:
                if state_flag & STATUS_RUNNING == 0 and time.time()-start_time>1.:
                    break 
                res, fb_data = self.ws.client.fb.try_receive_state_sync(self.ws.client.controller_info, 0.001)
                if res:
                    joint_angle=np.hstack((fb_data.group_state[0].feedback_position,fb_data.group_state[1].feedback_position,fb_data.group_state[2].feedback_position))
                    state_flag=fb_data.controller_flags
                    joint_recording.append(joint_angle)
                    timestamp=fb_data.time
                    scan_stamps.append(timestamp)
                    ###MTI scans YZ point from tool frame
                    try:
                        mti_recording.append(deepcopy(np.array([self.mti_client.lineProfile.X_data,self.mti_client.lineProfile.Z_data])))
                    except Exception as e:
                        if not mti_break_flag:
                            print(e)
                        mti_break_flag=True
            self.ws.client.servoMH(False)
            if not mti_break_flag:
                break
            print("MTI broke during robot move")
            while True:
                try:
                    input("MTI reconnect ready?")
                    self.regenerate_mti_rr()
                    break
                except:
                    pass
        mti_recording=np.array(mti_recording)
        joint_recording=np.array(joint_recording)
        scan_js_exe=joint_recording[:,6:]
        
        # scanning process: processing point cloud and get h
        
        z_height_start=curve_scan[0][2]-3
        crop_extend=15
        crop_min=tuple(np.min(curve_scan[:,:3],axis=0)-crop_extend)
        crop_max=np.max(curve_scan[:,:3],axis=0)+crop_extend
        crop_max[2]+=20
        crop_max=tuple(crop_max)
        
        crop_h_min=crop_min
        crop_h_max=crop_max
        
        scan_process = ScanProcess(self.robot_scan,self.positioner)
        pcd = scan_process.pcd_register_mti(mti_recording,scan_js_exe,scan_stamps,static_positioner_q=positioner_scan_q)
        pcd = scan_process.pcd_noise_remove(pcd,nb_neighbors=40,std_ratio=1.5,\
                                            min_bound=crop_min,max_bound=crop_max,cluster_based_outlier_remove=True,cluster_neighbor=1,min_points=300)
        # profile_height,Transz0_H = scan_process.pcd2height(deepcopy(pcd),z_height_start,bbox_min=crop_h_min,bbox_max=crop_h_max,Transz0_H=Transz0_H)
        # calibrate H
        pcd,Transz0_H = scan_process.pcd_calib_z(pcd,Transz0_H=Transz0_H)
        profile_dh = scan_process.pcd2dh(pcd,curve,drawing=draw_dh)
        # if draw_dh:
        #     plt.scatter(profile_dh[:,0],profile_dh[:,1])
        #     plt.show()
        
        # robot to home
        if wait_signal:
            input("scan home")
        r2_home_path=[robot_scan_home] if robot_scan_mid is None else [robot_scan_mid,robot_scan_home]
        self.ws.jog_single(self.robot_scan,r2_home_path,self.to_home_s)
        ########################################
        
        return profile_dh,weld_js_exe,weld_stamps,scan_js_exe,scan_stamps,mti_recording,pcd,Transz0_H

    def regenerate_mti_rr(self):
        
        self.mti_sub=RRN.SubscribeService("rr+tcp://192.168.55.10:60830/?service=MTI2D")
        self.mti_sub.ClientConnectFailed += self.connect_failed
        self.mti_client=self.mti_sub.GetDefaultClientWait(1)
        self.mti_client.setExposureTime("25")
    
    def connect_failed(self,s, client_id, url, err):
        print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
        self.mti_sub=RRN.SubscribeService(url)
        self.mti_client=self.mti_sub.GetDefaultClientWait(1)
        