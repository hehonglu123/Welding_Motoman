import numpy as np
from copy import deepcopy
from general_robotics_toolbox import *

def strategy_1():
    pass

def strategy_2(profile_height,last_mean_h,forward_flag,curve_sliced_relative,R_S1TCP,original_v,\
               noise_h_thres=3,peak_threshold=0.25,flat_threshold=2.5,correct_thres = 1.5,patch_nb = 2,\
               start_ramp_ratio = 0.67,end_ramp_ratio = 0.33):
    
    ## parameters
    noise_h_thres = 3
    peak_threshold=0.25
    flat_threshold=2.5
    correct_thres = 1.5
    patch_nb = 2 # 2*0.1
    start_ramp_ratio = 0.67
    end_ramp_ratio = 0.33
    #############

    ### delete noise
    mean_h = np.mean(profile_height[:,1])
    profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h>noise_h_thres),axis=0)
    profile_height=np.delete(profile_height,np.where(profile_height[:,1]-mean_h<-noise_h_thres),axis=0)
    ###

    h_largest = np.max(profile_height[:,1])
    # 1. h_target = last height point + designated dh value
    # h_target = h_largest+1.2
    # 2. h_target = last mean h + last_dh
    dh_last_layer = mean_h-last_mean_h
    h_target = mean_h+dh_last_layer

    profile_slope = np.gradient(profile_height[:,1])/np.gradient(profile_height[:,0])
    # find slope peak
    weld_terrain=[]
    last_peak_i=None
    lastlast_peak_i=None
    for sample_i in range(len(profile_slope)):
        if np.fabs(profile_slope[sample_i])<peak_threshold:
            weld_terrain.append(0)
        else:
            if profile_slope[sample_i]>=peak_threshold:
                weld_terrain.append(1)
            elif profile_slope[sample_i]<=-peak_threshold:
                weld_terrain.append(-1)
            if lastlast_peak_i:
                if (weld_terrain[-1]==weld_terrain[lastlast_peak_i]) and (weld_terrain[-1]!=weld_terrain[last_peak_i]):
                    weld_terrain[last_peak_i]=0
            lastlast_peak_i=last_peak_i
            last_peak_i=sample_i

    weld_terrain=np.array(weld_terrain)
    weld_peak=[]
    weld_peak_id=[]
    last_peak=None
    last_peak_i=None
    for sample_i in range(len(profile_slope)):
        if weld_terrain[sample_i]!=0:
            if last_peak is None:
                weld_peak.append(profile_height[sample_i])
                weld_peak_id.append(sample_i)
            else:
                # if the terrain change
                if (last_peak>0 and weld_terrain[sample_i]<0) or (last_peak<0 and weld_terrain[sample_i]>0):
                    weld_peak.append(profile_height[last_peak_i])
                    weld_peak.append(profile_height[sample_i])
                    weld_peak_id.append(last_peak_i)
                    weld_peak_id.append(sample_i)
                else:
                    # the terrain not change but flat too long
                    if profile_height[sample_i,0]-profile_height[last_peak_i,0]>flat_threshold:
                        weld_peak.append(profile_height[last_peak_i])
                        weld_peak.append(profile_height[sample_i])
                        weld_peak_id.append(last_peak_i)
                        weld_peak_id.append(sample_i)
            last_peak=deepcopy(weld_terrain[sample_i])
            last_peak_i=sample_i
    weld_peak=np.array(weld_peak)
    weld_peak_id=np.array(weld_peak_id)

    correction_index = np.where(profile_height[:,1]-h_largest<-1*correct_thres)[0]

    # identified patch
    correction_patches = []
    patch=[]
    for i in range(len(correction_index)):
        if len(patch)==0:
            patch = [correction_index[i]]
        else:
            if correction_index[i]-patch[-1]>patch_nb:
                correction_patches.append(deepcopy(patch))
                patch=[correction_index[i]]
            else:
                patch.append(correction_index[i])
    correction_patches.append(deepcopy(patch))
    # find motion start/end using ramp before and after patch
    motion_patches=[]
    for patch in correction_patches:
        motion_patch=[]
        # find start
        start_i = patch[0]
        if np.all(weld_peak_id>=start_i):
            motion_patch.append(start_i)
        else:
            start_ramp_start_i = np.where(weld_peak_id<=start_i)[0][-1]
            start_ramp_end_i = np.where(weld_peak_id>start_i)[0][0]
            start_ramp_start_i = max(0,start_ramp_start_i)
            start_ramp_end_i = min(start_ramp_end_i,len(weld_peak_id)-1)
            if profile_slope[weld_peak_id[start_ramp_start_i]]>0:
                start_ramp_start_i=start_ramp_start_i+1
                start_ramp_end_i=start_ramp_end_i+1
            if profile_slope[weld_peak_id[start_ramp_end_i]]>0:
                start_ramp_start_i=start_ramp_start_i-1
                start_ramp_end_i=start_ramp_end_i-1
            start_ramp_start_i = max(0,start_ramp_start_i)
            start_ramp_end_i = min(start_ramp_end_i,len(weld_peak_id)-1)
            start_ramp_start=weld_peak_id[start_ramp_start_i]
            start_ramp_end=weld_peak_id[start_ramp_end_i]
            
            if forward_flag:
                motion_patch.append(int(np.round(start_ramp_start*end_ramp_ratio+start_ramp_end*(1-end_ramp_ratio))))
            else:
                motion_patch.append(int(np.round(start_ramp_start*start_ramp_ratio+start_ramp_end*(1-start_ramp_ratio))))
        # find end
        end_i = patch[-1]
        if np.all(weld_peak_id<=end_i):
            motion_patch.append(end_i)
        else:
            end_ramp_start_i = np.where(weld_peak_id<=end_i)[0][-1]
            end_ramp_end_i = np.where(weld_peak_id>end_i)[0][0]
            if profile_slope[weld_peak_id[end_ramp_start_i]]<0:
                end_ramp_start_i=end_ramp_start_i+1
                end_ramp_end_i=end_ramp_end_i+1
            if profile_slope[weld_peak_id[end_ramp_end_i]]<0:
                end_ramp_start_i=end_ramp_start_i-1
                end_ramp_end_i=end_ramp_end_i-1
            end_ramp_start=weld_peak_id[end_ramp_start_i]
            end_ramp_end=weld_peak_id[end_ramp_end_i]
            
            if forward_flag:
                motion_patch.append(int(np.round(end_ramp_end*start_ramp_ratio+end_ramp_start*(1-start_ramp_ratio))))
            else:
                motion_patch.append(int(np.round(end_ramp_end*end_ramp_ratio+end_ramp_start*(1-end_ramp_ratio))))
        
        if forward_flag:
            motion_patches.append(motion_patch[::-1])
        else:
            motion_patches.append(motion_patch)
    if forward_flag:
        motion_patches=motion_patches[::-1]

    # find v
    # 140 ipm: dh=0.006477*v^2-0.2362v+3.339
    # 160 ipm: dh=0.006043*v^2-0.2234v+3.335    
    # new curve in positioner frame
    curve_sliced_relative_correct = []
    this_p = np.array([curve_sliced_relative[0][0],curve_sliced_relative[0][1],h_largest])
    curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative[0][3:]))
    path_T_S1 = [Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3])]
    this_weld_v = []
    all_dh=[]
    curve_x_upper = np.max(curve_sliced_relative[0,0],curve_sliced_relative[-1,0])
    curve_x_lower = np.min(curve_sliced_relative[0,0],curve_sliced_relative[-1,0]) 
    for mo_patch in motion_patches:
        if curve_x_upper>=profile_height[mo_patch[0],0]>=curve_x_lower:
            this_weld_v.append(original_v)
            this_p = np.array([profile_height[mo_patch[0],0],curve_sliced_relative[0][1],h_target])
            curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_correct[0][3:]))
            path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
        else:
            pass
        if curve_x_upper>=profile_height[mo_patch[-1],0]>=curve_x_lower:
            dh = h_largest-np.min(profile_height[np.min(mo_patch):np.max(mo_patch),1])
            all_dh.append(dh)
            a=0.006477
            b=-0.2362
            c=3.339-dh
            v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
            this_weld_v.append(v)

            this_p = np.array([profile_height[mo_patch[-1],0],curve_sliced_relative[0][1],h_target])
            curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_correct[0][3:]))
            path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))
        else:
            dh = h_largest-np.min(profile_height[np.min(motion_patches[-1]):np.max(motion_patches[-1]),1])
            all_dh.append(dh)
            a=0.006477
            b=-0.2362
            c=3.339-dh
            v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
            this_weld_v.append(v)

    if len(this_weld_v)<len(curve_sliced_relative_correct):
        this_weld_v.append(original_v)

    this_p = np.array([curve_sliced_relative[0][0],curve_sliced_relative[0][1],h_target])
    curve_sliced_relative_correct.append(np.append(this_p,curve_sliced_relative_correct[0][3:]))
    path_T_S1.append(Transform(R_S1TCP,curve_sliced_relative_correct[-1][:3]))

    return curve_sliced_relative_correct,path_T_S1,this_weld_v,all_dh