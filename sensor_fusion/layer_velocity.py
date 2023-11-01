

base_path = '../data/wall_weld_test/ER4043_correction_100ipm_2023_09_27_20_53_05/layer_3/'
weld_q=np.loadtxt(next_weld_dir+'weld_js_exe.csv',delimiter=',')
weld_stamp=np.loadtxt(next_weld_dir+'weld_robot_stamps.csv',delimiter=',')
scan_dir=next_weld_dir+'scans/'
profile_height = np.load(next_scan_dir+'height_profile.npy')