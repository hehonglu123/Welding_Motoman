import weldmicphone as wm

base_path = '../data/wall_weld_test/moveL_100_weld_scan_2023_08_02_15_17_25/layer_1/'

wm.audio_denoise(base_path)
a,b,c = wm.audio_MFCC(base_path)
print(a,b,c)