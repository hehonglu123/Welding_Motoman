import cv2,wave,copy
import pickle, sys
import numpy as np
import scipy.signal as signal
sys.path.append('../toolbox/')
from flir_toolbox import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 定义一个函数来均匀地分割一个列表
def chunk_list(l, n):
    chunk_size = len(l) // n
    return [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]

profile_height_10=np.load('../data/wall_weld_test/weld_scan_2023_08_23_15_23_45/layer_2/scans/height_profile.npy')
profile_height_9=np.load('../data/wall_weld_test/weld_scan_2023_08_23_15_23_45/layer_1/scans/height_profile.npy')

min_length = min(len(profile_height_10), len(profile_height_9))

profile_height_10 = profile_height_10[:min_length]
profile_height_9 = profile_height_9[:min_length]

difference = [(x1, y1 - y2) for (x1, y1), (_, y2) in zip(profile_height_10, profile_height_9)]

# 拆分坐标以便于绘图
x_values, y_values = zip(*difference)



# 将数据平均分成20份
x_chunks = chunk_list(list(x_values), 20)
y_chunks = chunk_list(list(y_values), 20)

# 获取实际的块数
num_chunks = min(len(x_chunks), len(y_chunks))

# 如果我们的颜色列表不足，我们可以重复使用颜色或选择一个更大的颜色列表
colors = list(mcolors.TABLEAU_COLORS.values()) * (num_chunks // len(mcolors.TABLEAU_COLORS) + 1)

# 绘制每个分段
plt.figure(figsize=(10,6))
for i in range(num_chunks):
    plt.plot(x_chunks[i], y_chunks[i], label=f'Segment {i+1}', color=colors[i])

plt.xlabel('X values')
plt.ylabel('Difference in Y values')
plt.title('Difference in Y values between profile_height_10 and profile_height_9')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()