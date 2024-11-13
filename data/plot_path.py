import numpy as np
from matplotlib import pyplot as plt

data_dir = 'face_mesh_tanja_straight/slicing_result_10/curve_sliced/'

vis_step=1

for base_i in range(0,500,5):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for i in range(5):
        curve_sliced_relative = np.loadtxt(data_dir+'slice'+str(base_i+i)+'_0.csv',delimiter=',')
        ax.plot3D(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],'b.-')
        ax.quiver(curve_sliced_relative[::vis_step,0],curve_sliced_relative[::vis_step,1],curve_sliced_relative[::vis_step,2],curve_sliced_relative[::vis_step,3],curve_sliced_relative[::vis_step,4],curve_sliced_relative[::vis_step,5],length=0.3, normalize=True)

    plt.show()