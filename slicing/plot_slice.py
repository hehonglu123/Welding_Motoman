from slicing import *
from utils import *
import glob

vis_step=5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=5
data_dir='../data/blade0.1/auto_slice/curve_sliced/'
for i in range(0,500,100):
    num_sections=len(glob.glob(data_dir+'slice%i_*.csv'%i))
    print('num_sections:',num_sections)
    for x in range(num_sections):
        layer=np.loadtxt(data_dir+'slice%i_%i.csv'%(i,x),delimiter=',')
        ax.plot3D(layer[::vis_step,0],layer[::vis_step,1],layer[::vis_step,2],'r.-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
set_axes_equal(ax)
plt.title('STL first X Layer Slicing')
plt.show()