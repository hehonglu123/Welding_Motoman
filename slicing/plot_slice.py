from slicing import *
import glob

vis_step=5
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=5
for i in range(634,637):
    num_sections=len(glob.glob('slicing_result/slice_'+str(i)+'_*.csv'))
    for x in range(num_sections):
        layer=np.loadtxt('slicing_result/slice_%i_%i.csv'%(i,x),delimiter=',')
        ax.plot3D(layer[::vis_step,0],layer[::vis_step,1],layer[::vis_step,2],'r.-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('STL first X Layer Slicing')
plt.show()