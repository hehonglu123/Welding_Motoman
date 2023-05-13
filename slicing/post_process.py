from slicing import *
import glob

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vis_step=1

slice_all=[]
for i in range(759):
    slice_ith_layer=[]
    for x in range(len(glob.glob('temp/slice%i_*.csv'%i))):
        slice_ith_layer.append(np.loadtxt('temp/slice%i_%i.csv'%(i,x),delimiter=','))
        
    slice_all.append(slice_ith_layer)

for i in range(759):
    for x in range(len(glob.glob('temp/slice%i_*.csv'%i))):
        try:
            ax.plot3D(slice_all[i][x][::vis_step,0],slice_all[i][x][::vis_step,1],slice_all[i][x][::vis_step,2],'r.-')
        except:
            break
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
slice_all,curve_normal_all=post_process(slice_all,point_distance=1)