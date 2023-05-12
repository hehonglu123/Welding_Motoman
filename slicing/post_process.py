from slicing import *
import glob

slice_all=[]
for i in range(759):
    slice_ith_layer=[]
    for x in range(len(glob.glob('slicing_result/slice%i_*.csv'%i))):
        slice_ith_layer.append(np.loadtxt('slicing_result/slice%i_%i.csv'%(i,x),delimiter=','))
    slice_all.append(slice_ith_layer)


slice_all,curve_normal_all=post_process(slice_all,point_distance=1)