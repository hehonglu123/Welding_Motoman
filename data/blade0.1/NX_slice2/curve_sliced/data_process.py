import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys, glob
sys.path.append('../../../../toolbox')
from path_calc import *

def find_norm(p1,p2,p3):
	#find normal vector from p1 pointing to line of p2p3
	p2p1=p2-p1
	p3p1=p3-p1
	p2p3=p2-p3
	vec=np.cross(np.cross(p2p1,p3p1),p2p3)
	return vec/np.linalg.norm(vec)

def smooth_curve(curve):
	lam=calc_lam_cs(curve)

	polyfit=np.polyfit(lam,curve,deg=47)
	polyfit_x=np.poly1d(polyfit[:,0])(lam)
	polyfit_y=np.poly1d(polyfit[:,1])(lam)
	polyfit_z=np.poly1d(polyfit[:,2])(lam)
	curve_smooth=np.vstack((polyfit_x, polyfit_y, polyfit_z)).T

	return curve_smooth

def smooth_normal(curve_normal,n=15):
	curve_normal_new=copy.deepcopy(curve_normal)
	for i in range(len(curve_normal)):
		curve_normal_new[i]=np.average(curve_normal[max(0,i-n):min(len(curve_normal),i+n)],axis=0)
		curve_normal_new[i]=curve_normal_new[i]/np.linalg.norm(curve_normal_new[i])
	return curve_normal_new

def moving_average(a, n=11, padding=False):
	#n needs to be odd for padding
	if padding:
		a=np.hstack(([np.mean(a[:int(n/2)])]*int(n/2),a,[np.mean(a[-int(n/2):])]*int(n/2)))
	ret = np.cumsum(a, axis=0)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n
	
vis_step=5
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

slice_prev=np.loadtxt('raw/slice0_0.csv',delimiter=',')
slice_normal0=np.zeros((len(slice_prev),3))
slice_normal0[:,-1]=1
num_layers=94
slice_normal=[slice_normal0]

np.savetxt('slice0_0.csv',np.hstack((slice_prev,slice_normal0)),delimiter=',')
for i in range(1,num_layers):
	###get number of disconnected sections
	num_sections=len(glob.glob('raw/slice'+str(i)+'_*.csv'))

	slicei_complete=[]
	for x in range(num_sections):

		slicei_normal=[]
		slicei=np.loadtxt('raw/slice'+str(i)+'_'+str(x)+'.csv',delimiter=',')
		###sort order
		if np.linalg.norm(slicei[0]-slice_prev[0])>10 and np.linalg.norm(slicei[-1]-slice_prev[-1])>10:
			slicei=np.flip(slicei,axis=0)
		slicei=smooth_curve(slicei)
		slicei_complete.append(slicei)
		for j in range(len(slicei)):
			#find closest 2 points
			idx_1,idx_2=np.argsort(np.linalg.norm(slicei[j]-slice_prev,axis=1))[:2]
			slicei_normal.append(find_norm(slicei[j],slice_prev[idx_1],slice_prev[idx_2]))

		slicei_normal=smooth_normal(np.array(slicei_normal))

		slice_normal.append(slicei_normal)
		ax.plot3D(slicei[::vis_step,0],slicei[::vis_step,1],slicei[::vis_step,2],'r.-')
		ax.quiver(slicei[::vis_step,0],slicei[::vis_step,1],slicei[::vis_step,2],slicei_normal[::vis_step,0],slicei_normal[::vis_step,1],slicei_normal[::vis_step,2],length=0.3, normalize=True)
		np.savetxt('slice'+str(i)+'_'+str(x)+'.csv',np.hstack((slicei,slicei_normal)),delimiter=',')


	slicei_complete=np.concatenate(slicei_complete,axis=0)
	slice_prev=slicei_complete


plt.title('0.1 blade first '+str(num_layers)+' slices')
plt.show()


