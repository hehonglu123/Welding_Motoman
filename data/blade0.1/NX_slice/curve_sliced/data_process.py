import numpy as np

def find_norm(p1,p2,p3):
	#find normal vector from p1 pointing to line of p2p3
	
slice1=np.readtxt('raw/slice1.csv',delimiter=',')
slice2=np.readtxt('raw/slice2.csv',delimiter=',')
slice3=np.readtxt('raw/slice3.csv',delimiter=',')
slice4=np.readtxt('raw/slice4.csv',delimiter=',')

slice1_normal=[]
for i in range(len(slice1)):
	#find closest 2 points
	idx_1,idx_2=np.argsort(np.linalg.norm(slice2-slice1[i],axis=1))[:2]
	norm=