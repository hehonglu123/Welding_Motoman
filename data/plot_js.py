import numpy as np
import sys, traceback, time, copy, glob
from general_robotics_toolbox import *
import matplotlib.pyplot as plt



def main():
	dataset='blade0.1/'
	sliced_alg='auto_slice/'
	data_dir='../data/'+dataset+sliced_alg
	num_layers=757
	num_baselayers=2
	curve_sliced_js=[]
	positioner_js=[]
	for i in range(num_layers):
		num_sections=len(glob.glob(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_*.csv'))
		for x in range(num_sections):
			if i % 2 == 0:
				# curve_sliced_js.append(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=','))
				positioner_js.append(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',delimiter=','))
			
			else:
				# curve_sliced_js.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/MA2010_js'+str(i)+'_'+str(x)+'.csv',delimiter=','),axis=0))
				positioner_js.append(np.flip(np.loadtxt(data_dir+'curve_sliced_js/D500B_js'+str(i)+'_'+str(x)+'.csv',delimiter=','),axis=0))

	# curve_sliced_js=np.concatenate( curve_sliced_js, axis=0)
	positioner_js=np.concatenate( positioner_js, axis=0 )

	plt.plot(positioner_js,label=('q1','q2'))
	plt.legend()
	plt.title('0.1 blade first '+str(num_layers)+' slices')
	plt.show()



if __name__ == '__main__':
	main()