import numpy as np
import glob

for fname in glob.glob('raw/*.txt'):

	f = open(fname, "r")
	curve=[]
	for line in f.readlines()[3:]:
		curve.append(list(map(float,line.split())))
	#convert to mm from inch
	curve=25.4*np.array(curve)
	filename=fname[:-4]
	np.savetxt(filename+'.csv',curve,delimiter=',')
