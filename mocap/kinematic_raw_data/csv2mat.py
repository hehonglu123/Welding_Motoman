import numpy
from scipy.io import savemat

all_data_dir = ['test0801_R1', 'test0804_R2']
all_filenames = ['error_T_CPA','error_T_FBF','error_T_NN','error_T_AE','error_T_NLS1', 'error_T_NLS0', 'error_T_nominal']

for data_dir in all_data_dir:
    for filename in all_filenames:
        try:
            data = numpy.loadtxt(f"{data_dir}/{filename}.csv")
        except ValueError:
            data = numpy.loadtxt(f"{data_dir}/{filename}.csv", delimiter=',')
        print(filename)
        print(data.shape)
        savemat(f"{data_dir}/{filename}.mat", {"data": data})
