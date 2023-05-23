import sys, glob
sys.path.append('../toolbox/')
from robot_def import *

data_dir='../data/spiral_cylinder/'
solution_dir='baseline/'
robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/torch.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[-0.7071, 0.7071, -0.    ],
            [ 0.7071, 0.7071,  0.    ],
            [0.,      0.,     -1.    ]])

q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])
for file in glob.glob(data_dir+solution_dir+'curve_sliced_in_base_frame/*.csv'):
	curve=np.loadtxt(file,delimiter=',')
	curve_js=car2js(robot,q_seed,curve[:,:3],[R]*len(curve))
	q_seed=curve_js[-1]

	filename=file.split('\\')[-1]
	np.savetxt(data_dir+solution_dir+'curve_sliced_js/MA2010_js'+filename,curve_js,delimiter=',')