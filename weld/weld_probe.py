import sys
sys.path.append('../toolbox/')
from robot_def import *
from dx200_motion_program_exec_client import *
from WeldSend import *
from lambda_calc import *

robot=robot_obj('MA2010_A0',def_path='../config/MA2010_A0_robot_default_config.yml',tool_file_path='../config/weldgun.csv',\
	pulse2deg_file_path='../config/MA2010_A0_pulse2deg_real.csv',d=15)

R=np.array([[ 0.7071, -0.7071, -0.    ],
			[-0.7071, -0.7071,  0.    ],
			[-0.,      0.,     -1.    ]])
initial_height=-250
p_start=np.array([1650,-850,-250])
p_end=np.array([1650,-780,-250])
q_seed=np.radians([-35.4291,56.6333,40.5194,4.5177,-52.2505,-11.6546])

client=MotionProgramExecClient()
ws=WeldSend(client)


height_profile=[]
layer_height=0
nominal_layer_height=10
num_baselayer=2
for i in range(num_baselayer):
    if i % 2 ==1:
        q_init=np.degrees(robot.inv(p_end+np.array([0,0,layer_height]),R,q_seed)[0])
        q_end=np.degrees(robot.inv(p_start+np.array([0,0,layer_height]),R,q_seed)[0])
    else:
        q_init=np.degrees(robot.inv(p_start+np.array([0,0,layer_height]),R,q_seed)[0])
        q_end=np.degrees(robot.inv(p_end+np.array([0,0,layer_height]),R,q_seed)[0])

    p_all=np.linspace(p_start+np.array([0,0,layer_height+nominal_layer_height]),p_end+np.array([0,0,layer_height+nominal_layer_height]),num=7)

    ws.weld_segment(robot,q_init,q_end,speed=5,cond_num=200,arc=True)
    
    ws.wire_cut(robot,speed=5)
    

    q_all=ws.touchsense(robot,p_all+np.array([0,0,20]),p_all-np.array([0,0,20]),R)
    p_act=robot.fwd(q_all).p_all
    height_profile.append(p_act[:,-1])
    layer_height=np.average(p_act[:,-1])-initial_height

    lam=calc_lam_cs(p_all)
    plt.plot(lam,p_act[:,-1])
    

plt.show()