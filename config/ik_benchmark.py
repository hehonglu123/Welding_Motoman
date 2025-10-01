import numpy as np
import time
from motoman_def import *
from matplotlib import pyplot as plt

config_dir=''
robot_weld=robot_obj('MA2010_A0',def_path=config_dir+'MA2010_A0_robot_default_config.yml',d=10,tool_file_path=config_dir+'torch_robot.csv',\
    pulse2deg_file_path=config_dir+'MA2010_A0_pulse2deg_real.csv',\
    base_marker_config_file=config_dir+'MA2010_marker_config/MA2010_marker_config.yaml',tool_marker_config_file=config_dir+'weldgun_marker_config/weldgun_marker_config.yaml')

elapsed_time_log = []
for i in range(1000):
    start_time = time.perf_counter()
    # random a joint angle between joint upper and lower limit
    random_joints = np.random.uniform(robot_weld.lower_limit, robot_weld.upper_limit)
    # forward kinematics
    tool_T = robot_weld.fwd(random_joints)
    # inverse kinematics
    ik_results = robot_weld.inv(tool_T.p, tool_T.R, last_joints=random_joints)[0]
    elapsed_time_log.append(time.perf_counter()-start_time)

print("Average IK time:", np.mean(elapsed_time_log)*1000, "ms")
print("Max IK time:", np.max(elapsed_time_log)*1000, "ms")
print("Standard deviation of IK time:", np.std(elapsed_time_log)*1000, "ms")
# plot distribution of elapsed time
plt.figure()
plt.hist(np.array(elapsed_time_log)*1000, bins=50)
plt.xlabel('Elapsed time (ms)')
plt.ylabel('Frequency')
plt.title('Distribution of IK elapsed time')
plt.grid()
plt.show()