import sys
sys.path.append('../../toolbox/')
from robot_def import *
from lambda_calc import calc_lam_cs
from error_check import *
import matplotlib.pyplot as plt
import time

robot=robot_obj('MA2010_A0',def_path='../../config/MA2010_A0_robot_default_config.yml',tool_file_path='../../config/torch.csv',\
pulse2deg_file_path='../../config/MA2010_A0_pulse2deg_real.csv',d=15)

displacement=400
p_start=np.array([1648,-900,-100])
p_mid=np.array([1648-displacement/(2*np.sqrt(3)),-900+displacement/2,-100])
p_end=np.array([1648,-900+displacement,-100])

curve=np.vstack((np.linspace(p_start,p_mid,1000),np.linspace(p_mid,p_end,1000)))
curve_normal=np.array([[0,0,-1]]*2000)

vd=800
data_movel=np.loadtxt('movel_test/joint_recording_%i.csv'%vd,delimiter=',')
curve_exe_js_movel=data_movel[:,1:]
timestamp_movel=data_movel[:,0]
pose_all=robot.fwd(curve_exe_js_movel)
curve_exe_movel=pose_all.p_all
curve_exe_R_movel=pose_all.R_all
lam_movel=calc_lam_cs(curve_exe_movel)
speed_movel=np.gradient(lam_movel)/np.gradient(timestamp_movel)
error_movel,angle_error_movel=calc_all_error_w_normal(curve_exe_movel,curve[:,:3],curve_exe_R_movel[:,:,-1],curve_normal)


data_streaming=np.loadtxt('streaming_test/wofronius/joint_recording_%i.csv'%vd,delimiter=',')
curve_exe_js_streaming=data_streaming[:,1:]
timestamp_streaming=data_streaming[:,0]
pose_all=robot.fwd(curve_exe_js_streaming)
curve_exe_streaming=pose_all.p_all
curve_exe_R_streaming=pose_all.R_all
lam_streaming=calc_lam_cs(curve_exe_streaming)
speed_streaming=np.gradient(lam_streaming)/np.gradient(timestamp_streaming)
print(timestamp_streaming)
error_streaming,angle_error_streaming=calc_all_error_w_normal(curve_exe_streaming,curve[:,:3],curve_exe_R_streaming[:,:,-1],curve_normal)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(curve[:,0], curve[:,1], curve[:,2], c='gray',label='desired')
ax.plot3D(curve_exe_movel[:,0], curve_exe_movel[:,1], curve_exe_movel[:,2], c='red',label='movel')
ax.plot3D(curve_exe_streaming[:,0], curve_exe_streaming[:,1], curve_exe_streaming[:,2], c='green',label='streaming')

ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
ax.axes.set_xlim3d(left=1648-displacement/(2*np.sqrt(3)), right=1648.2) 
ax.axes.set_ylim3d(bottom=-900, top=-900+displacement) 
ax.axes.set_zlim3d(bottom=-101, top=-99) 
ax.legend()
plt.title('Streaming vs. MoveL')
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(lam_movel, speed_movel, 'r-', label='MoveL Speed')
ax2.plot(lam_movel, error_movel, 'b-',label='MoveL Error')
ax1.plot(lam_streaming, speed_streaming, 'g-', label='Streaming Speed')
ax2.plot(lam_streaming, error_streaming, 'y-',label='Streaming Error')
# ax2.plot(lam_movel, np.degrees(angle_error_movel), 'y-',label='Normal Error')
ax2.axis(ymin=0,ymax=8)
ax1.axis(ymin=0,ymax=1.5*vd)

ax1.set_xlabel('lambda (mm)')
ax1.set_ylabel('Speed/lamdot (mm/s)', color='g')
ax2.set_ylabel('Error/Normal Error (mm/deg)', color='b')
plt.title("Speed and Error Plot v=%f"%vd)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=1)
# plt.savefig('blending_zone_test/speed_plots/pl'+str(pl_all[i]))
plt.show()


