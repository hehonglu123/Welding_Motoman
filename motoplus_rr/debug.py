import sys, glob
from RobotRaconteur.Client import *
from scipy.interpolate import interp1d
from StreamingSend import *



########################################################RR STREAMING########################################################

# RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.1.114:59945?service=robot')
RR_robot_sub = RRN.SubscribeService('rr+tcp://192.168.55.12:59945?service=robot')


SS=StreamingSend(RR_robot_sub)


SS.start_recording()
###########################################base layer welding############################################
while True:
    try:
        SS.jog2q(np.hstack((np.zeros(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
        SS.jog2q(np.hstack((-0.5*np.ones(6),[np.pi/2,0,0,0,0,0,np.radians(-15),np.pi])))
    except:
        js_recording = SS.stop_recording()
        break

np.savetxt('joint_recording_streaming.csv',js_recording,delimiter=',')