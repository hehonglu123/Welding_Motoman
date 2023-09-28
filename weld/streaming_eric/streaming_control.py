import numpy as np
import sys
sys.path.append('../')
from weld_dh2v import *

def weld_controller_lambda(delta_h,K=1,ipm_mode=140):
    
    # delta_h = min(dh_lambda,)
    v = dh2v_loglog(delta_h,mode=140)
    v = K*v
    return v