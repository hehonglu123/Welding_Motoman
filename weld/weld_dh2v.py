import numpy as np

def dh2v_loglog(dh,mode=140):

    logdh = np.log(dh)

    if mode==140:
        # 140 ipm
        # log(Δh)=-0.5068∗log(V)+1.643
        logv = (logdh-1.643)/(-0.5068)
    elif mode==160:
        # 160 ipm
        # log(Δh)=-0.4619∗log(V)+1.647 
        logv = (logdh-1.647)/(-0.4619)
    
    v = np.exp(logv)
    return v

def dh2v_quadratic(dh,mode=140):

    if mode==140:
        # 140 ipm
        a=0.006477
        b=-0.2362
        c=3.339-dh
    elif mode==160:
        # 160 ipm
        a=0.006043
        b=-0.2234
        c=3.335-dh
    
    v=(-b-np.sqrt(b**2-4*a*c))/(2*a)
    return v

dh=np.array([-2,-1,0,1,1.2,1.4,1.8,2])
loglog_v=dh2v_loglog(dh,160)
quad_v=dh2v_quadratic(dh,160)
print(loglog_v)
print(quad_v)