import numpy as np

def v2dh_loglog(v,mode=140):

    if mode==140:
        # 140 ipm
        # log(Δh)=-0.5068*log(V)+1.643
        logdh = -0.5068*np.log(v)+1.643
    elif mode==160:
        # 160 ipm
        # log(Δh)=-0.4619*log(V)+1.647 
        logdh = -0.4619*np.log(v)+1.647 
    
    dh = np.exp(logdh)
    return dh

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

if __name__=='__main__':
    dh=np.array([-2,-1,0,1,1.2,1.4,1.8,2])
    loglog_v=dh2v_loglog(dh,160)
    quad_v=dh2v_quadratic(dh,160)
    print(loglog_v)
    print(quad_v)

    print(v2dh_loglog(17.,160))
    print(dh2v_loglog(1.4,160))
    print(dh2v_loglog(1,160))