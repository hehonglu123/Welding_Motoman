import numpy as np

def v2dh_loglog(v,mode=140):

    if mode==140:
        # 140 ipm
        # log(Δh)=-0.5068*log(V)+1.643
        logdh = -0.5068*np.log(v)+1.643
    elif mode==100:
        # 100 ipm
        # log(Δh)=-0.6201508222063208∗log(V)+1.8491301017700346
        logdh = -0.6201508222063208*np.log(v)+1.8491301017700346
    elif mode==160:
        # 160 ipm
        # log(Δh)=-0.4619*log(V)+1.647 
        logdh = -0.4619*np.log(v)+1.647 
    elif mode==180:
        # 180 ipm
        # log Δℎ = −0.3713 ∗ log 𝑉 + 1.506 
        logdh = -0.3713*np.log(v)+1.506 
    elif mode==220:
        # 220 ipm
        # log(Δh)=-0.5699*log(V)+1.985 
        logdh = -0.5699*np.log(v)+1.985
    
    dh = np.exp(logdh)
    return dh

def dh2v_loglog(dh,mode=140):

    logdh = np.log(dh)

    if mode==140:
        # 140 ipm
        # log(Δh)=-0.5068∗log(V)+1.643
        logv = (logdh-1.643)/(-0.5068)
    elif mode==100:
        # 100 ipm
        # log(Δh)=-0.6201508222063208∗log(V)+1.8491301017700346
        logv = (logdh-1.8491301017700346)/(-0.6201508222063208)
    elif mode==160:
        # 160 ipm
        # log(Δh)=-0.4619∗log(V)+1.647 
        logv = (logdh-1.647)/(-0.4619)
    elif mode==180:
        # 180 ipm
        # log Δℎ = −0.3713 ∗ log 𝑉 + 1.506 
        logv = (logdh-1.506)/(-0.3713)
    elif mode==220:
        # 220 ipm
        logv = (logdh-1.985)/(-0.5699)
    
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

def v2dh_quadratic(v,mode=140):

    if mode==140:
        # 140 ipm
        a=0.006477
        b=-0.2362
        c=3.339
    elif mode==100:
        # 100ipm
        a=0.01824187
        b=-0.58723623
        c=5.68282353
    elif mode==160:
        # 160 ipm
        a=0.006043
        b=-0.2234
        c=3.335
    
    dh = a*(v**2)+b*v+c
    return dh

if __name__=='__main__':
    dh=np.array([-2,-1,0,1,1.2,1.4,1.8,2])
    loglog_v=dh2v_loglog(dh,160)
    quad_v=dh2v_quadratic(dh,160)
    # print(loglog_v)
    # print(quad_v)

    # print(v2dh_loglog(18,160))
    # print(dh2v_loglog(1.5,160))
    # print(v2dh_loglog(75,220))

    # print(v2dh_loglog(5,100))
    print(v2dh_loglog(16,160))
    print(v2dh_loglog(10,180))
    print(v2dh_loglog(9.411764,160))
    # print(v2dh_loglog(6,100))
    # print(v2dh_loglog(10,100))
    # print(dh2v_loglog(5,100))
    # print(v2dh_quadratic(5,100))