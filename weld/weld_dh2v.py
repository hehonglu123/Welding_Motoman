import numpy as np

material_param = {}
material_param['ER_4043'] = {
    #ER 4043
    "100ipm": [-0.62015, 1.84913],
    "110ipm": [-0.43334211,  1.23605507],
    "120ipm": [-0.4275129,   1.25877655],
    "130ipm": [-0.42601101,  1.29815246],
    "140ipm": [-0.5068, 1.643],
    "150ipm": [-0.44358649,  1.37366127],
    "160ipm": [-0.4619, 1.647],
    "170ipm": [-0.44802126,  1.40999683],
    "180ipm": [-0.3713, 1.506],
    "190ipm": [-0.5784, 1.999],
    "200ipm": [-0.5776, 2.007],
    "210ipm": [-0.5702, 1.990],
    "220ipm": [-0.5699, 1.985],
    "230ipm": [-0.5374, 1.848],
    "240ipm": [-0.46212367,  1.14990345],
}
material_param['ER_70S6'] = {
        #ER 70S-6
        "100ipm": [-0.31828998,  0.68503243],
        "110ipm": [-0.27499103,  0.63182495],
        "120ipm": [-0.31950134,  0.69261567],
        "130ipm": [-0.31630631,  0.70834374],
        "140ipm": [-0.31654673,  0.74122273],
        "150ipm": [-0.31903634,  0.78416199],
        "160ipm": [-0.31421562,  0.81825734],
        "170ipm": [-0.22845064,  0.77327933],
        "180ipm": [-0.20186512,  0.78154183],
        "190ipm": [-0.2810107 ,  0.88758474],
        "200ipm": [-0.33326134,  0.95033665],
        "210ipm": [-0.31767768,  0.94708744],
        "220ipm": [-0.32122332,  0.97605575],
        "230ipm": [-0.29245818,  0.99450003],
        "240ipm": [-0.31196034,  1.03601865],
        "250ipm": [-0.27141449,  1.03156706]
    }
material_param['316L'] = {   
        #316L
        "100ipm": [-0.27943327,  0.82598745],
        "110ipm": [-0.27403046,  0.85567816],
        "120ipm": [-0.19099690,  0.80205849],
        "130ipm": [-0.33658666,  0.93355497],
        "140ipm": [-0.35155543,  1.04383881],
        "150ipm": [-0.36475561,  1.06087784],
        "160ipm": [-0.31898775,  1.02420909],
        "170ipm": [-0.31235952,  1.04687969],
        "180ipm": [-0.30345950,  1.05420703],
        "190ipm": [-0.33028289,  1.09923509],
        "200ipm": [-0.37109996,  1.16727314],
        "210ipm": [-0.36137087,  1.19725649],
        "220ipm": [-0.32864204,  1.16586553],
        "230ipm": [-0.35960441,  1.16413227],
        "240ipm": [-0.43279062,  1.26611887],
        "250ipm": [-0.36653101,  1.21254436]
    }

def v2dh_loglog(v,mode=140,material='ER_4043'):
    
    mode=int(mode)
    param = material_param[material][str(mode)+'ipm']
    logdh = param[0]*np.log(v)+param[1]
    
    dh = np.exp(logdh)
    return dh

def dh2v_loglog(dh,mode=140,material='ER_4043'):
    
    mode=int(mode)
    param = material_param[material][str(mode)+'ipm']

    logdh = np.log(dh)
    logv = (logdh-param[1])/param[0]
    
    # v = np.exp(logv)
    v=np.float_power((dh/np.exp(param[1])),1/param[0])
    
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
    # dh=np.array([-2,-1,0,1,1.2,1.4,1.8,2])
    # loglog_v=dh2v_loglog(dh,160)
    # quad_v=dh2v_quadratic(dh,160)
    # print(loglog_v)
    # print(quad_v)

    # print(v2dh_loglog(18,160))
    # print(dh2v_loglog(1.5,160))
    # print(v2dh_loglog(75,220))

    # print(v2dh_loglog(5,100))
    # print(v2dh_loglog(16,160))
    # print(v2dh_loglog(10,180))
    # print(v2dh_loglog(9.411764,160))
    # print(v2dh_loglog(8,210))
    # print(dh2v_loglog(2.5,210))
    # print(dh2v_loglog(3.5,210))
    # print(v2dh_loglog(6,100))
    # print(v2dh_loglog(10,100))
    # print(dh2v_loglog(5,100))
    # print(v2dh_quadratic(5,100))

    print("ipm=150, dh=1.5, v=",dh2v_loglog(1.5,150))

    # v_list = np.arange(5,12,0.2)
    # ipm_list = [100,110,120,130,140,150,160,170,180]
    
    # for ipm in ipm_list:
    #     for v in v_list:
    #         print(ipm,round(v,1),v2dh_loglog(v,ipm))