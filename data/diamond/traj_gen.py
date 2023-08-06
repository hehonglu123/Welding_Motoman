import numpy as np
import matplotlib.pyplot as plt

def parametrize_hexagon(r,point_space=1):
    ###r: radius of hexagon
    ###point_space: distance between points
    #calculate number of points
    num_points=int(6*r/point_space)
    #caculate 6 vertices
    vertices=[]
    for i in range(6):
        vertices.append(np.array([r*np.cos(i*np.pi/3),r*np.sin(i*np.pi/3)]))
    
    hex=[]
    for i in range(num_points):
        if i*point_space>5*r:
            p=vertices[5]+(i*point_space-5*r)*(vertices[0]-vertices[5])/r
        elif i*point_space>4*r:
            p=vertices[4]+(i*point_space-4*r)*(vertices[5]-vertices[4])/r
        elif i*point_space>3*r:
            p=vertices[3]+(i*point_space-3*r)*(vertices[4]-vertices[3])/r
        elif i*point_space>2*r:
            p=vertices[2]+(i*point_space-2*r)*(vertices[3]-vertices[2])/r
        elif i*point_space>1*r:
            p=vertices[1]+(i*point_space-1*r)*(vertices[2]-vertices[1])/r
        else:
            p=vertices[0]+i*point_space*(vertices[1]-vertices[0])/r
        
        hex.append(p)

    return np.array(hex)


def main():
    vis_step=2
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    r_init=30
    r_middle=50
    z_middle=20
    z_end=100
    slope1=(r_middle-r_init)/z_middle
    slope2=-r_middle/(z_end-z_middle)

    line_resolution=1
    z_inc1=line_resolution/slope1
    z_inc2=abs(line_resolution/slope2)

    z=0
    r=r_init
    curve_dense=[]
    while z<z_end:
        hex=parametrize_hexagon(r,1)
        if len(hex)==0:
            break
        curve_dense.append(np.hstack((hex,np.ones((len(hex),1))*z)))
        if z<z_middle:
            z+=z_inc1
            r=r_init+slope1*z
        else:
            z+=z_inc2
            r=r_middle+slope2*(z-z_middle)




        ax.plot3D(hex[::vis_step,0],hex[::vis_step,1],z,'b.-')

    plt.show()

if __name__ == "__main__":
    main()