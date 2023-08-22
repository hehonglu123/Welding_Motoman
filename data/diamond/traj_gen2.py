import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../toolbox')
sys.path.append('../../slicing/')
from slicing import *
from utils import *



def parametrize_hexagon(r,point_space=1):
    ###r: radius of hexagon
    ###point_space: distance between points
    #calculate number of points
    num_points=int(6*r/point_space)
    #caculate 6 vertices
    vertices=[]
    for i in range(6):
        vertices.append(np.array([r*np.cos(i*np.pi/3),r*np.sin(i*np.pi/3)]))
    
    hex0=[]
    hex1=[]
    hex2=[]
    hex3=[]
    hex4=[]
    hex5=[]

    for i in range(num_points):
        if i*point_space>5*r:
            p=vertices[5]+(i*point_space-5*r)*(vertices[0]-vertices[5])/r
            hex5.append(p)
        elif i*point_space>4*r:
            p=vertices[4]+(i*point_space-4*r)*(vertices[5]-vertices[4])/r
            hex4.append(p)
        elif i*point_space>3*r:
            p=vertices[3]+(i*point_space-3*r)*(vertices[4]-vertices[3])/r
            hex3.append(p)
        elif i*point_space>2*r:
            p=vertices[2]+(i*point_space-2*r)*(vertices[3]-vertices[2])/r
            hex2.append(p)
        elif i*point_space>1*r:
            p=vertices[1]+(i*point_space-1*r)*(vertices[2]-vertices[1])/r
            hex1.append(p)
        else:
            p=vertices[0]+i*point_space*(vertices[1]-vertices[0])/r
            hex0.append(p)
            
    
    ###add the last point for each segment
    hex0.append(vertices[1])
    hex1.append(vertices[2])
    hex2.append(vertices[3])
    hex3.append(vertices[4])
    hex4.append(vertices[5])
    hex5.append(vertices[0])

    return [np.array(hex0),np.array(hex1),np.array(hex2),np.array(hex3),np.array(hex4),np.array(hex5)],\
            [[[-np.sqrt(3)/2,-0.5]]*len(hex0),[[0,-1]]*len(hex1),[[np.sqrt(3)/2,-0.5]]*len(hex2),[[np.sqrt(3)/2,0.5]]*len(hex3),[[0,1]]*len(hex4),[[-np.sqrt(3)/2,0.5]]*len(hex5)]


def main():
    vis_step=2
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    r_init=30
    r_middle=50
    z_middle=20
    z_end=100
    slope1=(r_middle-r_init)/z_middle
    slope2=-r_middle/(z_end-z_middle)

    angle1=np.arctan(1/slope1)
    angle2=np.arctan(-1/slope2)

    line_resolution=1.8
    z_inc1=line_resolution/slope1
    z_inc2=abs(line_resolution/slope2)

    z=0
    r=r_init
    curve_dense=[]

    ###establish baselayers first
    base_layer=[]
    r_base=np.linspace(0,r_init,int(r_init/line_resolution)+1)
    for r in r_base[1:]:
        
        hex_all,_=parametrize_hexagon(r,1)
        ###get rid of duplicates
        hex_comb=np.vstack(hex_all)
        indices=np.unique(hex_comb,axis=0,return_index=True)[1]
        hex_comb=[hex_comb[index] for index in sorted(indices)]
        base_layer_x_section=np.hstack((hex_comb,[[0,0,0,-1]]*len(hex_comb)))

        for hex in hex_all:
            if len(hex)==0:
                break
        
        ax.plot3D(base_layer_x_section[::vis_step,0],base_layer_x_section[::vis_step,1],base_layer_x_section[::vis_step,2],'r.-')
        ax.quiver(base_layer_x_section[::vis_step,0],base_layer_x_section[::vis_step,1],base_layer_x_section[::vis_step,2],base_layer_x_section[::vis_step,3],base_layer_x_section[::vis_step,4],base_layer_x_section[::vis_step,5],length=1, normalize=True)
        base_layer.append(base_layer_x_section)
    
    curve_dense.append(base_layer)

    ###other layers
    # z+=z_inc1
    # r=r_init+slope1*z
    while z<z_end:
        hex_all,hex_all_normal=parametrize_hexagon(r,1)

        # hex3d=np.hstack((hex,np.ones((len(hex),1))*z))

        if z<=z_middle:
            z+=z_inc1
            r=r_init+slope1*z
            normal_ori=1
            angle=angle1
        else:
            z+=z_inc2
            r=r_middle+slope2*(z-z_middle)
            if z-2*z_inc2<=z_middle:
                normal_ori=1
                angle=angle1
            else:
                normal_ori=-1
                angle=angle2

        if r<10:    ###tip overhang, connect all sections
            hex_comb=np.vstack(hex_all)
            indices=np.unique(hex_comb,axis=0,return_index=True)[1]
            hex_comb=[hex_comb[index] for index in sorted(indices)]
            ith_layer=[np.hstack((hex_comb,[[z,0,0,-1]]*len(hex_comb)))]

            ax.plot3D(ith_layer[0][::vis_step,0],ith_layer[0][::vis_step,1],ith_layer[0][::vis_step,2],'b.-')
            ax.quiver(ith_layer[0][::vis_step,0],ith_layer[0][::vis_step,1],ith_layer[0][::vis_step,2],ith_layer[0][::vis_step,3],ith_layer[0][::vis_step,4],ith_layer[0][::vis_step,5],length=5, normalize=True)

        else:
            ith_layer=[]
            for hex,hex_normal in zip(hex_all,hex_all_normal):
                if len(hex)==0:
                    break
                hex_section=np.hstack((hex,np.ones((len(hex),1))*z))
                #get normal:
                # normal=-get_curve_normal_from_curves(hex_section,np.vstack(curve_dense[-1])[:,:3])
                # normal[:]=np.average(normal[~np.isnan(normal[:,0])],axis=0)
                normal_temp=np.array([hex_normal[0][0],hex_normal[0][1],0])
                normal=rotate_vector_at_angle(normal_ori*normal_temp,np.array([0,0,-1]),angle)
                
                hex_section=np.hstack((hex_section,[normal]*len(hex_section)))
                ax.plot3D(hex_section[::vis_step,0],hex_section[::vis_step,1],hex_section[::vis_step,2],'b.-')
                ax.quiver(hex_section[::vis_step,0],hex_section[::vis_step,1],hex_section[::vis_step,2],hex_section[::vis_step,3],hex_section[::vis_step,4],hex_section[::vis_step,5],length=5, normalize=True)

                ith_layer.extend(hex_section)
            
            ith_layer=np.vstack(ith_layer)
            ###get rid of duplicates
            temp=np.delete(ith_layer[1:],np.diff(calc_lam_cs(ith_layer[:,:3]))==0,axis=0)
            ith_layer=[np.vstack((ith_layer[0],temp))]
        
        curve_dense.append(ith_layer)
    plt.show()

    
    for i in range(len(curve_dense)):
        for x in range(len(curve_dense[i])):
            np.savetxt('cont_sections/curve_sliced/slice%i_%i.csv'%(i,x),curve_dense[i][x],delimiter=',')

if __name__ == "__main__":
    main()