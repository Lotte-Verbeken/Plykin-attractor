import numpy as np

def get_angular_coordinates(x_vals#list
                            ,y_vals#list
                            ,z_vals#list
                            ):
    # transformation from cartisian to angular coordinates

    phi_val=[]
    theta_val=[]
    for k in range(len(x_vals)):
        theta = 0
        if z_vals[k] > 0:
            theta = np.atan(np.sqrt(y_vals[k]**2+x_vals[k]**2)/z_vals[k])
        elif z_vals[k] < 0:
            theta = np.pi + np.atan(np.sqrt(y_vals[k]**2+x_vals[k]**2)/z_vals[k])
        elif z_vals[k] == 0 and np.sqrt(y_vals[k]**2+x_vals[k]**2) != 0:
            theta = np.pi/2
        else:
            print("theta undefined")


        phi = 0
        if x_vals[k] > 0:
            phi = np.atan(y_vals[k]/x_vals[k])
        elif x_vals[k]<0 and y_vals[k]>=0:
            phi = np.atan(y_vals[k]/x_vals[k]) + np.pi
        elif x_vals[k]<0 and y_vals[k]<0:
            phi = np.atan(y_vals[k] / x_vals[k]) - np.pi
        elif x_vals[k]==0 and y_vals[k]>0:
            phi = np.pi/2
        elif x_vals[k] ==0 and y_vals[k]<0:
            phi = -np.pi/2
        else:
            print("phi undefined")

        phi = phi+np.pi #since we want positive values we move all values of phi with pi

        theta_val.append(theta)
        phi_val.append(phi)

    return phi_val, theta_val

def get_complex_coordinates(x,#list
                            y,#list
                            z#list
                            ):
    #transforms from cartesian to complex coordinates
    real_val=[]
    imag_val=[]
    for k in range(len(x)):
        W = (x[k]-z[k]+1j * y[k]*np.sqrt(2))/(x[k]+z[k]+np.sqrt(2))
        real_val.append(np.real(W))
        imag_val.append(np.imag(W))
    return real_val, imag_val


