import numpy as np
import matplotlib.pyplot as plt
import time
from  Utils import get_angular_coordinates,get_complex_coordinates

#defold
mu=1
eps=1
iterations = 100000

#initial value
theta_initial = 5*np.pi /4
phi_initial = np.pi /3
x = np.cos(phi_initial) * np.sin(theta_initial)
y = np.sin(phi_initial)*np.sin(theta_initial)
z = np.cos(theta_initial)

def getting_points(x,y,z):
    # this is the Poincaré half-period map
    # 1 period would be x_n+1 = f_1_1(f_-1_-1(x_n)) with f_sigma_s(x)
    # getting_point gives all the points so x_n+1/2 and x_n+1

    x_vals = []
    y_vals = []
    z_vals = []

    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

    x_old = x
    y_old = y
    z_old = z

    for k in range(1,iterations):
        if k%2==1: #If k is odd, 1st 3 transformations
            s  = -1
            sigma = -1
        else:#last 3 transformations
            s = 1
            sigma = 1

        P = ((1 - np.exp(-mu)) * np.sqrt(x_old ** 2 + y_old ** 2 + z_old ** 2) + np.exp(-mu)) ** (-1)
        Q = np.sqrt(x_old**2+y_old**2)/np.sqrt(x_old**2*np.exp(-eps*(x_old**2+y_old**2))+y_old**2*np.exp(eps*(x_old**2+y_old**2)))

        x_new = P*s*z_old
        y_new = P*Q*(sigma*x_old*np.exp(-eps*(x_old**2+y_old**2)/2)*np.sin(np.pi/2 * (z_old*np.sqrt(2)+1))+y_old*np.exp(eps*(x_old**2+y_old**2)/2)*np.cos(np.pi/2 * (z_old*np.sqrt(2)+1)))
        z_new = P*Q*(-s*x_old*np.exp(-eps*(x_old**2+y_old**2)/2)*np.cos(np.pi/2 * (z_old*np.sqrt(2)+1))+s*sigma*y_old*np.exp(eps*(x_old**2+y_old**2)/2)*np.sin(np.pi/2 * (z_old*np.sqrt(2)+1)))

        x_vals.append(x_new)
        y_vals.append(y_new)
        z_vals.append(z_new)

        x_old = x_new
        y_old = y_new
        z_old = z_new
    return x_vals, y_vals, z_vals


# we are only interested in the x_n with natural index thus the x_n+k with k natural number
def get_usefull_points(x_vals,#list
                       y_vals,#list
                       z_vals#list
                       ):
    x_full=[]
    y_full=[]
    z_full=[]
    for k in range(len(x_vals)):
        if k%2==0:
            x_full.append(x_vals[k])
            y_full.append(y_vals[k])
            z_full.append(z_vals[k])
        else:
            continue
    return x_full, y_full, z_full


#different coordinates, solutions of functions

start_cartesian = time.time()
x_val = get_usefull_points(getting_points(x,y,z)[0],getting_points(x,y,z)[1],getting_points(x,y,z)[2])[0]
y_val = get_usefull_points(getting_points(x,y,z)[0],getting_points(x,y,z)[1],getting_points(x,y,z)[2])[1]
z_val = get_usefull_points(getting_points(x,y,z)[0],getting_points(x,y,z)[1],getting_points(x,y,z)[2])[2]
end_cartesian = time.time()

start_angular = time.time()
phi = get_angular_coordinates(x_val, y_val, z_val)[0]
theta = get_angular_coordinates(x_val, y_val, z_val)[1]
end_angular = time.time()

start_complex = time.time()
real = get_complex_coordinates(x_val, y_val, z_val)[0]
imag = get_complex_coordinates(x_val, y_val, z_val)[1]
end_complex = time.time()

#timing code
elapsed_cartesian = end_cartesian - start_cartesian
elapsed_angular = end_angular - start_angular
elapsed_complex = end_complex - start_complex
#elapsed_steriographic = end_steriographic -start_steriographic

print(f'Time taken for calculating cartesian coordinates : {elapsed_cartesian:.6f} seconds')
print(f'Time taken for calculating angular coordinates : {elapsed_angular:.6f} seconds')
print(f'Time taken for calculating complex coordinates : {elapsed_complex:.6f} seconds')
#print(f'Time taken for calculating steriographic coordinates : {elapsed_steriographic:.6f} seconds')

#plotting figures
a = (1/np.sqrt(2), 0 , 1/np.sqrt(2))
b = (1/np.sqrt(2), 0 , -1/np.sqrt(2))
c = (-1/np.sqrt(2), 0 , -1/np.sqrt(2))
d = (-1/np.sqrt(2), 0 , 1/np.sqrt(2))

xcod = np.array([a[0],b[0],c[0],d[0]])
ycod = np.array([a[1],b[1],c[1],d[1]])
zcod = np.array([a[2],b[2],c[2],d[2]])
l = ['A', 'B', 'C','D']


fig1 = plt.figure(1)
ax = plt.axes(projection='3d')

# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)
xline = np.linspace(0, 15, 1000)
yline = np.linspace(0, 15, 1000)

ax.scatter3D(x_val, y_val, z_val,  c=z_val, cmap='viridis', linewidth=0.05)
#name four points
for x,y,z,k in zip(xcod,ycod,zcod,range(len(l))):
    ax.text(x,y,z,l[k])
ax.set_xlabel('X',fontsize=15)
ax.set_ylabel('Y',fontsize=15)
ax.set_zlabel('Z',fontsize=15)



fig2 = plt.figure(2)
plt.axis([0, 2 * np.pi, 0, np.pi])
plt.plot(phi, theta, '.', markersize=0.7)
plt.xlabel(r'${\varphi}$',fontsize=15)
plt.ylabel(r'${\theta}$',fontsize=15)



def mark_points(a,b,d):
    real_mark=[]
    imag_mark=[]
    Wa = (a[0] - a[2] + 1j * a[1] * np.sqrt(2)) / (a[0] + a[2] + np.sqrt(2))
    Wb = (b[0] - b[2] + 1j * b[1] * np.sqrt(2)) / (b[0] + b[2] + np.sqrt(2))
    Wd = (d[0] - d[2] + 1j * d[1] * np.sqrt(2)) / (d[0] + d[2] + np.sqrt(2))
    real_mark.append(np.real(Wa))
    imag_mark.append(np.imag(Wa))
    real_mark.append(np.real(Wb))
    imag_mark.append(np.imag(Wb))
    real_mark.append(np.real(Wd))
    imag_mark.append(np.imag(Wd))
    return real_mark, imag_mark

M = mark_points(a, b, d)[0]
N = mark_points(a, b, d)[1]
L = ['A', 'B','D']

fig3 = plt.figure(3)
plt.axis([-4, 4, -4, 3])
plt.plot(real, imag, '.', markersize=1)
#name three points
for k in range(len(L)):
    plt.text(M[k], N[k], L[k])
plt.xlabel(r'$\Re(W)$',fontsize=15)
plt.ylabel(r'$\Im(W)$',fontsize=15)


plt.show()
