### Draws the magnetic field lines of two long wires using the constant-value contours of the flux coordinate surface
### The coordinate system is situated with the two long parallel wires extenting into the z-direction, this program then 
### maps the magnetic field of the wires onto the x-y plane. In the definition of Afield(x,y), only one return operation 
### can be used at a time. The difference between the two return values is just the directions of the currents in the two 
### wires, parallel or anti-parallel. 

import numpy as np
import matplotlib.pyplot as plt 
import random


size = 251
dx, dy = 0.04, 0.04

surface_values = [[0. for i in range(size)] for j in range(size)]

def Afield(x,y):
    #[1]# [opposite current directions]:
    #return np.log((np.sqrt(10**2 + np.sqrt((x+1.3)**2 + y**2)**2) + 10)/(np.sqrt((x+1.3)**2 + y**2))) - np.log((np.sqrt(10**2 + np.sqrt((x-1.3)**2 + y**2)**2) + 10)/(np.sqrt((x-1.3)**2 + y**2)))
    #[2]# [identical current directions]:
    return np.log((np.sqrt(10**2 + np.sqrt((x+1.3)**2 + y**2)**2) + 10)/(np.sqrt((x+1.3)**2 + y**2))) + np.log((np.sqrt(10**2 + np.sqrt((x-1.3)**2 + y**2)**2) + 10)/(np.sqrt((x-1.3)**2 + y**2)))

for r in range(size):
    for c in range(size):
        x = c*dx - ((size-1)/(2))*dx 
        y = r*dy - ((size-1)/(2))*dy
        value = Afield(x,y)
        surface_values[r][c] = value
        
### The contours of the flux coordinate surface, in this case A_z (z-component of the magnetic vector potential), are projected onto the x-y plane producing the magnetic field lines. 
levels = 25
plt.contour(surface_values,levels)

plt.axes().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
