import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Python 3.~~
# Last Updated: 2019/05/03
# Author: Tyler E. Bagwell

# Purpose: This simple program solves and plots the solution to Laplace's equation 
# in polar coordinates directly by using Numpy's matrix inversion method -- linalg.solve(). 
 
# Solution: This program specfically solves the problem of finding the steady-state 
# temperature distribution in a circular plate where the upper half is held at 100 and the lovwer
# half is held at 0.

# --- Gridding
N = 30  # Number of grid points along r 
Q = 60  # Number of grid points dividing 2pi radians (Should be divisible by 4)

rmin, rmax = 0.0, 1.0 

drho = (rmax-rmin)/(N-1)
dphi = 2.0*np.pi/Q

# --- Direct Solution of the Matrix Equation: Au=0 

ele = (N-1)*Q+1     # Number of rows and columns including origin point
A_mat = np.array([[0. for i in range(ele)] for j in range(ele)])

var_vector = []
b_vector = np.array([0. for i in range(ele)])


# coef A
def loc_coef_A(row,x_1):
    col = row
    A_mat[row][col] = -2.*(dphi**2 + (1./x_1**2))

# coef B
def loc_coef_B(row,x_1):
    if x_1 < N-1:
        col = row + Q
        A_mat[row][col] = dphi**2*(1. + (1./(2.*x_1)))

# coef C
def loc_coef_C(row,x_1):
    if x_1 == 1:
        col = 0
        A_mat[row][col] = dphi**2*(1. - (1./(2.*x_1)))
    else:
        col = row - Q
        A_mat[row][col] = dphi**2*(1. - (1./(2.*x_1)))
        
# coef D
def loc_coef_D(row,x_1):
    if x_2 == Q-1:
        col = row - (Q-1)
        A_mat[row][col] = 1./(x_1**2)
    else:
        col = row+1
        A_mat[row][col] = 1./(x_1**2)
        
# coef E
def loc_coef_E(row,x_1):
    if x_2 == 0:
        col = row + (Q-1)
        A_mat[row][col] = 1./(x_1**2)
    else:
        col = row-1
        A_mat[row][col] = 1./(x_1**2)
        
# Build the coefficient matrix        
A_mat[0][0] = -1.0 
A_mat[0][1] = 1.0/4.0
A_mat[0][1+int(Q/2)] = 1.0/4.0
A_mat[0][1+int(Q/4)] = 1.0/4.0
A_mat[0][1+int(3*Q/4)] = 1.0/4.0
    

row = 1
for x_1 in range(1,N):
    for x_2 in range(Q):
        loc_coef_A(row,x_1)
        loc_coef_B(row,x_1)
        loc_coef_C(row,x_1)
        loc_coef_D(row,x_1)
        loc_coef_E(row,x_1)
        row += 1

# --- Apply boundary conditions:

# Problem 13.5.12 ----
i = 1
for row in range(ele-Q, ele):
    if i <= Q/2:
        b_vector[row] = 100.
    else: 
        b_vector[row] = 0.
    i += 1
    for col in range(ele):
        if col == row:
            A_mat[row][col] = 1.
        else:
            A_mat[row][col] = 0.

## Problem 13.5.14 ----
#i = 1
#for row in range(ele-Q, ele):
#    if i <= Q/2:
#        b_vector[row] = 100.
#    else: 
#        b_vector[row] = 0.
#    i += 1
#    for col in range(ele):
#        if col == row:
#            A_mat[row][col] = 1.
#        else:
#            A_mat[row][col] = 0.
#
#mid = int(((N-1)/2) + 1)
#for row in range(ele-mid*Q, ele-mid*Q+Q):
#    b_vector[row] = 0.
#    for col in range(ele):
#        if col == row:
#            A_mat[row][col] = 1.
#        else:
#            A_mat[row][col] = 0.
    
# --- Directly solve for the variables by solving the inverse matrix equation. 

var_vector = np.linalg.solve(A_mat,b_vector)

# --- Plotting:

rho = np.linspace(rmin,rmax,N)
phi = np.linspace(0,2.*np.pi,Q+1)

phi_m, rho_m = np.meshgrid(phi,rho)

phi_m[0][:] = 0.

print('---------------')

X = np.array([[0. for j in range(Q+1)] for i in range(N)])
Y = np.array([[0. for j in range(Q+1)] for i in range(N)])
for i in range(N):
    for j in range(Q+1):
        X[i][j] = rho_m[i][j]*np.cos(phi_m[i][j])
        Y[i][j] = rho_m[i][j]*np.sin(phi_m[i][j])

Z = np.array([[0. for i in range(Q+1)] for j in range(N)])

Z[0][:] = var_vector[0]

for i in range(N-1):
    for j in range(Q):
        if j == 0:
            Z[i+1][j] = var_vector[i*Q+j+1]
            Z[i+1][Q] = var_vector[i*Q+j+1]
        else:
            Z[i+1][j] = var_vector[i*Q+j+1]

### SURFACE PLOT --------------------
#fig = plt.figure(figsize=plt.figaspect(0.6)*1.25)
#ax = fig.add_subplot(111, projection='3d')
#
#ax.plot_surface(X, Y, Z, cmap='plasma')
#
#plt.title('Steady-State Temperature')
#ax.set_xlabel(r'$x$')
#ax.set_ylabel(r'$y$')
#ax.set_zlabel(r'$T(x,y)$')
#plt.show()
## ----------------------------------


## DENSITY PLOT ---------------------
plt.pcolormesh(X, Y, Z, cmap='plasma')

plt.colorbar(extend='both')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.axis('on')

plt.title(r'Steady-State Temperature $T(x,y)$')
plt.axes().set_aspect('equal')
plt.show()
## ----------------------------------
