import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Python 3.~~
# Last Updated: 2019/05/05
# Author: Tyler E. Bagwell

# Purpose: This simple program solves and animates Poisson's equation in polar 
# coordinates implicitly by implementing the Crank-Nicolson finite difference method
# and using Numpy's matrix inversion method -- linalg.solve() -- to solve the matrix 
# equations at each time step. 

# Normalized Equation:
# dtU = drrU + drU/r + dppU/r^2 = ∇^2U
# where: dt = d/dt, dr = d/dr, drr = d^2/dr^2, dpp = d^2/dphi^2
# Solves for: U = U(r,phi,t) = U(p*drho,q*dphi,j*dt)

# Solution: This program specfically solves the evolution of the temperature within a 
# circular ring where the temperature along the outer and inner ring circumfrences can
# be adjusted.

print('-----------------')

### ----- Parameters and Inputs ----- ###

N = 61  # Number of grid points along r (Should be odd)
Q = 100  # Number of grid points dividing 2pi radians (Should be divisible by 4)
N_half = int(((N-1)/2)+1)

rmin, rmax = 0.0, 1.0 

drho = (rmax-rmin)/(N-1)
dphi = 2.0*np.pi/Q

dt = (drho/1.)**2

temp_max = 1.
temp_min = 0. 


### ----- Initialize ----- ###

# Matrix equation to solve at each time step for s_tnext:  dot(A_mat,s_tnext) = dot(B_mat,s_t),
# where:
#       s_t:        state vector at t, contains all U_t
#       s_tnext:    state vector at t+dt, contains all U_tnext
#       A_mat:      coefficient matrix for forward time values of U held in s_tnext
#       B_mat:      coefficient matrix for current time values of U held in s_t


ele = int((N_half)*Q)       # Number of rows and columns 

s_t = np.array([0. for i in range(ele)])        # State vector at t
s_tnext = np.array([0. for i in range(ele)])    # State vector at t+dt

A_mat = np.array([[0. for i in range(ele)] for j in range(ele)])
B_mat = np.array([[0. for i in range(ele)] for j in range(ele)])


# coef A for U_{p,j,t+1} & U_{p,j,t}
def loc_coef_A(row,x_1):
    col = row
    p = x_1+N_half
    A_mat[row][col] = 4*(1 + (dt/drho**2) + (dt/(p*drho*dphi)**2))
    B_mat[row][col] = 4*(1 - (dt/drho**2) - (dt/(p*drho*dphi)**2))  

# coef B for U_{p+1,j,t+1} & U_{p+1,j,t}
def loc_coef_B(row,x_1):
    p = x_1+N_half
    if x_1 < N_half-1:
        col = row + Q
        A_mat[row][col] = -(dt/drho**2)*(2+(1/p))
        B_mat[row][col] =  (dt/drho**2)*(2+(1/p))

# coef C for U_{p-1,j,t+1} & U_{p-1,j,t}
def loc_coef_C(row,x_1):
    p = x_1+N_half
    if x_1 == 0:
        pass
    else:
        col = row - Q
        A_mat[row][col] =  (dt/drho**2)*((1/p)-2)
        B_mat[row][col] = -(dt/drho**2)*((1/p)-2)
        
# coef D for U_{p,j+1,t+1} and U_{p,j+1,t}
def loc_coef_D(row,x_1):
    p = x_1+N_half
    if x_2 == Q-1:
        col = row - (Q-1)
        A_mat[row][col] = -(2*dt)/((p*drho*dphi)**2)
        B_mat[row][col] =  (2*dt)/((p*drho*dphi)**2)
    else:
        col = row+1
        A_mat[row][col] = -(2*dt)/((p*drho*dphi)**2)
        B_mat[row][col] =  (2*dt)/((p*drho*dphi)**2)
        
# coef E for U_{p,j-1,t+1} and U_{p,j-1,t}
def loc_coef_E(row,x_1):
    p = x_1+N_half
    if x_2 == 0:
        col = row + (Q-1)
        A_mat[row][col] = -(2*dt)/((p*drho*dphi)**2)
        B_mat[row][col] =  (2*dt)/((p*drho*dphi)**2)
    else:
        col = row-1
        A_mat[row][col] = -(2*dt)/((p*drho*dphi)**2)
        B_mat[row][col] =  (2*dt)/((p*drho*dphi)**2)
        
# Initialize the matrices 
row = 0
for x_1 in range(N_half):
    for x_2 in range(Q):
        loc_coef_A(row,x_1)
        loc_coef_B(row,x_1)
        loc_coef_C(row,x_1)
        loc_coef_D(row,x_1)
        loc_coef_E(row,x_1)
        row += 1


# Problem 13.5.14 ----

# Initialize the s_t state vector:
s_t[:] = temp_min     # Initial Temperature Distribution 

# Inner ring B.C. 
for row in range(Q):
    s_t[row] = temp_max      # Value held at inner ring
    for col in range(ele):
        if col == row:
            A_mat[row][col] = 1.
            B_mat[row][col] = 1.
        else:
            A_mat[row][col] = 0.
            B_mat[row][col] = 0.

# Outer ring B.C. 
mid = int(((N-1)/2) + 1)
i = 1
for row in range(ele-Q,ele):
    if i <= Q/2:
        s_t[row] = temp_min     # Value held at upper half of outer ring
    else:
        s_t[row] = temp_min     # Value held at lower half of outer ring
    i += 1
    for col in range(ele):
        if col == row:
            A_mat[row][col] = 1.
            B_mat[row][col] = 1.
        else:
            A_mat[row][col] = 0.
            B_mat[row][col] = 0.
    

### ----- Define Polar to Cartesian Gridding ----- ###:

rho = np.linspace(drho*(N_half-1),rmax,N_half)
phi = np.linspace(0,2.*np.pi,Q+1)

phi_m, rho_m = np.meshgrid(phi,rho)


X = np.array([[0. for j in range(Q+1)] for i in range(N_half)])
Y = np.array([[0. for j in range(Q+1)] for i in range(N_half)])
for i in range(N_half):
    for j in range(Q+1):
        X[i][j] = rho_m[i][j]*np.cos(phi_m[i][j])
        Y[i][j] = rho_m[i][j]*np.sin(phi_m[i][j])

### ----- Initialize Z ----- ###
# Z is a matrix of U_t that maps to X and Y for plotting purposes. 

Z = np.array([[0. for i in range(Q+1)] for j in range(N_half)])

for i in range(N_half):
    for j in range(Q):
        if j == 0:
            Z[i][j] = s_t[i*Q+j]
            Z[i][Q] = s_t[i*Q+j]
        else:
            Z[i][j] = s_t[i*Q+j]
            
### ----- Plotting and Animating ----- ###

fig = plt.figure()
plt.axes().set_aspect('equal') 
plt.xlabel(r'$x/L_{0}$')
plt.ylabel(r'$y/L_{0}$')

plt.pcolormesh(X, Y, Z, cmap='plasma', vmin=temp_min, vmax=temp_max)
plt.colorbar(extend='both')

k = 0
def animate(k):
    global A_mat, B_mat, s_t, s_tnext, X, Y
            
    right_side = np.dot(B_mat,s_t)
    s_tnext = np.linalg.solve(A_mat,right_side)
    s_t = s_tnext
    
    Z = np.array([[0. for i in range(Q+1)] for j in range(N_half)])

    for i in range(N_half):
        for j in range(Q):
            if j == 0:
                Z[i][j] = s_t[i*Q+j]
                Z[i][Q] = s_t[i*Q+j]
            else:
                Z[i][j] = s_t[i*Q+j]
                
    cont = plt.pcolormesh(X, Y, Z, cmap='plasma', vmin=temp_min, vmax=temp_max)

    plt.title(r'$t$ = %i$\Delta t$' %(k))
    plt.suptitle(r'Temperature $U/U_{0}$')

    return cont

anim = animation.FuncAnimation(fig, animate, frames=20, blit=False)

# Un-commenting the following line will save the animation as a mp4 file. 
#anim.save('ring_bc3.mp4', fps=15)

# Un-commenting the following line will show the animation of the solution being 
# solved in real time. 
plt.show()

