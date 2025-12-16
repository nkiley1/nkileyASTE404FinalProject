"""
One-dimensional Crank Nicholson solver for the unsteady heat equation with a time-dependent point pulse source to simulate the
exposure of a rocket thrust chamber wall to hot combustion gases under a pulsed operation
"""

import numpy as np
import scipy.sparse as sp_mat
from scipy.sparse.linalg import spsolve

def run_cn_cool(dt, nt, l=0.01, ni=100):
    # Physical Constants
    kap = 65.0       # Thermal conductivity [W/m/K]
    a = 1.7e-5       # Thermal diffusivity [m^2/s]
    hg = 1.5e4       # Convective heat transfer coefficient [W/m^2/K]
    T0g = 2500       # Combustion gas stagnation temperature [K]

    # Time and Position Constants
    dx = l / (ni - 1)
    ft = 0.15        # Firing duration [s]
    fp = 0.2         # Firing period [s]
    t0 = 0.0         # First firing begins [s]

    
    # Storage Matrix for Plotting
    T = np.zeros((ni, nt)) # Temp
    R = np.zeros((ni,nt)) # Source Terms
    T[:, 0] = 300
    
    L = sp_mat.lil_matrix((ni, ni))
    I = sp_mat.identity(ni)

    for k in range(nt-1):
        t = dt * k
        # figure out if firing
        firing = (t >= t0) and ((t - t0) // fp < 6) and ((t - t0) % fp < ft)
        for i in range(ni-1):
            # Left Boundary
            if i == 0:
                if firing:
                    L[0, 0] = (-2 -((2*dx*hg)/kap))/(dx*dx)
                    L[0, 1] = 2/(dx*dx)
                else:
                    L[0, 0] = -2/(dx*dx)
                    L[0, 1] = 2/(dx*dx)
            # Interior Nodes 
            elif i < ni-1:
                L[i, i-1] = 1/(dx*dx)
                L[i, i] = -2/(dx*dx)
                L[i, i+1] = 1/(dx*dx)
            # Right Boundary
            
            L[ni-1, ni-2] = 2/(dx*dx)
            L[ni-1, ni-1] = -2/(dx*dx)
        
        A = (I - ((a*dt)/2)*L)
        B = (I + ((a*dt)/2)*L)@T[:,k]
        if firing:
            B[0] += ((2*dx*hg*T0g)/(kap*dx*dx))*(a*dt)

        T[:,k+1] = spsolve(A,B)

    return T