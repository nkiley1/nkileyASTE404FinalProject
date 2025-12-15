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

    # Initial Conditions
    T = np.full(ni, 300.0)  # Temperature vector for the current time step

    # Storage Matrix for Plotting
    T_history = np.zeros((ni, nt))
    T_history[:, 0] = T

    # Precompute constant
    r = a * dt / (dx * dx)  # Fourier number
    beta = 2 * hg * dx / kap  # Dimensionless convection parameter

    # Solver Function
    def heat_solver_cn(T_current, is_firing):
        # Build matrix A (LHS) and vector b (RHS) for each timestep
        A = sp_mat.lil_matrix((ni, ni))
        b = np.zeros(ni)

        # Left boundary (i=0): Convective or Insulated
        if is_firing:  # Convective boundary
            # LHS: T_0^{n+1} - r*T_1^{n+1} + (r/2)*(2+β)*T_0^{n+1}
            A[0, 0] = 1 + (r/2) * (2 + beta)
            A[0, 1] = -r
            # RHS: T_0^n + r*T_1^n - (r/2)*(2+β)*T_0^n + r*β*T_g
            b[0] = (1 - (r/2) * (2 + beta)) * T_current[0] + r * T_current[1] + r * beta * T0g
        else:  # Insulated boundary (dT/dx = 0)
            A[0, 0] = 1 + r
            A[0, 1] = -r
            b[0] = (1 - r) * T_current[0] + r * T_current[1]

        # Interior nodes (i = 1 to ni-2)
        for i in range(1, ni - 1):
            A[i, i-1] = -r/2
            A[i, i] = 1 + r
            A[i, i+1] = -r/2
            b[i] = (r/2) * T_current[i-1] + (1 - r) * T_current[i] + (r/2) * T_current[i+1]

        # Right boundary (i=ni-1): Insulated (dT/dx = 0)
        A[ni-1, ni-2] = -r
        A[ni-1, ni-1] = 1 + r
        b[ni-1] = r * T_current[ni-2] + (1 - r) * T_current[ni-1]

        # Convert to CSC format for efficient solving
        A = A.tocsc()

        # Solve the sparse linear system A * T_next = b
        T_next = spsolve(A, b)

        return T_next

    # Main Loop: March Solution Forward in Time
    for k in range(1, nt+1):
        t = dt * (k - 1)

        # Determine if the rocket is firing at the current time
        firing = (t >= t0) and ((t - t0) // fp < 6) and ((t - t0) % fp < ft)

        # Get the new solution for the next time step
        T = heat_solver_cn(T, firing)

        # Store result
        T_history[:, k-1] = T

    return T_history