"""
One-dimensional Forward Time, Centered Space solver for the unsteady heat equation with a time-dependent point pulse source to simulate the
exposure of a rocket thrust chamber wall to hot combustion gases under a pulsed operation
"""

import numpy as np

def run_ftcs_cool(dt, nt, l=0.01, ni=100):

    # Physical Constants
    kap = 65.0    # Thermal conductivity [W/m/K]
    a = 1.7e-5    # Thermal diffusivity [m^2/s]
    hg = 1.5e4    # Convective heat transfer coefficient [W/m^2/K]
    T0g = 2500    # Combustion gas stagnation temperature [K]


    # Time and Position Constants
    #l = 0.01       # domain length [m]
    #ni = 100       # number of spacial cells
    dx = l/(ni-1)

    #dt = 1e-4      # time step [s]
    #nt = 15000     # number of time steps

    nu = ni*nt     # total amount of points

    ft = 0.15      # firing duration [s]
    fp = 0.2       # firing period [s]
    t0 = 0.0       # first firing begins [s]

    # Initial Conditions
    firing = False
    T = np.full((ni, nt), np.nan)
    T[:, 0] = 300


    for k in range(nt-1):
        t = dt * k

        # figure out if firing
        firing = (t >= t0) and ((t - t0) // fp < 6) and ((t - t0) % fp < ft)

        for i in range(ni):
            if i == 0:
                if firing:
                    ghost = ((2 * hg * dx) / kap) * (T0g - T[i,k]) + T[i+1,k]
                    T[i, k + 1] = T[i, k] + a * dt * (ghost - 2*T[i, k]+ T[i + 1,k]) / dx ** 2  # convective heat transfer
                else:
                    T[i,k+1] = T[i,k] + a * dt * 2 * (T[i+1,k] - T[i,k]) / dx ** 2  # dT/dx = 0
            elif i == ni-1:
                T[i,k+1] = T[i,k] + a * dt * 2 * (T[i-1,k] - T[i,k]) / dx ** 2  # dT/dx = 0
            else:
                T[i, k + 1] = T[i, k] + a * dt * (T[i - 1, k] - 2*T[i, k] + T[i + 1,k]) / dx ** 2  # interior node

    return T