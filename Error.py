import numpy as np
import time
import matplotlib.pyplot as plt

from FTCS_Cool import run_ftcs_cool
from CN_Cool import run_cn_cool

# Physical / numerical parameters
L = 0.01
ni = 100
dx = L/(ni-1)
T_end = 1.4   # total simulated time [s]


# Timesteps to test
dt_max = dx**2 / (2 * 1.7e-5)
dt = dt_max * 0.1 # s
nt = T_end/(dt)
dt_list = dt_max*np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 1.1, 1.2, 1.3, 1.4, 1.5])



runtime_cn = []
error_cn = []

runtime_ftcs = []
error_ftcs = []

for dt in dt_list:
    nt = int(T_end / dt)

    T_ref = run_cn_cool(dt_max*.1, nt, L, ni)

    # --- CN ---
    t0 = time.perf_counter()
    T_cn = run_cn_cool(dt, nt, L, ni)
    runtime_cn.append(time.perf_counter() - t0)

    error_cn.append(np.mean([T_cn - T_ref]))

    # --- FTCS ---
    t0 = time.perf_counter()
    T_ftcs = run_ftcs_cool(dt, nt, L, ni)
    runtime_ftcs.append(time.perf_counter() - t0)

    error_ftcs.append(np.mean([T_ftcs - T_ref]))

plt.figure()
plt.plot(dt_list, runtime_cn, 'o-', label='CN')
plt.plot(dt_list, runtime_ftcs, 's-', label='FTCS')
plt.xlabel(r'Timestep $\Delta t$ [s]')
plt.ylabel('Runtime [s]')
plt.legend()
plt.grid(True, which='both')
plt.title('Runtime vs Timestep')
plt.show()

plt.figure()
plt.plot(dt_list, error_cn, 'o-', label='CN')
plt.plot(dt_list, error_ftcs, 's-', label='FTCS')
plt.xlabel(r'Timestep $\Delta t$ [s]')
plt.ylabel('L2 Error at Final Time [K]')
plt.legend()
plt.grid(True, which='both')
plt.title('Error vs Timestep')
plt.show()

plt.figure()
plt.plot(runtime_cn, error_cn, 'o-', label='CN')
plt.plot(runtime_ftcs, error_ftcs, 's-', label='FTCS')
plt.xlabel('Runtime [s]')
plt.ylabel('Average Error [K]')
plt.legend()
plt.grid(True, which='both')
plt.title('Error vs Computational Cost')
plt.show()
