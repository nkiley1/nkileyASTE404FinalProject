import numpy as np
import matplotlib.pyplot as plt
import time

from FTCS_Cool import run_ftcs_cool
from CN_Cool import run_cn_cool

# -----------------------------
# User parameters
# -----------------------------
dt = 1e-4
nt = 15000
L = 0.01
ni = 100

x = np.linspace(0, L, ni)
t = np.arange(nt) * dt

# -----------------------------
# Run solvers (with timing)
# -----------------------------
t0 = time.time()
T_cn = run_cn_cool(dt, nt, L, ni)
cn_time = time.time() - t0

t0 = time.time()
T_ftcs = run_ftcs_cool(dt, nt, L, ni)
ftcs_time = time.time() - t0

print(f"CN runtime   : {cn_time:.3f} s")
print(f"FTCS runtime: {ftcs_time:.3f} s")

# -----------------------------
# 1. Max absolute difference vs time
# -----------------------------
abs_diff = np.abs(T_cn - T_ftcs)
max_diff_t = np.max(abs_diff, axis=0)

plt.figure()
plt.plot(t, max_diff_t)
plt.xlabel("Time [s]")
plt.ylabel("Max |T_CN - T_FTCS| [K]")
plt.title("Maximum Temperature Difference vs Time")
plt.grid()
plt.show()

# -----------------------------
# 3. Wall temperature history
# -----------------------------
plt.figure()
plt.plot(t, T_cn[0, :], label="CN")
plt.plot(t, T_ftcs[0, :], '--', label="FTCS")
plt.xlabel("Time [s]")
plt.ylabel("Wall Temperature [K]")
plt.title("Wall Temperature History")
plt.legend()
plt.grid()
plt.show()

