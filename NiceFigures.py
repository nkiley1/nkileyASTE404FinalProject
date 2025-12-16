"""
Plot Nice Figures of time vs position for the two solvers
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import time

from FTCS_Cool import run_ftcs_cool
from CN_Cool import run_cn_cool

# --- Simulation Parameters ---
L = 0.01
ni = 100
dx = L / (ni - 1)
a = 1.7e-5

dt_max = dx**2 / (2 * a)
dt = dt_max * 0.1 # s
t_max = 1.4 # s
nt = int(t_max / dt)
#T_standard = run_cn_cool(dt_max * .1, nt, L, ni)

############ CN ##############
#start_time = time.time()
T = run_cn_cool(dt, nt, L, ni)
#cn_time = time.time() - start_time

############ FTCS ##############
#start_time = time.time()
T1 = run_ftcs_cool(dt,nt,L,ni)
#ftcs_time = time.time() - start_time

#T_CN = T - T_standard
#T_FTCS = T1 - T_standard
#max_CN = np.sqrt(np.mean((T_CN)**2))
#max_FTCS = np.sqrt(np.mean((T_FTCS)**2))

#print(f"FTCS:         {ftcs_time:.4f} s")
#print(f"CN:           {cn_time:.4f} s")
#print(f"CN_Tavg:      {max_CN:.4f} K")
#print(f"FTCS_Tavg:    {max_FTCS:.4f} K")

plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 7))

im = ax.imshow(T.T, extent=[0, L, 0, dt*nt], origin='lower', aspect='auto', cmap='plasma')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (K)')

ax.set_xlabel('Wall Position, x (m)')
ax.set_ylabel('Time (s)')




plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 7))

im = ax.imshow(T1.T, extent=[0, L, 0, dt*nt], origin='lower', aspect='auto', cmap='plasma')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (K)')

ax.set_xlabel('Wall Position, x (m)')
ax.set_ylabel('Time (s)')


T_dif = T - T1
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 7))

v = np.percentile(np.abs(T_dif), 99.95) 
norm = TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)
im = ax.imshow(T_dif.T, extent=[0, L, 0, dt*nt], origin='lower', aspect='auto', cmap='seismic', norm=norm)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Temperature (K)')

ax.set_xlabel('Wall Position, x (m)')
ax.set_ylabel('Time (s)')


#dT1dx = np.gradient(T1, dx, axis=1)
#dTdx = np.gradient(T, dx, axis=1)
#ddTdx = dTdx - dT1dx
# --- Color normalization (increase contrast) ---
#v = np.percentile(np.abs(ddTdx), 95)   # adjust 90–99 as needed
#norm = TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v)

# # --- Plot gradient heatmap ---
# plt.style.use('default')
# fig, ax = plt.subplots(figsize=(10, 7))

# im = ax.imshow(
#     ddTdx.T,
#     extent=[0, L, 0, dt * nt],
#     origin='lower',
#     aspect='auto',
#     cmap='seismic',
#     norm=norm
# )

# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(r'$\partial T / \partial x$ (K/m)')

# ax.set_xlabel('Wall Position, x (m)')
# ax.set_ylabel('Time (s)')

# plt.tight_layout()

# plt.style.use('default')
# fig, ax = plt.subplots(figsize=(10, 7))

# im = ax.imshow(
#     dT1dx.T,
#     extent=[0, L, 0, dt * nt],
#     origin='lower',
#     aspect='auto',
#     cmap='seismic',
#     norm=norm
# )

# cbar = fig.colorbar(im, ax=ax)
# cbar.set_label(r'$\partial T / \partial x$ (K/m)')

# ax.set_xlabel('Wall Position, x (m)')
# ax.set_ylabel('Time (s)')

plt.tight_layout()
plt.show()