"""Debug: unit-test line_velocity and the flat-plate AIC solve magnitude."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bing_joint_solver import WingLattice, line_velocity, ring_velocity

# --- 1. infinite-line limit ---
origin = np.array([[0.0, -10000.0, 0.0]])
termination = np.array([[0.0, 10000.0, 0.0]])
target = np.array([[5.0, 0.0, 0.0]])
v = line_velocity(target, origin, termination)[0, 0]
v_exact = np.array([0.0, 0.0, -1.0 / (2 * np.pi * 5.0)])  # s=+Y, r=+x -> y cross x = -z
print("line test  :", np.round(v, 6), "exact:", np.round(v_exact, 6))

# --- 2. flat plate single strip, uniform chord panels ---
sp, ch = 1, 8
L_ch, L_sp = 5.0, 75.0
alpha = np.deg2rad(5.0)
V = 10.0
pivot = 0.25 * L_ch
xs = np.linspace(0.0, L_ch, ch + 1)
ys = np.array([-L_sp / 2, L_sp / 2])
nodes = np.zeros((sp + 1, ch + 1, 3))
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        nodes[i, j] = (x, y, (pivot - x) * np.tan(alpha))

lat = WingLattice(sp, ch, nodes)
N = sp * ch
targets = lat.collocation.reshape(N, 3)
rings_flat = lat.ring_vertices.reshape(N, 4, 3)
aic = np.einsum("ij,ikj->ik", lat.panel_normal.reshape(N, 3),
                ring_velocity(targets, rings_flat))
rhs = np.einsum("ij,ij->i", np.broadcast_to(np.array([V, 0, 0]), (N, 3)),
                lat.panel_normal.reshape(N, 3))
gamma = np.linalg.solve(aic, -rhs)
print("G per chord panel:", np.round(gamma, 3))
print(f"G_TE = {gamma[-1]:.3f}   pi*U*c*alpha = {np.pi * V * L_ch * alpha:.3f}")
print("AIC diag range:", np.round(np.diag(aic), 3))

# --- 3. same with many strips (3D, AR=15) ---
sp = 12
ys = np.linspace(-L_sp / 2, L_sp / 2, sp + 1)
nodes = np.zeros((sp + 1, ch + 1, 3))
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        nodes[i, j] = (x, y, (pivot - x) * np.tan(alpha))
lat = WingLattice(sp, ch, nodes)
N = sp * ch
targets = lat.ring_vertices.reshape(N, 4, 3)  # placeholder to keep shapes
targets = lat.collocation.reshape(N, 3)
rings_flat = lat.ring_vertices.reshape(N, 4, 3)
aic = np.einsum("ij,ikj->ik", lat.panel_normal.reshape(N, 3),
                ring_velocity(targets, rings_flat))
rhs = np.einsum("ij,ij->i", np.broadcast_to(np.array([V, 0, 0]), (N, 3)),
                lat.panel_normal.reshape(N, 3))
gamma = np.linalg.solve(aic, -rhs).reshape(sp, ch)
print("mid-strip G:", np.round(gamma[sp // 2], 3))
print(f"mid-strip G_TE = {gamma[sp//2,-1]:.3f}")
