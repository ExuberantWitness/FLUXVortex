"""Debug G0: isolate solve vs loads by comparing pressure-CL with the Kutta-
Joukowski estimate from the solved trailing-edge ring strengths."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bing_joint_solver import BingConfig, BingJointSolver, WingLattice

sp, ch = 12, 8
L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
alpha_deg = 5.0
V = 10.0
dt = 0.05
rho = 1.225
n_steps = 40

alpha = np.deg2rad(alpha_deg)
pivot = 0.25 * L_ch

def cos_pos(a, b, n):
    t = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
    return a + (b - a) * t

xs = cos_pos(0.0, L_ch, ch)
ys = cos_pos(-L_sp / 2, L_sp / 2, sp)
nodes = np.zeros((sp + 1, ch + 1, 3))
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        nodes[i, j] = (x, y, (pivot - x) * np.tan(alpha))

qS = 0.5 * rho * V * V * L_ch * L_sp
dys = np.diff(ys)

cfg = BingConfig(dt=dt, rho=rho, v_freestream=np.array([V, 0.0, 0.0]),
                 n_span=sp, n_chord=ch)
solver = BingJointSolver(cfg)
for n in range(n_steps):
    last = solver.step(n, nodes, nodes, nodes)

g = solver.gamma_hist[-1].reshape(sp, ch)
lat = WingLattice(sp, ch, nodes)

# KJ estimate from TE ring strengths
L_kj = rho * V * np.sum(g[:, -1] * dys)
cl_kj = L_kj / qS
print(f"CL (pressure)     = {last.force[2] / qS:+.4f}")
print(f"CL (KJ from G_TE) = {cl_kj:+.4f}")
print(f"G_TE per strip    = {np.round(g[:, -1], 3)}")
print(f"G_LE per strip    = {np.round(g[:, 0], 3)}")
print(f"row sums per strip= {np.round(g.sum(axis=1), 3)}")

# interior telescoping check: sum_j gamma_ch * d_ch should = G_TE
from bing_joint_solver import gamma_ch_from_rings
gch = gamma_ch_from_rings(solver.gamma_hist[-1], sp, ch)
telescoped = np.sum(gch * lat.d_ch, axis=1)
print(f"telescoped sum vs G_TE: {np.round(telescoped, 3)} vs {np.round(g[:, -1], 3)}")

# (V . n_ch) sanity
n_ch_x = lat.n_ch[..., 0]
print(f"mean(V.n_ch) = {V * n_ch_x.mean():.3f} (V={V})")
