"""G0 steady regression: rectangular wing AR=15, alpha=5 deg vs lifting-line bounds.

Reference test case from small-particle-computing-master
(230307 新型求解LESP求解模式.py): CL must land between the two
lifting-line estimates CL_ALPHA1 (upper) and CL_ALPHA2 (lower).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bing_joint_solver import BingConfig, BingJointSolver

# --- reference test parameters ---
sp, ch = 12, 8
L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
alpha_deg = 5.0
V = 10.0
dt = 0.05
rho = 1.225
n_steps = 80

alpha = np.deg2rad(alpha_deg)
pivot = 0.25 * L_ch

# cosine spacing (chord and span)
def cos_pos(a, b, n):
    t = 0.5 * (1.0 - np.cos(np.pi * np.arange(n + 1) / n))
    return a + (b - a) * t

xs = cos_pos(0.0, L_ch, ch)      # LE -> TE
ys = cos_pos(-L_sp / 2, L_sp / 2, sp)

nodes = np.zeros((sp + 1, ch + 1, 3))
for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        nodes[i, j, 0] = x
        nodes[i, j, 1] = y
        nodes[i, j, 2] = (pivot - x) * np.tan(alpha)   # LE raised, flow +X

qS = 0.5 * rho * V * V * L_ch * L_sp
cl1 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.05) / (np.pi * AR)) * alpha
cl2 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.25) / (np.pi * AR)) * alpha
print(f"lifting-line bounds: [{cl2:.4f}, {cl1:.4f}]")

cfg = BingConfig(
    dt=dt, rho=rho, v_freestream=np.array([V, 0.0, 0.0]),
    n_span=sp, n_chord=ch, lesp_crit=0.11, lev_start_step=10,
    enable_lev=True, unsteady_loads=True)

solver = BingJointSolver(cfg)
last = None
for n in range(n_steps):
    last = solver.step(n, nodes, nodes, nodes)
    if n % 10 == 0 or n == n_steps - 1:
        fz = last.force[2]
        print(f"step {n:3d}: Fz={fz:+.2f} CL={fz / qS:+.4f}  np={last.n_particles:5d} "
              f"neumann={last.neumann_residual:.2e} kelvin={last.kelvin_row_residual:.2e} "
              f"circ={last.global_circulation:+.3e} lesp={last.lesp_max:.4f} "
              f"lev_strips={last.lev_active_strips}")

cl_final = last.force[2] / qS
print(f"\nfinal CL = {cl_final:.4f}")
print(f"bounds   = [{cl2:.4f}, {cl1:.4f}]")
ok = cl2 * 0.98 <= cl_final <= cl1 * 1.02
print("G0 STEADY GATE:", "PASS" if ok else "FAIL")
# direction check: positive alpha must give positive lift
print("lift sign:", "PASS" if cl_final > 0 else "FAIL (flip ring orientation)")
