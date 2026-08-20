"""Compare lattice arrangements on a single-strip flat plate (2D-ish).

(a) 1/4-chord ring lattice + 3/4 collocation (current BING port)
(b) panel-corner ring lattice + panel-center collocation (standard K&P VLM)
(c) panel-corner rings + collocation at 3/4 of each panel
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bing_joint_solver import ring_velocity

ch = 8
L_ch, L_sp = 5.0, 2000.0   # huge span -> quasi-2D
alpha = np.deg2rad(5.0)
V = 10.0
pivot = 0.25 * L_ch
xs = np.linspace(0.0, L_ch, ch + 1)
ys = np.array([-L_sp / 2, L_sp / 2])


def make_nodes():
    nodes = np.zeros((2, ch + 1, 3))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            nodes[i, j] = (x, y, (pivot - x) * np.tan(alpha))
    return nodes


nodes = make_nodes()
gamma_exact = np.pi * V * L_ch * alpha
print(f"thin-airfoil G = {gamma_exact:.3f}")

n_up, n_dn = nodes[1], nodes[0]  # station i+1 (tip), station i (root)


def solve_lattice(ring_v, targets):
    rings = ring_v.reshape(ch, 4, 3)
    # normal from flat-plate geometry (all panels same plane)
    diag1 = rings[:, 0] - rings[:, 2]
    diag2 = rings[:, 3] - rings[:, 1]
    cross = np.cross(diag1, diag2)
    nrm = cross / np.linalg.norm(cross, axis=1)[:, None]
    aic = np.einsum("ij,ikj->ik", nrm, ring_velocity(targets, rings))
    rhs = np.array([V, 0, 0]) @ nrm.T
    return np.linalg.solve(aic, -rhs), aic, rhs


# (a) 1/4-lattice rings + 3/4 collocation (current)
aero_front = nodes[:, :-1] + 0.25 * (nodes[:, 1:] - nodes[:, :-1])  # (2, ch, 3)
back = aero_front.copy()
back[:, :-1] = aero_front[:, 1:]
back[:, -1] = nodes[:, -1]
tq = nodes[:, :-1] + 0.75 * (nodes[:, 1:] - nodes[:, :-1])
targets_a = 0.5 * (tq[1] + tq[0])
rings_a = np.stack((aero_front[1], back[1], back[0], aero_front[0]), axis=1)
g_a, aic_a, rhs_a = solve_lattice(rings_a, targets_a)
print(f"(a) 1/4 rings + 3/4 CP : G_TE={g_a[-1]:.3f}  G={np.round(g_a,2)}")

# (b) corner rings + center collocation
rings_b = np.stack((nodes[1, 1:], nodes[1, :-1], nodes[0, :-1], nodes[0, 1:]), axis=1)
centers = 0.25 * (nodes[0, :-1] + nodes[0, 1:] + nodes[1, :-1] + nodes[1, 1:])
g_b, _, _ = solve_lattice(rings_b, centers)
print(f"(b) corner rings + center CP : G_TE={g_b[-1]:.3f}  G={np.round(g_b,2)}")

# (c) corner rings + 3/4-of-panel CP
tq_c = nodes[:, :-1] + 0.75 * (nodes[:, 1:] - nodes[:, :-1])
targets_c = 0.5 * (tq_c[1] + tq_c[0])
g_c, _, _ = solve_lattice(rings_b, targets_c)
print(f"(c) corner rings + 3/4 CP : G_TE={g_c[-1]:.3f}  G={np.round(g_c,2)}")

# (d) 1/4-lattice rings + panel-center collocation
midpanel = nodes[:, :-1] + 0.5 * (nodes[:, 1:] - nodes[:, :-1])
targets_d = 0.5 * (midpanel[1] + midpanel[0])
g_d, _, _ = solve_lattice(rings_a, targets_d)
print(f"(d) 1/4 rings + center CP : G_TE={g_d[-1]:.3f}  G={np.round(g_d,2)}")
