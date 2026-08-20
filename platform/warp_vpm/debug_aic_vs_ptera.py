"""Final AIC parity: my ring_velocity on pterasoftware's actual rvp rings."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/fluxv-v5-nextgen/src")
sys.path.insert(0, "/tmp/fluxv-v5-nextgen/platform")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pterasoftware as ps
from bing_joint_solver import ring_velocity

L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
outline = np.asarray([[1.0, 1e-4], [0.5, 1e-4], [0.0, 0.0], [0.5, -1e-4], [1.0, -1e-4]])
airfoil = ps.geometry.airfoil.Airfoil(name="flat", outline_A_lp=outline, resample=False)
root = ps.geometry.wing_cross_section.WingCrossSection(
    airfoil=airfoil, chord=L_ch, num_spanwise_panels=12, spanwise_spacing="cosine")
tip = ps.geometry.wing_cross_section.WingCrossSection(
    airfoil=airfoil, chord=L_ch, Lp_Wcsp_Lpp=(0.0, L_sp, 0.0),
    num_spanwise_panels=None, spanwise_spacing=None)
wing = ps.geometry.wing.Wing(
    name="flat", wing_cross_sections=[root, tip],
    angles_Gs_to_Wn_ixyz=(0.0, 5.0, 0.0),
    symmetric=False, num_chordwise_panels=8, chordwise_spacing="uniform")
airplane = ps.geometry.airplane.Airplane(
    wings=[wing], name="flat", s_ref=L_ch * L_sp, c_ref=L_ch, b_ref=L_sp)
op = ps.operating_point.OperatingPoint(rho=1.225, vCg__E=10.0, alpha=0.0, beta=0.0, nu=1.5e-5)
problem = ps.problems.SteadyProblem([airplane], operating_point=op)
solver = ps.steady_ring_vortex_lattice_method.SteadyRingVortexLatticeMethodSolver(problem)
solver.run()

ptera_aic = np.asarray(solver._gridWingWingInfluences__E)
cpp = np.asarray(solver.stackCpp_GP1_CgP1)
norm = np.asarray(solver.stackUnitNormals_GP1)

flat_panels = solver.panels
N = len(flat_panels)
rings = np.array([[np.asarray(p.ring_vortex.Frrvp_GP1_CgP1),
                   np.asarray(p.ring_vortex.Brrvp_GP1_CgP1),
                   np.asarray(p.ring_vortex.Blrvp_GP1_CgP1),
                   np.asarray(p.ring_vortex.Flrvp_GP1_CgP1)] for p in flat_panels])

vel = ring_velocity(cpp, rings)
my_aic = np.einsum("ij,ikj->ik", norm, vel)
print("N =", N)
print("diag mine :", np.round(np.diag(my_aic)[:5], 4))
print("diag ptera:", np.round(np.diag(ptera_aic)[:5], 4))
d = np.abs(my_aic - ptera_aic)
print("max |diff|:", d.max(), " mean |diff|:", d.mean())

fs = np.asarray(solver.stackFreestreamWingInfluences__E)
g_mine = np.linalg.solve(my_aic, -fs)
g_ptera = np.array([p.ring_vortex.strength for p in flat_panels])
print("G mine  (first 10):", np.round(g_mine[:10], 3))
print("G ptera (first 10):", np.round(g_ptera[:10], 3))
print("max |dG|:", np.abs(g_mine - g_ptera).max())
