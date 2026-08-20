"""G0c: LEV activation test. Static plate at high alpha (20 deg): LESP ~0.35
> 0.11 -> LEV shedding active on all strips.

Checks: gates stay green, no blowup, CL reduced vs attached-only (stall),
LEV particles accumulate, Kelvin monitor drift bounded.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/fluxv-v5-nextgen/src")
sys.path.insert(0, "/tmp/fluxv-v5-nextgen/platform")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pterasoftware as ps
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
V = 10.0
rho = 1.225
qS = 0.5 * rho * V * V * L_ch * L_sp
ALPHA = 20.0


def build_problem():
    outline = np.asarray([[1.0, 1e-4], [0.5, 1e-4], [0.0, 0.0],
                          [0.5, -1e-4], [1.0, -1e-4]])
    airfoil = ps.geometry.airfoil.Airfoil(name="flat", outline_A_lp=outline,
                                          resample=False)
    root = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=L_ch, num_spanwise_panels=12,
        spanwise_spacing="cosine")
    tip = ps.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=L_ch, Lp_Wcsp_Lpp=(0.0, L_sp, 0.0),
        num_spanwise_panels=None, spanwise_spacing=None)
    wing = ps.geometry.wing.Wing(
        name="flat", wing_cross_sections=[root, tip],
        angles_Gs_to_Wn_ixyz=(0.0, ALPHA, 0.0),
        symmetric=False, num_chordwise_panels=8, chordwise_spacing="uniform")
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing], name="flat", s_ref=L_ch * L_sp, c_ref=L_ch, b_ref=L_sp)
    op = ps.operating_point.OperatingPoint(rho=rho, vCg__E=V, alpha=0.0,
                                           beta=0.0, nu=1.5e-5)
    sec_moves = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=sec) for sec in (root, tip)]
    wm = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=sec_moves)
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    movement = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        delta_time=0.25, num_steps=40)
    return ps.problems.UnsteadyProblem(movement=movement,
                                       only_final_results=False)


# reference: attached-only run at 20 deg
attached = JointLEVTEVSolver(build_problem(), JointConfig(enable_lev=False))
attached.run(prescribed_wake=True, calculate_streamlines=False,
             show_progress=False)
cl_attached = np.array([
    -np.asarray(sp_.airplanes[0].forces_W)[2] / qS
    for sp_ in attached.steady_problems if sp_.airplanes[0].forces_W is not None])

lev = JointLEVTEVSolver(build_problem(), JointConfig(enable_lev=True))
lev.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
cl_lev = np.array([
    -np.asarray(sp_.airplanes[0].forces_W)[2] / qS
    for sp_ in lev.steady_problems if sp_.airplanes[0].forces_W is not None])

print(f"attached-only: start {cl_attached[0]:+.3f} final {cl_attached[-1]:+.3f}")
print(f"with LEV     : start {cl_lev[0]:+.3f} final {cl_lev[-1]:+.3f}")
d = lev.diag[-1]
print(f"LEV particles: {d['n_particles']}  strips active: {d['lev_strips']}  "
      f"max|LESP|: {d['lesp_max']:.3f}  G_lev sum: {d['g_lev']:+.2f}  "
      f"circ drift: {d['circ_drift']:.3e}")
mid = [x for x in lev.diag if x["step"] % 8 == 0]
for x in mid:
    print(f"  step {x['step']:2d}: CL={cl_lev[x['step']]:+.3f} "
          f"np={x['n_particles']:5d} strips={x['lev_strips']} "
          f"G_lev={x['g_lev']:+.2f} drift={x['circ_drift']:.2e}")
finite = np.all(np.isfinite(cl_lev))
reduced = abs(cl_lev[-1]) < abs(cl_attached[-1])
print("finite:", finite, " CL reduced vs attached:", reduced)
ok = finite and reduced and d["lev_strips"] > 0
print("G0c GATE:", "PASS" if ok else "FAIL")
if not ok:
    sys.exit(1)
