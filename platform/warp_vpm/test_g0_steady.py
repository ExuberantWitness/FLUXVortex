"""G0 steady regression: rectangular wing AR=15, alpha=5 deg vs lifting-line.

Uses the PRODUCTION chassis (bing_joint_ptera, pterasoftware mixin) — the
from-scratch architecture (bing_joint_solver) is a documented prototype and
is not the production path. See BING_SESSION_RESULTS.md for details.

Gate: CL must land in [lower, upper*1.015] where bounds are the lifting-line
estimates and the 1.5% allowance covers the un-decayed impulsive transient
at 40 steps.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pterasoftware
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
alpha_deg = 5.0
V, rho = 10.0, 1.225
alpha = np.deg2rad(alpha_deg)
qS = 0.5 * rho * V * V * L_ch * L_sp

cl1 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.05) / (np.pi * AR)) * alpha
cl2 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.25) / (np.pi * AR)) * alpha
cl_upper = cl1 * 1.015   # documented unsteady tolerance
print(f"lifting-line bounds: [{cl2:.4f}, {cl1:.4f}] "
      f"(tolerance upper: {cl_upper:.4f})")


def build_problem():
    outline = np.asarray([[1.0, 1e-4], [0.5, 1e-4], [0.0, 0.0],
                          [0.5, -1e-4], [1.0, -1e-4]])
    airfoil = pterasoftware.geometry.airfoil.Airfoil(
        name="flat", outline_A_lp=outline, resample=False)
    root = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=L_ch, num_spanwise_panels=12,
        spanwise_spacing="cosine")
    tip = pterasoftware.geometry.wing_cross_section.WingCrossSection(
        airfoil=airfoil, chord=L_ch, Lp_Wcsp_Lpp=(0.0, L_sp, 0.0),
        num_spanwise_panels=None, spanwise_spacing=None)
    wing = pterasoftware.geometry.wing.Wing(
        name="flat", wing_cross_sections=[root, tip],
        angles_Gs_to_Wn_ixyz=(0.0, alpha_deg, 0.0),
        symmetric=False, num_chordwise_panels=8,
        chordwise_spacing="uniform")
    airplane = pterasoftware.geometry.airplane.Airplane(
        wings=[wing], name="flat", s_ref=L_ch * L_sp, c_ref=L_ch,
        b_ref=L_sp)
    op = pterasoftware.operating_point.OperatingPoint(
        rho=rho, vCg__E=V, alpha=0.0, beta=0.0, nu=1.5e-5)
    sec_moves = [
        pterasoftware.movements.wing_cross_section_movement
        .WingCrossSectionMovement(base_wing_cross_section=sec)
        for sec in (root, tip)]
    wm = pterasoftware.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=sec_moves)
    am = pterasoftware.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wm])
    opm = pterasoftware.movements.operating_point_movement.\
        OperatingPointMovement(base_operating_point=op)
    movement = pterasoftware.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        delta_time=0.25, num_steps=40)
    return pterasoftware.problems.UnsteadyProblem(
        movement=movement, only_final_results=False)


solver = JointLEVTEVSolver(build_problem(), JointConfig(enable_lev=False))
solver.run(prescribed_wake=True, calculate_streamlines=False,
           show_progress=False)
cls = []
for sp in solver.steady_problems:
    f = sp.airplanes[0].forces_W
    cls.append(-np.asarray(f)[2] / qS if f is not None else np.nan)
cls = np.array(cls)
print(f"CL start={cls[0]:+.4f} mid={cls[len(cls)//2]:+.4f} "
      f"final={cls[-1]:+.4f}")

cl_final = cls[-1]
print(f"\nfinal CL = {cl_final:.4f}")
print(f"bounds   = [{cl2:.4f}, {cl_upper:.4f}]")
ok = cl2 <= cl_final <= cl_upper
sign_ok = cl_final > 0
print("G0 STEADY GATE:", "PASS" if ok else "FAIL")
print("lift sign:", "PASS" if sign_ok else "FAIL")
if not ok or not sign_ok:
    sys.exit(1)
