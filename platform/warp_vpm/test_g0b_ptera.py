"""G0b: static flat plate AR=15 alpha=5 through the pterasoftware unsteady
chassis, plain vs joint mixin. All three must converge to CL~0.47 (lifting
line [0.470, 0.481]).

1) plain unsteady solver (baseline)
2) joint mixin, LEV off, joint_tev on   (must match 1)
3) joint mixin, LEV on (no activation at 5 deg)
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
cl1 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.05) / (np.pi * AR)) * np.deg2rad(5)
cl2 = 2 * np.pi / (1 + 2 * np.pi * (1 + 0.25) / (np.pi * AR)) * np.deg2rad(5)
print(f"lifting-line bounds: [{cl2:.4f}, {cl1:.4f}]")


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
        angles_Gs_to_Wn_ixyz=(0.0, 5.0, 0.0),
        symmetric=False, num_chordwise_panels=8, chordwise_spacing="uniform")
    airplane = ps.geometry.airplane.Airplane(
        wings=[wing], name="flat", s_ref=L_ch * L_sp, c_ref=L_ch, b_ref=L_sp)
    op = ps.operating_point.OperatingPoint(rho=rho, vCg__E=V, alpha=0.0,
                                           beta=0.0, nu=1.5e-5)
    sec_moves = [
        ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
            base_wing_cross_section=sec) for sec in (root, tip)]
    wing_movement = ps.movements.wing_movement.WingMovement(
        base_wing=wing, wing_cross_section_movements=sec_moves)
    am = ps.movements.airplane_movement.AirplaneMovement(
        base_airplane=airplane, wing_movements=[wing_movement])
    opm = ps.movements.operating_point_movement.OperatingPointMovement(
        base_operating_point=op)
    movement = ps.movements.movement.Movement(
        airplane_movements=[am], operating_point_movement=opm,
        delta_time=0.25, num_steps=40)
    return ps.problems.UnsteadyProblem(movement=movement,
                                       only_final_results=False)


def run_and_report(tag, solver):
    solver.run(prescribed_wake=True, calculate_streamlines=False,
               show_progress=False)
    cls = []
    for sp_ in solver.steady_problems:
        f = sp_.airplanes[0].forces_W
        cls.append(-np.asarray(f)[2] / qS if f is not None else np.nan)
    cls = np.array(cls)
    print(f"{tag}: CL start={cls[0]:+.4f} mid={cls[len(cls)//2]:+.4f} "
          f"final={cls[-1]:+.4f}")
    return cls


problem = build_problem()
plain = ps.unsteady_ring_vortex_lattice_method.UnsteadyRingVortexLatticeMethodSolver(
    build_problem())
c1 = run_and_report("plain          ", plain)

problem = build_problem()
mix_off = JointLEVTEVSolver(build_problem(), JointConfig(enable_lev=False))
c2 = run_and_report("joint, LEV off ", mix_off)

problem = build_problem()
mix_on = JointLEVTEVSolver(build_problem(), JointConfig(enable_lev=True))
c3 = run_and_report("joint, LEV on  ", mix_on)

print("\nfinal diffs: off-vs-plain", f"{abs(c2[-1]-c1[-1]):.2e}",
      " on-vs-off", f"{abs(c3[-1]-c2[-1]):.2e}")
print("LEV particles (on):", mix_on.lev_pf.n,
      "max |LESP|:", f"{mix_on.diag[-1]['lesp_max']:.4f}")
ok = cl2 * 0.98 <= c1[-1] <= cl1 * 1.02
print("G0b GATE:", "PASS" if ok else "FAIL")
