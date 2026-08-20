"""Debug the LEV feedback sign: per-step v_lev.n at collocation + Gamma trace."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/fluxv-v5-nextgen/src")
sys.path.insert(0, "/tmp/fluxv-v5-nextgen/platform")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pterasoftware as ps
import bing_joint_ptera as bjp
from bing_joint_ptera import JointLEVTEVSolver, JointConfig

L_ch, AR = 5.0, 15.0
L_sp = L_ch * AR
V, rho = 10.0, 1.225
qS = 0.5 * rho * V * V * L_ch * L_sp

# monkey-patch a probe into _calculate_vortex_strengths via wrapper
orig = JointLEVTEVSolver._calculate_vortex_strengths


def probed(self):
    orig(self)
    cps = np.asarray(self.stackCpp_GP1_CgP1)
    norms = np.asarray(self.stackUnitNormals_GP1)
    if self.lev_pf.n > 0:
        v = self.lev_pf.velocity_at(cps)
        vn = np.einsum("ij,ij->i", v, norms)
        print(f"step {self._current_step:2d}: np={self.lev_pf.n:4d} "
              f"v_lev.n mean={vn.mean():+.3f} max={vn.max():+.3f} "
              f"G_LEV_sum={self._lev_solved.sum():+.2f}")
    C, S = self._panel_grid()[0].shape
    gb = self._current_bound_vortex_strengths.reshape(C, S)
    print(f"          G_LE mid={gb[0, S//2]:+.2f} G_TE mid={gb[C-1, S//2]:+.2f}")


JointLEVTEVSolver._calculate_vortex_strengths = probed

outline = np.asarray([[1.0, 1e-4], [0.5, 1e-4], [0.0, 0.0], [0.5, -1e-4], [1.0, -1e-4]])
airfoil = ps.geometry.airfoil.Airfoil(name="flat", outline_A_lp=outline, resample=False)
root = ps.geometry.wing_cross_section.WingCrossSection(
    airfoil=airfoil, chord=L_ch, num_spanwise_panels=12, spanwise_spacing="cosine")
tip = ps.geometry.wing_cross_section.WingCrossSection(
    airfoil=airfoil, chord=L_ch, Lp_Wcsp_Lpp=(0.0, L_sp, 0.0),
    num_spanwise_panels=None, spanwise_spacing=None)
wing = ps.geometry.wing.Wing(
    name="flat", wing_cross_sections=[root, tip], angles_Gs_to_Wn_ixyz=(0.0, 20.0, 0.0),
    symmetric=False, num_chordwise_panels=8, chordwise_spacing="uniform")
airplane = ps.geometry.airplane.Airplane(
    wings=[wing], name="flat", s_ref=L_ch * L_sp, c_ref=L_ch, b_ref=L_sp)
op = ps.operating_point.OperatingPoint(rho=rho, vCg__E=V, alpha=0.0, beta=0.0, nu=1.5e-5)
sec_moves = [ps.movements.wing_cross_section_movement.WingCrossSectionMovement(
    base_wing_cross_section=sec) for sec in (root, tip)]
wm = ps.movements.wing_movement.WingMovement(base_wing=wing,
                                             wing_cross_section_movements=sec_moves)
am = ps.movements.airplane_movement.AirplaneMovement(
    base_airplane=airplane, wing_movements=[wm])
opm = ps.movements.operating_point_movement.OperatingPointMovement(base_operating_point=op)
movement = ps.movements.movement.Movement(
    airplane_movements=[am], operating_point_movement=opm, delta_time=0.25, num_steps=16)
problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)

solver = JointLEVTEVSolver(problem, JointConfig(lesp_crit=0.11,
                                                lev_start_step=3, enable_lev=True))
solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
