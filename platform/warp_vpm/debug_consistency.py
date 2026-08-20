"""Surgical consistency check: matrix column x Gamma_LEV this step vs the
particle-field contribution of those exact particles at the collocation
points next step (before convection they barely move)."""
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
V, rho = 10.0, 1.225

orig = JointLEVTEVSolver._calculate_vortex_strengths
prev_dbg = {}


def probed(self):
    # before solving: the RHS trail injection used the particle field from
    # last step; isolate the contribution of particles born at step n-1
    n = self._current_step
    if n > 0 and (n - 1) in prev_dbg:
        d = prev_dbg[n - 1]
        cps = np.asarray(self.stackCpp_GP1_CgP1)
        norms = np.asarray(self.stackUnitNormals_GP1)
        born = np.flatnonzero(self.lev_pf.birth_step[:self.lev_pf.n] == n - 1)
        if len(born):
            # NOTE: particles were convected once already (this step's
            # _calculate_wake_wing_influences ran); approximate anyway
            sub_pos = self.lev_pf.pos[born]
            sub_gam = self.lev_pf.gamma[born]
            sub_sig = self.lev_pf.sigma[born]
            from pfield import ParticleField
            tmp = ParticleField(capacity=len(born) + 1)
            tmp.add_particles(sub_pos, sub_gam, sub_sig,
                              ptype=self.lev_pf.ptype[born])
            v = tmp.velocity_at(cps)
            vn = np.einsum("ij,ij->i", v, norms)
            col_pred = d["a_lev"] @ d["lev"]
            print(f"step {n}: column@d(n-1) LE-panel={col_pred[0]:+.4f} "
                  f"TE-panel={col_pred[-1]:+.4f}")
            print(f"        particle vn     LE-panel={vn[0]:+.4f} "
                  f"TE-panel={vn[-1]:+.4f}")
    orig(self)
    prev_dbg[n] = self._dbg


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
    airplane_movements=[am], operating_point_movement=opm, delta_time=0.25, num_steps=8)
problem = ps.problems.UnsteadyProblem(movement=movement, only_final_results=False)

solver = JointLEVTEVSolver(problem, JointConfig(lesp_crit=0.11, lev_start_step=3))
solver.run(prescribed_wake=True, calculate_streamlines=False, show_progress=False)
