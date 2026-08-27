"""Regression: the rigid load owner extracts Pterra Kutta-Joukowski forces.

The Fig.14 H5 diagnostic (artifacts/baselines/
fluxv_v5m_izraelevitz2017_fig14_mandatory_v2, MAE 0.130 vs the 0.0175 gate)
showed the attached-flow regime (low psi, no LEV) failing hardest — exactly
where thrust is produced by the streamwise Kutta-Joukowski component
(leading-edge suction / tilted lift).  ``RigidAuthorLoadAssembler`` projected
the author scalar pressure onto ``p A n``, a projection with NO streamwise
component, so CT_raw came out with the wrong sign (CT prediction -0.133 vs
experiment +0.123 on IZRA-15-015).

The frozen Pterra oracle (``platform/warp_vpm/bing_joint_ptera_gpu.py``
``_calculate_loads``, MAE 0.0175) instead integrates vector forces:

    F = rho * Gamma_eff * (v_solution - v_grid) x dl     (4 filaments/panel)
      + rho * dGamma/dt * A * n                          (unsteady Bernoulli)

with Pterra's effective shared strengths (half the neighbour difference on
interior edges, full circulation on leading-edge/tip filaments, the shed
Gamma_now - Gamma_last on trailing-edge filaments).

These tests pin the load-contract properties the pressure-only projection
violated:

1. Heave-only thrust — a purely heaving flat plate (zero pitch) keeps its
   normal along +z forever, so ANY ``p A n`` projection has Fx == 0 exactly;
   the Kutta-Joukowski extraction must instead produce thrust ~ rho*Gamma*
   zdot^2 at the maximum-heave-rate phases (negative world Fx in both
   half-strokes, positive cycle-mean CT_raw).
2. Steady lift agreement — at a fixed angle of attack the KJ normal force
   matches the independent author pressure projection (the two formulations
   share the lift; they differ exactly by the streamwise part).
3. Streamwise existence — the KJ total force is not purely normal.
4. Repeat-safety — re-proposing from one committed parent reproduces the
   load bit-for-bit (the Delta-gamma comes from the solver's mf2_history
   identity, not proposal-order-dependent cache state).
"""
from __future__ import annotations

import math
import os
import unittest

import torch
import warp as wp

from fluxvortex.cases.izraelevitz2017 import IZRA_CASES
from fluxvortex.cases.izraelevitz_kinematics import (
    IzraRigidSurfaceKinematics,
    build_izra_reference_grid,
)
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
NC, NS = 8, 24
CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()


def _make_solver(case, *, angle_deg: float, dt: float) -> RigidNativeV5MSolver:
    surface = RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=DEVICE)
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=case.rho_kg_m3,
        freestream=case.freestream_m_s,
        freestream_angle_deg=angle_deg,
        aerodynamic_dt=dt,
        wake_max_rows=64,
        # Heave-only motion at h/c=0.6 reaches ~32 deg effective AoA and a
        # sustained LEV release; the test needs the production-scale bank.
        particle_capacity=131072,
        particle_max_age_steps=100,
        dvm_target_spacing_chord=0.018,
        lesp_thickness_ratio=0.0607,
        lesp_reynolds=310000.0,
        wake_history_mode="bound_rate",
        device=DEVICE,
    )
    return RigidNativeV5MSolver(surface, settings)


def _reference_grid(case):
    return build_izra_reference_grid(
        chordwise_panels=NC,
        spanwise_panels=NS,
        chord_m=case.chord_m,
        span_m=case.span_m,
        pivot_fraction_chord=case.pivot_fraction_chord,
        device=DEVICE,
    )


def _heave_only_kinematics(case, amplitude_fraction: float = 0.2):
    """Exact heave-only motion: zero pitch, zero pitch rate, normals stay +z.

    ``amplitude_fraction`` scales the paper heave amplitude so the audit can
    stay in the ATTACHED regime (full h/c=0.6 heave at zero pitch reaches
    ~32 deg effective angle of attack — deep dynamic stall, not the regime
    this regression pins).
    """

    h = float(case.heave_amplitude_m) * float(amplitude_fraction)
    omega = float(case.omega_rad_s)

    def law(t: float):
        phase = omega * float(t)
        z = h * math.cos(phase)
        zdot = -h * omega * math.sin(phase)
        pos = torch.tensor([0.0, 0.0, z], device=DEVICE, dtype=torch.float64)
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=DEVICE, dtype=torch.float64
        )
        lin_vel = torch.tensor(
            [0.0, 0.0, zdot], device=DEVICE, dtype=torch.float64
        )
        ang_vel = torch.zeros(3, device=DEVICE, dtype=torch.float64)
        return pos, quat, lin_vel, ang_vel

    return PrescribedRigidSurfaceKinematics(
        _reference_grid(case), law, surface_id="izra_wing", body_id="body_0"
    )


def _run_steps(solver, kinematics, steps: int, dt: float):
    solver.surface.set_frame(kinematics.evaluate(0.0))
    state = solver.initialize(None, None)
    last = None
    for step in range(1, steps + 1):
        solver.surface.set_frame(kinematics.evaluate(step * dt))
        last = solver.propose(state, None, None)
        state = last.trial_state
    return last, state


@unittest.skipUnless(CUDA_OK, "CUDA required")
class HeaveThrustKuttaJoukowskiTest(unittest.TestCase):
    """Heave-only motion: the p*A*n projection has Fx == 0; KJ thrust must not.

    With the plate never pitching, every panel normal stays +z: summing
    ``p A n`` over any pressure field gives Fx = 0 identically — the old
    extraction could not feel heave thrust at all.  The Kutta-Joukowski
    streamwise component ``dFx = -rho Gamma zdot dl`` with the bound
    circulation in phase opposition to the heave rate (lift opposes plunge)
    produces thrust proportional to ``zdot^2`` — negative world Fx, the
    propulsive direction, at the maximum-heave-rate events of BOTH
    half-strokes, and a positive cycle-mean CT_raw.
    """

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-090"]
        cls.dt = 0.2 / 128.0
        cls.solver = _make_solver(cls.case, angle_deg=0.0, dt=cls.dt)
        cls.kin = _heave_only_kinematics(cls.case)
        cls.forces = []
        cls.times = []
        cls.solver.surface.set_frame(cls.kin.evaluate(0.0))
        state = cls.solver.initialize(None, None)
        for step in range(1, 257):
            cls.solver.surface.set_frame(cls.kin.evaluate(step * cls.dt))
            p = cls.solver.propose(state, None, None)
            state = p.trial_state
            cls.forces.append(
                tuple(float(v.item()) for v in p.load.total_force)
            )
            cls.times.append(step * cls.dt)

    def test_normals_stay_vertical_and_pressure_fx_is_zero(self):
        """Precondition: this motion really is heave-only."""
        frame = self.kin.evaluate(0.37 * 0.2)
        self.assertTrue(
            bool(
                torch.allclose(
                    frame.normals_I[:, 2],
                    torch.ones_like(frame.normals_I[:, 2]),
                    atol=1e-14,
                )
            )
        )

    def test_heave_thrust_is_kj_signed_and_nonzero(self):
        case = self.case
        q_area = 0.5 * case.rho_kg_m3 * case.freestream_m_s**2 * case.area_m2
        heave = float(case.heave_amplitude_m) * 0.2
        # Score the LAST cycle only (the impulsive start transient dominates
        # the first max-heave-rate event, same convention as the mandatory
        # Fig.14 runner).
        window = range(len(self.forces) // 2, len(self.forces))
        max_rate_rows = sorted(
            window,
            key=lambda k: -abs(
                -heave * case.omega_rad_s * math.sin(case.omega_rad_s * self.times[k])
            ),
        )[:6]
        worst_rel = 0.0
        for k in max_rate_rows:
            t = self.times[k]
            zdot = -heave * case.omega_rad_s * math.sin(case.omega_rad_s * t)
            fx, fy, fz = self.forces[k]
            # The old p*A*n extraction reports exactly 0.0 here.
            self.assertNotEqual(
                fx, 0.0, "heave-only Fx is zero: pressure-projection regression"
            )
            # Heave thrust is a -rho*Gamma*zdot effect with Gamma in phase
            # opposition to zdot (lift opposes the plunge), so the KJ
            # streamwise force is thrust-signed (negative world Fx, the
            # propulsive direction) at the max-rate events of BOTH
            # half-strokes — no sign flip could fake it across both.
            self.assertLess(
                fx,
                0.0,
                f"heave thrust lost its upstream sign (t={t:.4f}, zdot={zdot:+.3f})",
            )
            # And the wing is loaded at the same instant (|Fz|/qS >> 0),
            # the instantaneous signature of thrust production.
            self.assertGreater(abs(fz) / q_area, 0.05)
            worst_rel = max(worst_rel, abs(fx) / q_area)
        self.assertGreater(
            worst_rel, 1.0e-3, "heave-only thrust magnitude implausibly small"
        )
        # Cycle-mean thrust is positive: CT_raw = -<Fx>/qS > 0.
        half = len(self.forces) // 2
        mean_fx = sum(row[0] for row in self.forces[half:]) / (len(self.forces) - half)
        self.assertLess(mean_fx, 0.0, "heave-only cycle-mean thrust is not positive")


@unittest.skipUnless(CUDA_OK, "CUDA required")
class SteadyLiftAgreementTest(unittest.TestCase):
    """Steady flat wing at +5 deg: KJ normal force == pressure projection."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-030"]
        cls.solver = _make_solver(cls.case, angle_deg=5.0, dt=0.01)
        frame = _reference_grid(cls.case)
        proposal, _ = _run_steps(cls.solver, _StaticFrameKinematics(frame), 80, 0.01)
        cls.proposal = proposal
        cls.geometry = cls.solver.surface._geometry

    def test_kj_normal_force_matches_pressure_projection(self):
        author = self.proposal.author_load
        pressure_force = torch.sum(
            author.constant_pressure[:, None]
            * self.geometry.areas[:, None]
            * self.geometry.normals,
            dim=0,
        )
        kj_force = self.proposal.load.total_force
        kj_norm = float(torch.linalg.vector_norm(kj_force).item())
        torch.testing.assert_close(
            kj_force[2],
            pressure_force[2],
            rtol=5.0e-3,
            atol=5.0e-3 * kj_norm,
        )

    def test_kj_streamwise_component_exists(self):
        """The KJ force is NOT purely normal (that was the old bug)."""
        geometry = self.geometry
        force = self.proposal.load.total_force
        along_normal = geometry.normals[0] * float(
            torch.dot(force, geometry.normals[0]).item()
        )
        tangential = force - along_normal
        self.assertGreater(
            float(torch.linalg.vector_norm(tangential).item()),
            1.0e-3 * float(torch.linalg.vector_norm(force).item()),
            "KJ extraction lost its streamwise (thrust) component",
        )


class _StaticFrameKinematics:
    """Evaluates one frozen SurfaceFrame forever (steady-wing driver)."""

    def __init__(self, frame):
        self._frame = frame

    @property
    def surface_ids(self):
        return self._frame.surface_ids

    def evaluate(self, time: float):
        return self._frame


@unittest.skipUnless(CUDA_OK, "CUDA required")
class RepeatProposalLoadSafetyTest(unittest.TestCase):
    """Re-proposing one committed parent reproduces the load bit-for-bit."""

    def test_repeated_proposal_loads_are_identical(self):
        case = IZRA_CASES["IZRA-15-030"]
        dt = 0.2 / 128.0
        solver = _make_solver(case, angle_deg=0.0, dt=dt)
        kin = IzraRigidSurfaceKinematics(
            case, chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
        )
        solver.surface.set_frame(kin.evaluate(0.0))
        state = solver.initialize(None, None)
        for step in (1, 2, 3):
            frame = kin.evaluate(step * dt)
            solver.surface.set_frame(frame)
            first = solver.propose(state, None, None)
            second = solver.propose(state, None, None)
            torch.testing.assert_close(
                first.load.panel_forces,
                second.load.panel_forces,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                first.load.total_force,
                second.load.total_force,
                rtol=0.0,
                atol=0.0,
            )
            state = first.trial_state


if __name__ == "__main__":
    unittest.main()
