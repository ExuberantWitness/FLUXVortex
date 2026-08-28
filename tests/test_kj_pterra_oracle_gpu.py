"""Oracle: native KJ extraction vs the frozen Pterra formula on ONE V5M state.

The external review of Bug 3 (KJ force replacement) asked for a comparison
that the existing regressions (tests/test_rigid_kj_loads_gpu.py) do not
provide: those compare the KJ load against the pressure ledger *inside one
native proposal*.  They never exercise the frozen Pterra GPU oracle, the
full three-dimensional moment, or more than a couple of real phases.

This test closes that gap.  It drives the Izraelevitz production setup
(IZRA-15-030: heave h/c=0.6 + pitch 15 deg at psi=30 deg, 8x24 grid,
128 steps/cycle, the mandatory Fig.14 runner's exact solver settings) for
one full cycle to develop the wake/circulation state, then samples eight
phases of the next cycle (t = T + k*T/8, k = 0..7).  At every sampled phase
it compares three extractions of the SAME solver state:

1. native — ``RigidAuthorLoadAssembler``/``RigidNativeV5MSolver`` KJ loads
   (the refined filament-velocity path that owns the production wrench);
2. pterra — a standalone transcription of the frozen Pterra oracle
   (``platform/warp_vpm/bing_joint_ptera_gpu.py``,
   ``CudaJointLEVTEVSolver._calculate_loads``), applied to the same V5M
   rings/gamma/velocities/normals/areas;
3. pressure — the author pressure ledger projected on ``p A n`` (reported
   only; it carries no streamwise component by construction).

Pterra formula (frozen source, lines ~2103-2209 of the GPU module)
-----------------------------------------------------------------
Per panel, four Kutta-Joukowski filament forces with Pterra's effective
shared strengths on the (nc, ns) circulation grid::

    right[:, :-1] = 0.5 * (grid[:, :-1] - grid[:, 1:])   # full at tip j=ns-1
    front[1:, :]  = 0.5 * (grid[1:, :]  - grid[:-1, :])  # full at LE row i=0
    left[:, 1:]   = 0.5 * (grid[:, 1:]   - grid[:, :-1]) # full at tip j=0
    back[:-1, :]  = 0.5 * (grid[:-1, :] - grid[1:, :])   # full interior
    back[-1, :]   = grid[-1, :] - last[-1, :]             # shed TE row

    F_leg = rho * Gamma_eff * (v_solution + movement) x dl_leg
    movement = -(x_n - x_{n-1}) / dt          (grid-motion velocity, negated)
    F_unsteady = -rho * (Gamma - Gamma_last) * A * n / dt
    M = sum(r_leg x F_leg) + sum(r_panel x F_unsteady)

with ``v_solution`` evaluated at the four filament centres per panel.  This
IS the same KJ formula the native path implements: ``rho * Gamma_eff *
(v_sol - v_grid) x dl`` — Pterra enters the relative velocity as
``velocity + movement`` with ``movement = -(dx)/dt``, i.e. minus the grid
velocity.  Documented representation differences handled by the conversion
below:

* Pterasoftware rings are counter-clockwise about ``+n``; native V5M rings
  are clockwise.  The conversion reverses the corner order and negates the
  circulation, which leaves every filament's ``Gamma dl`` product (and so
  every KJ force) invariant while flipping the unsteady term's sign back to
  Pterra's frozen ``-rho dGamma/dt A n``.
* Pterra's movement is the backward difference of the leg-centre
  trajectory; the rigid path owns the exact analytic SurfaceFrame corner
  velocities, which are the continuous-time limit of the same quantity.
  Both variants are evaluated here (the analytic one is the asserted
  oracle; the discrete one is reported as the frozen-Pterra-faithful
  variant).
* Pterra transforms panel loads GP1->W before summing; the native V5M
  state is already expressed in the world/scientific frame, so the oracle
  uses the identity transform (rotation I, translation 0).
* Pterra additionally adds a free+bound vortex-impulse derivative when the
  legacy (non-node-ribbon) LEV mode is active.  The native V5M path keeps
  its LEV particles in the solver state (the node-ribbon analogue, where
  Pterra itself zeroes the impulse term and lets KJ + dGamma own the
  load), so the oracle compares the panel-force part only.

With those conversions the two extractions evaluate algebraically identical
expressions on identical tensors, so agreement is expected at floating-
point round-off; the task gates (5 percent Fx, 2 percent Fz, sign match)
then hold with orders of magnitude of margin, and the tight round-off gate
below additionally pins the formula equivalence itself.
"""
from __future__ import annotations

import math
import os
import unittest

import torch
import warp as wp

from fluxvortex.cases.izraelevitz2017 import IZRA_CASES
from fluxvortex.cases.izraelevitz_kinematics import IzraRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidAuthorLoad,
    RigidNativeV5MSolver,
    RigidV5MSurface,
)

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()

# The mandatory Fig.14 production setup (reproduce_izraelevitz2017_fig14_
# v5m_mandatory.py): 8x24 grid, 128 steps/cycle, static-polar LESP override,
# bounded wake-history scheme, production-scale LEV bank.
NC, NS = 8, 24
STEPS_PER_CYCLE = 128
SAMPLE_PHASES = 8
LESP_CRIT_STATIC_POLAR = 0.2393

# Task gates (Bug 3 review): thrust component 5 percent, lift 2 percent.
FX_RTOL = 5.0e-2
FZ_RTOL = 2.0e-2
# Same-formula equivalence gate: identical tensors through reordered float
# ops must agree far below any physical tolerance.
FORMULA_RTOL = 1.0e-9
MOMENT_RTOL = 1.0e-8


def _pterra_effective_strengths(
    grid: torch.Tensor, last_grid: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Pterra ``_effective_strengths`` transcribed verbatim (CCW rings).

    ``grid``/``last_grid`` are the current and previous bound circulations
    on the (nc, ns) panel grid, chord-major rows (row 0 = leading edge),
    span-minor columns — the reshape convention of the frozen source.
    """

    right = grid.clone()
    right[:, :-1] = 0.5 * (grid[:, :-1] - grid[:, 1:])
    front = grid.clone()
    front[1:, :] = 0.5 * (grid[1:, :] - grid[:-1, :])
    left = grid.clone()
    left[:, 1:] = 0.5 * (grid[:, 1:] - grid[:, :-1])
    back = grid.clone()
    back[:-1, :] = 0.5 * (grid[:-1, :] - grid[1:, :])
    # Step > 0 on every sampled phase (>= 128 committed steps): the freshly
    # shed Gamma_now - Gamma_last owns the trailing-edge back filaments.
    back[-1, :] = grid[-1, :] - last_grid[-1, :]
    return right, front, left, back


def pterra_calculate_loads_on_v5m_state(
    *,
    rings: torch.Tensor,
    ring_velocity: torch.Tensor,
    panel_centers: torch.Tensor,
    normals: torch.Tensor,
    areas: torch.Tensor,
    gamma: torch.Tensor,
    gamma_last: torch.Tensor,
    dt: float,
    rho: float,
    leg_solution_velocity: dict[str, torch.Tensor],
    nc: int,
    ns: int,
    leg_previous_centers: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen Pterra ``_calculate_loads`` applied to one native V5M state.

    Inputs are the raw native (clockwise-ring) V5M tensors of a single
    solver proposal: ``rings``/``ring_velocity`` (N, 4, 3), ``gamma`` and
    ``gamma_last`` (N,), ``normals``/``areas``/``panel_centers`` (N,3)/(N,)/
    (N,3), plus the full solution velocity at the four filament centres per
    panel keyed by leg name (front/right/back/left, each (N,3)).  Returns
    ``(panel_forces, total_force, total_moment)`` in the world frame, i.e.
    Pterra's GP1->W transform with identity rotation and zero translation.

    ``leg_previous_centers`` (same schema as the velocities) switches the
    movement term from the analytic corner velocities to Pterra's frozen
    backward difference ``-(x_n - x_{n-1}) / dt``.
    """

    count = nc * ns
    for name, value in (("rings", rings), ("ring_velocity", ring_velocity)):
        if tuple(value.shape) != (count, 4, 3):
            raise ValueError(f"{name} must have shape ({count},4,3)")
    if gamma.shape != (count,) or gamma_last.shape != (count,):
        raise ValueError("gamma/gamma_last must have shape (count,)")

    # CCW conversion: reverse the corner cycle (0,1,2,3 -> 0,3,2,1) and
    # negate the circulations.  Every filament Gamma x dl is unchanged.
    order = (0, 3, 2, 1)
    rings_ccw = rings[:, order, :]
    vel_ccw = ring_velocity[:, order, :]
    gamma_p = -gamma
    gamma_last_p = -gamma_last

    right, front, left, back = _pterra_effective_strengths(
        gamma_p.reshape(nc, ns), gamma_last_p.reshape(nc, ns)
    )

    # Legs of the counter-clockwise ring [c0, c3, c2, c1], paired with the
    # strengths by geometry: left = c0->c3 (chordwise, span station j),
    # back = c3->c2 (spanwise, trailing-edge side), right = c2->c1
    # (chordwise, station j+1), front = c1->c0 (spanwise, leading-edge side).
    leg_corners = {
        "left": (0, 1),
        "back": (1, 2),
        "right": (2, 3),
        "front": (3, 0),
    }
    strengths = {
        "right": right.reshape(-1),
        "front": front.reshape(-1),
        "left": left.reshape(-1),
        "back": back.reshape(-1),
    }
    panel_forces = torch.zeros((count, 3), device=rings.device, dtype=torch.float64)
    panel_moments = torch.zeros((count, 3), device=rings.device, dtype=torch.float64)
    # Pterra accumulates the four leg batches in (right, front, left, back)
    # order; keep that order so only genuine formula drift can differ.
    for name in ("right", "front", "left", "back"):
        a, b = leg_corners[name]
        center = 0.5 * (rings_ccw[:, a] + rings_ccw[:, b])
        vector = rings_ccw[:, b] - rings_ccw[:, a]
        if leg_previous_centers is None:
            movement = -(0.5 * (vel_ccw[:, a] + vel_ccw[:, b]))
        else:
            movement = -(center - leg_previous_centers[name]) / float(dt)
        velocity = leg_solution_velocity[name]
        if tuple(velocity.shape) != (count, 3):
            raise ValueError(f"leg_solution_velocity[{name}] must be (count,3)")
        force = (
            float(rho)
            * strengths[name].reshape(-1, 1)
            * torch.linalg.cross(velocity + movement, vector, dim=1)
        )
        panel_forces = panel_forces + force
        panel_moments = panel_moments + torch.linalg.cross(center, force, dim=1)

    # Unsteady-Bernoulli term with Pterra's frozen (CCW) sign.
    unsteady = -(
        float(rho)
        * (gamma_p - gamma_last_p).unsqueeze(1)
        * areas.unsqueeze(1)
        * normals
        / float(dt)
    )
    panel_forces = panel_forces + unsteady
    panel_moments = panel_moments + torch.linalg.cross(panel_centers, unsteady, dim=1)

    total_force = torch.sum(panel_forces, dim=0)
    total_moment = torch.sum(panel_moments, dim=0)
    return panel_forces, total_force, total_moment


def _build_production_solver(case):
    """IZRA production solver + kinematics (mandatory Fig.14 settings)."""

    period = 1.0 / float(case.frequency_hz)
    dt = period / STEPS_PER_CYCLE
    surface = RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=DEVICE)
    settings = NativeV5MConfig(
        chordwise_panels=NC,
        spanwise_panels=NS,
        density=case.rho_kg_m3,
        freestream=case.freestream_m_s,
        aerodynamic_dt=dt,
        wake_max_rows=128,
        particle_capacity=131_072,
        particle_max_age_steps=64,
        dvm_target_spacing_chord=0.018,
        lesp_crit_override=LESP_CRIT_STATIC_POLAR,
        wake_history_mode="bound_rate",
        device=DEVICE,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    kin = IzraRigidSurfaceKinematics(
        case, chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
    )
    return solver, kin, dt


def _leg_midpoints_and_velocity(solver, proposal, gamma):
    """Filament centres and the exact solution velocity the refinement used.

    Rebuilt independently from the trial state: freestream + bound sheet
    (Gamma_n) + pre-shed wake (this step's TEV row skipped) + LEV
    particles, at the four filament centres per panel — the same inputs
    ``RigidNativeV5MSolver._refine_load_with_filament_velocity`` feeds its
    own KJ re-integration.
    """

    trial = proposal.trial_state
    geometry = solver.surface.evaluate(None, None)
    ns = solver.settings.spanwise_panels
    mids = []
    for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):  # front right back left
        mids.append(0.5 * (geometry.rings[:, start] + geometry.rings[:, end]))
    mids_flat = torch.cat(mids, dim=0)
    velocity = solver.v_inf[None, :].expand(mids_flat.shape[0], 3) + (
        solver._ring_velocity(mids_flat, geometry.rings, gamma, rough=False)
    )
    if trial.wake_rings.shape[0] > ns:
        velocity = velocity + solver._ring_velocity(
            mids_flat, trial.wake_rings[ns:], trial.wake_gamma[ns:], rough=False
        )
    if trial.particle_field.n:
        velocity = velocity + trial.particle_field.velocity_at_cuda(mids_flat)
    count = mids_flat.shape[0] // 4
    legs = {
        "front": velocity[0:count],
        "right": velocity[count : 2 * count],
        "back": velocity[2 * count : 3 * count],
        "left": velocity[3 * count : 4 * count],
    }
    return geometry, legs


def _previous_leg_centers(kin, t_prev):
    """Pterra's previous leg centres from the analytic frame at ``t_prev``."""

    frame = kin.evaluate(t_prev)
    rings = frame.panel_rings_I
    order = (0, 3, 2, 1)
    rings_ccw = rings[:, order, :]
    leg_corners = {
        "left": (0, 1),
        "back": (1, 2),
        "right": (2, 3),
        "front": (3, 0),
    }
    return {
        name: 0.5 * (rings_ccw[:, a] + rings_ccw[:, b])
        for name, (a, b) in leg_corners.items()
    }


@unittest.skipUnless(CUDA_OK, "CUDA required")
class KJVersusPterraOracleTest(unittest.TestCase):
    """Native KJ vs frozen Pterra formula on one V5M state at 8 real phases."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-030"]
        cls.solver, cls.kin, cls.dt = _build_production_solver(cls.case)
        cls.period = cls.dt * STEPS_PER_CYCLE
        cls.q_area = (
            0.5
            * cls.case.rho_kg_m3
            * cls.case.freestream_m_s**2
            * cls.case.area_m2
        )
        cls.samples = []
        stride = STEPS_PER_CYCLE // SAMPLE_PHASES
        cls.solver.surface.set_frame(cls.kin.evaluate(0.0))
        state = cls.solver.initialize(None, None)
        # Sample the 8 phases of the SECOND cycle: steps 128, 144, ..., 240
        # (t = T + k*T/8, k = 0..7) after the one-cycle wake spin-up.
        total_steps = STEPS_PER_CYCLE + (SAMPLE_PHASES - 1) * stride
        for step in range(1, total_steps + 1):
            cls.solver.surface.set_frame(cls.kin.evaluate(step * cls.dt))
            proposal = cls.solver.propose(state, None, None)
            state = proposal.trial_state
            if step >= STEPS_PER_CYCLE and step % stride == 0:
                cls.samples.append(
                    cls._extract_sample(proposal, step, step // stride - 8)
                )

    @classmethod
    def _extract_sample(cls, proposal, step, phase_index):
        solver = cls.solver
        rho = solver.settings.density
        dt = cls.dt
        nc, ns = solver.settings.chordwise_panels, solver.settings.spanwise_panels
        author = proposal.author_load
        if not isinstance(author, RigidAuthorLoad):
            raise TypeError("the rigid native solver did not publish a RigidAuthorLoad")
        gamma = proposal.trial_state.gamma_bound.detach().clone()
        # bound_rate identity: the author ledger's wake-history pressure is
        # rho * dGamma/dt of this proposal, so Gamma_last is exact.
        gamma_rate = author.wake_history_pressure / rho
        gamma_last = gamma - gamma_rate * dt

        geometry, legs = _leg_midpoints_and_velocity(solver, proposal, gamma)
        oracle_kwargs = dict(
            rings=geometry.rings,
            ring_velocity=geometry.ring_velocity,
            panel_centers=geometry.collocation,
            normals=geometry.normals,
            areas=geometry.areas,
            gamma=gamma,
            gamma_last=gamma_last,
            dt=dt,
            rho=rho,
            leg_solution_velocity=legs,
            nc=nc,
            ns=ns,
        )
        _, force_p, moment_p = pterra_calculate_loads_on_v5m_state(**oracle_kwargs)
        prev_centers = _previous_leg_centers(cls.kin, (step - 1) * dt)
        _, force_pd, moment_pd = pterra_calculate_loads_on_v5m_state(
            **oracle_kwargs, leg_previous_centers=prev_centers
        )
        pressure_force = torch.sum(
            author.constant_pressure[:, None]
            * geometry.areas[:, None]
            * geometry.normals,
            dim=0,
        )
        return {
            "step": step,
            "phase_index": phase_index,
            "phase_deg": 360.0 * phase_index / SAMPLE_PHASES,
            "particles": int(proposal.trial_state.particle_field.n),
            "wake_rows": int(proposal.trial_state.wake_gamma.numel() // ns),
            "native_force": author.total_force.detach().clone(),
            "native_moment": author.total_moment.detach().clone(),
            "pterra_force": force_p,
            "pterra_moment": moment_p,
            "pterra_force_discrete": force_pd,
            "pterra_moment_discrete": moment_pd,
            "pressure_force": pressure_force,
        }

    def _relative(self, delta: float, reference: float, floor: float) -> float:
        return abs(delta) / max(abs(reference), floor)

    def _assert_component(
        self, sample, native, pterra, name, rtol, floor, tight=True
    ):
        scale_note = (
            f"phase {sample['phase_index']} ({sample['phase_deg']:.0f} deg, "
            f"step {sample['step']})"
        )
        for axis, label in enumerate(("x", "y", "z")):
            value = float(native[axis].item())
            oracle = float(pterra[axis].item())
            self.assertTrue(
                math.isfinite(value) and math.isfinite(oracle),
                f"{name}{label} non-finite at {scale_note}",
            )
            rel = self._relative(value - oracle, oracle, floor)
            self.assertLess(
                rel,
                rtol,
                f"{name}{label} relative difference {rel:.3e} >= {rtol:.1e} "
                f"at {scale_note}: native={value:.9g} pterra={oracle:.9g}",
            )
            if tight and abs(oracle) > floor:
                self.assertLess(
                    rel,
                    FORMULA_RTOL if name == "F" else MOMENT_RTOL,
                    f"{name}{label} exceeds the same-formula round-off gate "
                    f"({rel:.3e}): native={value:.9g} pterra={oracle:.9g}",
                )
            if abs(oracle) > floor:
                self.assertTrue(
                    (value > 0.0) == (oracle > 0.0),
                    f"{name}{label} sign mismatch at {scale_note}: "
                    f"native={value:.9g} pterra={oracle:.9g}",
                )

    def test_forces_agree_with_pterra_formula_at_every_phase(self):
        """Fx within 5 percent, Fz within 2 percent, signs equal — 8 phases."""
        floor = 1.0e-6 * self.q_area
        for sample in self.samples:
            with self.subTest(phase=sample["phase_index"]):
                self._assert_component(
                    sample,
                    sample["native_force"],
                    sample["pterra_force"],
                    "F",
                    FX_RTOL,
                    floor,
                )
                fz_rel = self._relative(
                    float(sample["native_force"][2].item())
                    - float(sample["pterra_force"][2].item()),
                    float(sample["pterra_force"][2].item()),
                    floor,
                )
                self.assertLess(
                    fz_rel,
                    FZ_RTOL,
                    f"Fz relative difference {fz_rel:.3e} >= {FZ_RTOL:.1e} at "
                    f"phase {sample['phase_index']}",
                )

    def test_full_three_dimensional_moments_agree(self):
        """Mx/My/Mz of both extractions agree to round-off at every phase."""
        floor = 1.0e-6 * self.q_area * self.case.chord_m
        for sample in self.samples:
            with self.subTest(phase=sample["phase_index"]):
                self._assert_component(
                    sample,
                    sample["native_moment"],
                    sample["pterra_moment"],
                    "M",
                    2.0e-2,
                    floor,
                )

    def test_all_extractions_finite_and_report_table(self):
        """Finiteness of every extraction plus the per-phase agreement table."""

        floor_f = 1.0e-6 * self.q_area
        for sample in self.samples:
            for key in (
                "native_force",
                "native_moment",
                "pterra_force",
                "pterra_moment",
                "pterra_force_discrete",
                "pterra_moment_discrete",
                "pressure_force",
            ):
                self.assertTrue(
                    bool(torch.isfinite(sample[key]).all().item()),
                    f"{key} non-finite at phase {sample['phase_index']}",
                )
        lines = [
            "KJ vs frozen-Pterra oracle on the IZRA-15-030 V5M state "
            f"(qS={self.q_area:.3f} N, phases of cycle 2):",
            "  ph  step  part   Fx          Fy          Fz          "
            "| Mx         My         Mz",
        ]
        for s in self.samples:
            for tag, fkey, mkey in (
                ("native", "native_force", "native_moment"),
                ("pterra", "pterra_force", "pterra_moment"),
                ("ptrDsc", "pterra_force_discrete", "pterra_moment_discrete"),
                ("press", "pressure_force", None),
            ):
                f = [float(v) for v in s[fkey]]
                m = [float(v) for v in s[mkey]] if mkey else [0.0, 0.0, 0.0]
                lines.append(
                    f"  {s['phase_index']}  {s['step']:4d}  {tag}  "
                    + "  ".join(f"{v:+10.4f}" for v in f)
                    + " | "
                    + " ".join(f"{v:+9.4f}" for v in m)
                )
            nfx = float(s["native_force"][0])
            pfx = float(s["pterra_force"][0])
            nfz = float(s["native_force"][2])
            pfz = float(s["pterra_force"][2])
            lines.append(
                f"    relFx={self._relative(nfx - pfx, pfx, floor_f):.3e}  "
                f"relFz={self._relative(nfz - pfz, pfz, floor_f):.3e}  "
                f"(particles={s['particles']}, wake_rows={s['wake_rows']})"
            )
        table = "\n".join(lines)
        print(table)
        # The pressure projection must differ from the KJ thrust at the
        # loaded phases (it has no streamwise component to represent) — the
        # original Bug 3 finding, kept visible in the report.
        max_fx_gap = max(
            abs(float(s["native_force"][0]) - float(s["pressure_force"][0]))
            for s in self.samples
        )
        self.assertGreater(
            max_fx_gap, 1.0e-3 * self.q_area,
            "pressure projection unexpectedly reproduces the KJ thrust",
        )


if __name__ == "__main__":
    unittest.main()
