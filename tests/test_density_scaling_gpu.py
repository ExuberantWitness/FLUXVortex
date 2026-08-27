"""Regression: every author pressure term must carry the fluid density.

External review (2026-08-27) confirmed an implementation-contract bug: the
velocity part of the author load — lift2 and Mf2_1 — was assembled WITHOUT the
density in both native paths:

- ``warp_fsi/rigid_flux_v5m_native.py`` summed a bare ``-V . grad(Gamma)``
  (m^2/s^2) into the Pa total of lift1 + Mf2;
- ``warp_fsi/q16_flux_v5m_author_loads.py`` ``velocity_force()`` did the same
  for the Q16 FSI substep loads.

The frozen reference paths multiply every term by rho (``warp_fsi/coupled.py``
``dp_lift1_flat_kernel`` bakes rho into dp/dp2, and
``platform/warp_vpm/q16_real_fsi_coupling.py`` multiplies both the lift2
operator and the Mf2_1 solve by rho).

These tests run one propose()/assemble() at rho = 1.0 and rho = 1000.0 and
require EVERY pressure component and every force/wrench output to scale
exactly linearly with rho.  With the bug, lift2 and Mf2_1 do not scale at all
and the totals come out off by factors of ~rho; these gates fail permanently
if the density ever drops out of any term again.
"""
from __future__ import annotations

import os
import unittest
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.cases.izraelevitz2017 import IZRA_CASES
from fluxvortex.cases.izraelevitz_kinematics import IzraRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
    assemble_native_ring_geometry,
    native_aic,
)
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidAuthorLoadAssembler,
    RigidNativeV5MSolver,
    RigidV5MSurface,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    make_yamano2020_q16_model,
)

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
RHO_LOW, RHO_HIGH, RATIO = 1.0, 1000.0, 1000.0
CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()


def _synthetic_geometry(
    nc: int, ns: int, seed: int
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """A flat nc x ns native grid with seeded random material velocities."""

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)
    device = torch.device(DEVICE)
    chord, span = 1.0, 1.2
    xs = torch.linspace(0.0, chord, nc + 1, device=device, dtype=torch.float64)
    ys = torch.linspace(0.0, span, ns + 1, device=device, dtype=torch.float64)
    # Quarter lines at (i + 0.25) dx and the trailing edge at x = chord.
    quarter_x = xs[:-1] + 0.25 * (xs[1] - xs[0])
    quarter = torch.cartesian_prod(quarter_x, ys)
    quarter = torch.stack(
        (quarter[:, 0], quarter[:, 1], torch.zeros_like(quarter[:, 0])), dim=1
    ).reshape(nc, ns + 1, 3)
    trailing = torch.stack(
        (torch.full_like(ys, chord), ys, torch.zeros_like(ys)), dim=1
    )
    quarter_velocity = 0.3 * torch.randn(
        (nc, ns + 1, 3), generator=generator, device=device, dtype=torch.float64
    )
    trailing_velocity = 0.3 * torch.randn(
        (ns + 1, 3), generator=generator, device=device, dtype=torch.float64
    )
    rings, ring_velocity, collocation, collocation_velocity, normals, areas = (
        assemble_native_ring_geometry(
            quarter,
            quarter_velocity,
            trailing,
            trailing_velocity,
            panel_count=nc * ns,
        )
    )
    from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MGeometry

    geometry = NativeV5MGeometry(
        rings=rings,
        ring_velocity=ring_velocity,
        collocation=collocation,
        collocation_velocity=collocation_velocity,
        normals=normals,
        areas=areas,
        leading_edge=torch.stack(
            (
                torch.zeros_like(ys),
                ys,
                torch.zeros_like(ys),
            ),
            dim=1,
        ),
        trailing_edge=trailing,
        leading_velocity=trailing_velocity.clone(),
        trailing_velocity=trailing_velocity,
        quarter_points=quarter,
    )
    panel_count = nc * ns
    fields = {
        "gamma": 0.7 * torch.randn(
            (panel_count,), generator=generator, device=device, dtype=torch.float64
        ),
        "external_flow": 0.5
        + 0.2 * torch.randn(
            (panel_count, 3), generator=generator, device=device, dtype=torch.float64
        ),
        "gamma_gradient": 0.4 * torch.randn(
            (panel_count, 3), generator=generator, device=device, dtype=torch.float64
        ),
        "mf2_history": 0.6 * torch.randn(
            (panel_count,), generator=generator, device=device, dtype=torch.float64
        ),
    }
    return geometry, fields["gamma"], fields["external_flow"], fields


def _assemble_at_density(density: float, nc: int = 4, ns: int = 3, seed: int = 20260827):
    geometry, gamma, external_flow, fields = _synthetic_geometry(nc, ns, seed)
    aic = native_aic(geometry, chordwise_panels=nc)
    assembler = RigidAuthorLoadAssembler(density=density, device=DEVICE)
    return assembler.assemble(
        structural_state=None,
        geometry=geometry,
        aic=aic,
        gamma=gamma,
        external_flow=external_flow,
        gamma_gradient=fields["gamma_gradient"],
        mf2_history=fields["mf2_history"],
    )


@unittest.skipUnless(CUDA_OK, "CUDA required")
class RigidAssemblerDensityScalingTest(unittest.TestCase):
    """Every one of the four author pressure parts scales exactly with rho."""

    def test_all_four_pressure_parts_scale_exactly_linearly(self):
        low = _assemble_at_density(RHO_LOW)
        high = _assemble_at_density(RHO_HIGH)
        # Each part is a single rho multiply, so scaling is bitwise exact.
        for name in ("lift1_pressure", "wake_history_pressure"):
            self.assertTrue(
                torch.equal(RATIO * getattr(low, name), getattr(high, name)),
                f"{name} does not scale exactly with rho",
            )
        # The two terms that were missing rho before the fix: they must now
        # scale exactly AND be nonzero so the gate actually discriminates.
        for name in ("lift2_pressure", "mf2_1_pressure"):
            part_low = getattr(low, name)
            part_high = getattr(high, name)
            self.assertGreater(
                float(part_low.abs().max().item()),
                1.0e-9,
                f"{name} is zero: the scaling gate would be vacuous",
            )
            self.assertTrue(
                torch.equal(RATIO * part_low, part_high),
                f"{name} does not scale exactly with rho",
            )
        # The summed pressure and integrated forces are linear up to the
        # roundoff of the rho-scaled additions.
        for name, tol in (
            ("constant_pressure", 1.0e-12),
            ("panel_forces", 1.0e-12),
            ("total_force", 1.0e-12),
            ("total_moment", 1.0e-12),
        ):
            torch.testing.assert_close(
                RATIO * getattr(low, name),
                getattr(high, name),
                rtol=tol,
                atol=1.0e-12 * float(getattr(high, name).abs().max().item()),
            )
        wrench_low = wp.to_torch(low.constant_generalized_force)
        wrench_high = wp.to_torch(high.constant_generalized_force)
        torch.testing.assert_close(
            RATIO * wrench_low,
            wrench_high,
            rtol=1.0e-12,
            atol=1.0e-12 * float(wrench_high.abs().max().item()),
        )


@unittest.skipUnless(CUDA_OK, "CUDA required")
class RigidProposeDensityScalingTest(unittest.TestCase):
    """One full rigid propose() step: all four parts + wrench scale with rho."""

    NC, NS = 8, 24

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-030"]
        cls.kin = IzraRigidSurfaceKinematics(
            cls.case, chordwise_panels=cls.NC, spanwise_panels=cls.NS, device=DEVICE
        )

    def _propose_at_density(self, density: float):
        surface = RigidV5MSurface(
            chordwise_panels=self.NC, spanwise_panels=self.NS, device=DEVICE
        )
        solver = RigidNativeV5MSolver(
            surface,
            NativeV5MConfig(
                chordwise_panels=self.NC,
                spanwise_panels=self.NS,
                density=density,
                freestream=self.case.freestream_m_s,
                aerodynamic_dt=0.2 / 128.0,
                wake_max_rows=128,
                particle_capacity=32768,
                particle_max_age_steps=100,
                dvm_target_spacing_chord=0.018,
                device=DEVICE,
            ),
        )
        dt = 0.2 / 128.0
        surface.set_frame(self.kin.evaluate(0.0))
        committed = solver.initialize(None, None)
        # Two steps: the material wake-history term is exactly zero on the
        # first step (no wake has been shed yet), so compare the SECOND step
        # where all four pressure parts are live.
        for step in (1, 2):
            frame = self.kin.evaluate(step * dt)
            surface.set_frame(frame)
            proposal = solver.propose(committed, None, None)
            committed = proposal.trial_state
        return proposal

    def test_propose_outputs_scale_linearly_with_density(self):
        low = self._propose_at_density(RHO_LOW)
        high = self._propose_at_density(RHO_HIGH)
        # The aero state is density-free: identical digests prove the two runs
        # solved the same circulation, so any load mismatch is a load bug.
        self.assertEqual(
            low.trial_state.digest(), high.trial_state.digest()
        )
        parts = (
            "lift1_pressure",
            "wake_history_pressure",
            "lift2_pressure",
            "mf2_1_pressure",
        )
        for name in parts:
            part_low = getattr(low.author_load, name)
            part_high = getattr(high.author_load, name)
            self.assertGreater(
                float(part_low.abs().max().item()),
                1.0e-9,
                f"{name} is zero: the scaling gate would be vacuous",
            )
            self.assertTrue(
                torch.equal(RATIO * part_low, part_high),
                f"{name} does not scale exactly with rho",
            )
        for name in ("constant_pressure", "panel_forces"):
            torch.testing.assert_close(
                RATIO * getattr(low.author_load, name),
                getattr(high.author_load, name),
                rtol=1.0e-12,
                atol=1.0e-12 * float(getattr(high.author_load, name).abs().max().item()),
            )
        wrench_low = wp.to_torch(low.generalized_force)
        wrench_high = wp.to_torch(high.generalized_force)
        torch.testing.assert_close(
            RATIO * wrench_low,
            wrench_high,
            rtol=1.0e-12,
            atol=1.0e-12 * float(wrench_high.abs().max().item()),
        )
        # The total pressure is exactly the four parts summed (no term left
        # out of the rho scaling by accident).
        for load in (low.author_load, high.author_load):
            torch.testing.assert_close(
                load.constant_pressure,
                load.lift1_pressure
                + load.wake_history_pressure
                + load.lift2_pressure
                + load.mf2_1_pressure,
                rtol=1.0e-12,
                atol=1.0e-12,
            )


@unittest.skipUnless(CUDA_OK, "CUDA required")
class Q16VelocityForceDensityScalingTest(unittest.TestCase):
    """The Q16 native endpoint ``velocity_force()`` (lift2 + Mf2_1 for the FSI
    substep) must scale exactly linearly with rho, like the constant part."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, _, _ = make_yamano2020_q16_model(
            chordwise_element_count=5, spanwise_element_count=3
        )
        cls.state_wp = wp.array(
            np.ascontiguousarray(cls.mesh.reference_state[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        # Rigid pitch rate 0.5 rad/s about +y (vz = -0.5 x) plus 1 m/s surge:
        # pure heave gives exactly zero lift2 (velocity in-plane-dotted) and
        # zero Mf2_1 (no normal rotation, no relative ring motion), so a
        # velocity field with BOTH in-plane and rotational components is
        # required for a discriminating gate.
        velocity = np.zeros_like(cls.mesh.reference_state)
        velocity.reshape(-1, 6)[:, 0] = 1.0
        velocity.reshape(-1, 6)[:, 2] = -0.5 * cls.mesh.reference_rows[:, 0]
        cls.velocity_wp = wp.array(
            np.ascontiguousarray(velocity[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )

    def _propose_at_density(self, density: float):
        surface = Q16NativeV5MSurface(
            self.mesh,
            q16_chordwise_elements=5,
            q16_spanwise_elements=3,
            aerodynamic_chordwise_panels=15,
            aerodynamic_spanwise_panels=10,
            device=config.DEVICE,
        )
        solver = Q16NativeV5MSolver(
            surface,
            NativeV5MConfig(
                density=density,
                freestream=YAMANO_2020_SINGLE_SHEET.freestream_m_s,
                aerodynamic_dt=YAMANO_2020_SINGLE_SHEET.aerodynamic_dt_s,
                freestream_angle_deg=5.0,
                device=config.DEVICE,
            ),
        )
        committed = solver.initialize(self.state_wp, self.velocity_wp)
        proposal = solver.propose(committed, self.state_wp, self.velocity_wp)
        velocity_force = wp.to_torch(
            proposal.author_load.velocity_force(self.velocity_wp)
        )[0]
        return proposal, velocity_force

    def test_velocity_force_and_constant_part_scale_with_density(self):
        low, vf_low = self._propose_at_density(RHO_LOW)
        high, vf_high = self._propose_at_density(RHO_HIGH)
        self.assertEqual(low.trial_state.digest(), high.trial_state.digest())
        self.assertGreater(float(vf_low.abs().max().item()), 1.0e-9)
        torch.testing.assert_close(
            RATIO * vf_low,
            vf_high,
            rtol=1.0e-12,
            atol=1.0e-12 * float(vf_high.abs().max().item()),
            msg="velocity_force (lift2 + Mf2_1) does not scale with rho",
        )
        # The constant part (lift1 + Mf2 history) already carried rho and
        # must stay exactly linear (single multiply per term).
        self.assertTrue(
            torch.equal(
                RATIO * low.author_load.dp_lift1, high.author_load.dp_lift1
            )
        )
        self.assertTrue(
            torch.equal(
                RATIO * low.author_load.constant_pressure,
                high.author_load.constant_pressure,
            )
        )


if __name__ == "__main__":
    unittest.main()
