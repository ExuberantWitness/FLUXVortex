"""H2/H3: full-wing Izraelevitz SurfaceFrame + rigid native V5M direct drive.

H2 (HANDOFF_IZRAELEVITZ2017_FIG14_SCHERER1968_20260827 §8): the analytic
rigid kinematics for the flat rectangular AR=3 wing — reference grid in the
native V5M ring convention (origin at the x/c=0.75 pitch axis, full span
-b/2..+b/2 with two free tips), exact heave/pitch motion law, and the
complete SurfaceFrame with analytic velocities.

H3: the rigid SurfaceFrame drives the native V5M solver directly —
RigidV5MSurface + RigidNativeV5MSolver + RigidV5MStepper — with no Q16
structural-state bridge anywhere on the path.

These are CUDA float64 interface/contract gates; the 12-condition Figure 14
science run is H4/H5 work and must not be replaced by this file.
"""
from __future__ import annotations

import math
import os
import unittest

import torch
import warp as wp

from fluxvortex.aero.protocol import AerodynamicStepper
from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz2017 import IZRA_CASES
from fluxvortex.cases.izraelevitz_kinematics import (
    IzraRigidSurfaceKinematics,
    build_izra_reference_grid,
    izra_motion_law,
)
from fluxvortex.coupling.one_way import OneWayPrescribedCoupling
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.state.world import WorldDynamicState, WorldOwner
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidAuthorLoadAssembler,
    RigidNativeV5MSolver,
    RigidV5MSurface,
)
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
NC, NS = 8, 24  # frozen formal grid: 8 chordwise, 24 full-span panels
AERO_DT = 0.2 / 128.0  # 128 steps per 5 Hz cycle
STEPS = 3

CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()


@unittest.skipUnless(CUDA_OK, "CUDA required")
class IzraReferenceGridTest(unittest.TestCase):
    """H2: the undeformed full-wing reference SurfaceFrame."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-015"]
        cls.frame = build_izra_reference_grid(
            chordwise_panels=NC,
            spanwise_panels=NS,
            chord_m=cls.case.chord_m,
            span_m=cls.case.span_m,
            pivot_fraction_chord=cls.case.pivot_fraction_chord,
            device=DEVICE,
        )

    def test_shapes_and_topology(self):
        frame = self.frame
        self.assertIsInstance(frame, SurfaceFrame)
        self.assertEqual(frame.chordwise_panels, NC)
        self.assertEqual(frame.spanwise_panels, NS)
        self.assertEqual(tuple(frame.panel_rings_I.shape), (NC * NS, 4, 3))
        self.assertEqual(tuple(frame.panel_ring_velocity_I.shape), (NC * NS, 4, 3))
        self.assertEqual(tuple(frame.collocation_I.shape), (NC * NS, 3))
        self.assertEqual(tuple(frame.collocation_velocity_I.shape), (NC * NS, 3))
        self.assertEqual(tuple(frame.normals_I.shape), (NC * NS, 3))
        self.assertEqual(tuple(frame.areas.shape), (NC * NS,))
        # ns+1 spanwise LE/TE nodes (full span, two free tips).
        self.assertEqual(tuple(frame.leading_edge_I.shape), (NS + 1, 3))
        self.assertEqual(tuple(frame.trailing_edge_I.shape), (NS + 1, 3))
        self.assertEqual(tuple(frame.leading_velocity_I.shape), (NS + 1, 3))
        self.assertEqual(tuple(frame.trailing_velocity_I.shape), (NS + 1, 3))
        for tensor in (
            frame.panel_rings_I,
            frame.panel_ring_velocity_I,
            frame.collocation_I,
            frame.collocation_velocity_I,
            frame.normals_I,
            frame.areas,
            frame.leading_edge_I,
            frame.trailing_edge_I,
        ):
            self.assertIs(tensor.dtype, torch.float64)
            self.assertEqual(tensor.device.type, "cuda")
        # The undeformed reference is static.
        self.assertTrue(bool(torch.count_nonzero(frame.panel_ring_velocity_I).item() == 0))
        self.assertTrue(bool(torch.count_nonzero(frame.leading_velocity_I).item() == 0))

    def test_pitch_axis_origin_and_full_span(self):
        c = self.case.chord_m
        b = self.case.span_m
        device = self.frame.leading_edge_I.device
        # Origin at the x/c=0.75 pivot: LE at -0.75c, TE at +0.25c.
        self.assertTrue(
            torch.allclose(
                self.frame.leading_edge_I[:, 0],
                torch.full((NS + 1,), -0.75 * c, device=device, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                self.frame.trailing_edge_I[:, 0],
                torch.full((NS + 1,), 0.25 * c, device=device, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.allclose(
                self.frame.leading_edge_I[:, 2],
                torch.zeros(NS + 1, device=device, dtype=torch.float64),
            )
        )
        # Full span from -b/2 to +b/2, both free tips present.
        span_y = self.frame.leading_edge_I[:, 1]
        self.assertAlmostEqual(float(span_y[0]), -0.5 * b, places=15)
        self.assertAlmostEqual(float(span_y[-1]), +0.5 * b, places=15)
        self.assertAlmostEqual(float(span_y[1] - span_y[0]), b / NS, places=15)

    def test_native_v5m_ring_convention(self):
        """Rings follow the shared native layout: quarter lines at
        (i+0.25)dx, rear row one panel behind the last quarter line, panel
        index i*ns+j, +z normals, exact dx*dy rectangles."""
        c = self.case.chord_m
        b = self.case.span_m
        dx, dy = c / NC, b / NS
        device = self.frame.panel_rings_I.device
        rings = self.frame.panel_rings_I.reshape(NC, NS, 4, 3)
        # Quarter-line front edges.
        for i in (0, NC // 2, NC - 1):
            front_x = (i + 0.25) * dx - 0.75 * c
            self.assertTrue(
                torch.allclose(
                    rings[i, :, 0, 0],
                    torch.full((NS,), front_x, device=device, dtype=torch.float64),
                )
            )
            self.assertTrue(
                torch.allclose(
                    rings[i, :, 1, 0],
                    torch.full((NS,), front_x, device=device, dtype=torch.float64),
                )
            )
        # Back edges sit one chordwise panel behind the front edge, including
        # the rear row past the TE (rear = TE + (4/3)(TE - quarter_last)).
        for i in (0, NC - 1):
            back_x = (i + 1.25) * dx - 0.75 * c
            self.assertTrue(
                torch.allclose(
                    rings[i, :, 2, 0],
                    torch.full((NS,), back_x, device=device, dtype=torch.float64),
                )
            )
            self.assertTrue(
                torch.allclose(
                    rings[i, :, 3, 0],
                    torch.full((NS,), back_x, device=device, dtype=torch.float64),
                )
            )
        # Collocation = ring corner mean.
        torch.testing.assert_close(
            self.frame.collocation_I,
            self.frame.panel_rings_I.mean(dim=1),
            rtol=0.0,
            atol=0.0,
        )
        # Flat plate: unit normals are exactly +z and areas exactly dx*dy.
        torch.testing.assert_close(
            self.frame.normals_I,
            torch.tile(
                torch.tensor(
                    [0.0, 0.0, 1.0], device=device, dtype=torch.float64
                ),
                (NC * NS, 1),
            ),
            rtol=0.0,
            atol=1e-15,
        )
        torch.testing.assert_close(
            self.frame.areas,
            torch.full(
                (NC * NS,), dx * dy, device=device, dtype=torch.float64
            ),
            rtol=0.0,
            atol=1e-15,
        )
        # Planform totals to the full-wing reference area b*c.
        self.assertAlmostEqual(
            float(self.frame.areas.sum().item()), b * c, places=14
        )


@unittest.skipUnless(CUDA_OK, "CUDA required")
class IzraMotionLawTest(unittest.TestCase):
    """H2: the exact analytic motion law and its derivatives."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-030"]
        cls.law = staticmethod(izra_motion_law(cls.case, device=DEVICE))

    def _law_values(self, t):
        pos, quat, lin_vel, ang_vel = self.law(t)
        return pos, quat, lin_vel, ang_vel

    def test_t0_extreme_positions_zero_velocities(self):
        case = self.case
        psi = math.radians(case.phase_offset_deg)
        theta0 = math.radians(case.theta_max_deg) * math.cos(psi)
        thetadot0 = -math.radians(case.theta_max_deg) * case.omega_rad_s * math.sin(psi)
        pos, quat, lin_vel, ang_vel = self._law_values(0.0)
        self.assertAlmostEqual(float(pos[0]), 0.0, places=15)
        self.assertAlmostEqual(float(pos[1]), 0.0, places=15)
        self.assertAlmostEqual(float(pos[2]), case.heave_amplitude_m, places=12)
        # theta(0) = theta_max cos(psi): rotation quaternion about +y.
        half = 0.5 * theta0
        torch.testing.assert_close(
            quat,
            torch.tensor(
                [math.cos(half), 0.0, math.sin(half), 0.0],
                device=quat.device,
                dtype=torch.float64,
            ),
            rtol=0.0,
            atol=1e-14,
        )
        self.assertAlmostEqual(float(torch.linalg.vector_norm(quat)), 1.0, places=14)
        # Heave peaks at t=0 (no phase), so zdot(0) = 0; the pitch law carries
        # psi, so thetadot(0) = -theta_max omega sin(psi).
        self.assertTrue(bool(torch.count_nonzero(lin_vel).item() == 0))
        self.assertAlmostEqual(float(ang_vel[0]), 0.0, places=15)
        self.assertAlmostEqual(float(ang_vel[1]), thetadot0, places=9)
        self.assertAlmostEqual(float(ang_vel[2]), 0.0, places=15)

    def test_quarter_period_analytic_values(self):
        case = self.case
        period = 1.0 / case.frequency_hz
        t = 0.25 * period
        psi = math.radians(case.phase_offset_deg)
        theta_max = math.radians(case.theta_max_deg)
        omega = case.omega_rad_s
        pos, quat, lin_vel, ang_vel = self._law_values(t)
        # z(T/4) = 0, zdot = -h omega; theta = theta_max cos(pi/2 + psi) =
        # -theta_max sin(psi); thetadot = -theta_max omega cos(psi).
        self.assertAlmostEqual(float(pos[2]), 0.0, places=12)
        self.assertAlmostEqual(float(lin_vel[2]), -case.heave_amplitude_m * omega, places=9)
        theta = 2.0 * math.atan2(float(quat[2]), float(quat[0]))
        self.assertAlmostEqual(theta, -theta_max * math.sin(psi), places=12)
        self.assertAlmostEqual(float(ang_vel[1]), -theta_max * omega * math.cos(psi), places=9)

    def test_velocity_is_analytic_derivative(self):
        case = self.case
        h = 1.0e-6
        t0 = 0.0371
        pos, quat, lin_vel, ang_vel = self._law_values(t0)

        def z(t):
            return float(self._law_values(t)[0][2])

        def theta(t):
            q = self._law_values(t)[1]
            return 2.0 * math.atan2(float(q[2]), float(q[0]))

        numeric_zdot = (z(t0 + h) - z(t0 - h)) / (2.0 * h)
        self.assertAlmostEqual(float(lin_vel[2]), numeric_zdot, places=6)
        numeric_thetadot = (theta(t0 + h) - theta(t0 - h)) / (2.0 * h)
        self.assertAlmostEqual(float(ang_vel[1]), numeric_thetadot, places=5)


@unittest.skipUnless(CUDA_OK, "CUDA required")
class IzraRigidKinematicsTest(unittest.TestCase):
    """H2: the evaluated SurfaceFrame matches the analytic rigid transform."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-030"]
        cls.kin = IzraRigidSurfaceKinematics(
            cls.case, chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
        )

    def test_surface_ids_and_reference(self):
        self.assertEqual(self.kin.surface_ids, ("izra_wing",))
        self.assertIsInstance(self.kin.reference_frame, SurfaceFrame)
        self.assertEqual(self.kin.reference_frame.chordwise_panels, NC)
        self.assertEqual(self.kin.reference_frame.spanwise_panels, NS)

    def test_frame_is_exact_rigid_transform_about_pivot(self):
        case = self.case
        t = 0.0371
        omega = case.omega_rad_s
        psi = math.radians(case.phase_offset_deg)
        theta_max = math.radians(case.theta_max_deg)
        theta = theta_max * math.cos(omega * t + psi)
        thetadot = -theta_max * omega * math.sin(omega * t + psi)
        z = case.heave_amplitude_m * math.cos(omega * t)
        zdot = -case.heave_amplitude_m * omega * math.sin(omega * t)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotation = torch.tensor(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ],
            device=DEVICE,
            dtype=torch.float64,
        )
        pos = torch.tensor([0.0, 0.0, z], device=DEVICE, dtype=torch.float64)
        lin_vel = torch.tensor([0.0, 0.0, zdot], device=DEVICE, dtype=torch.float64)
        ang_vel = torch.tensor([0.0, thetadot, 0.0], device=DEVICE, dtype=torch.float64)

        ref = self.kin.reference_frame
        frame = self.kin.evaluate(t)
        for current, reference in (
            (frame.panel_rings_I, ref.panel_rings_I.reshape(-1, 3)),
            (frame.collocation_I, ref.collocation_I),
            (frame.leading_edge_I, ref.leading_edge_I),
            (frame.trailing_edge_I, ref.trailing_edge_I),
        ):
            expected = (reference @ rotation.T + pos).reshape_as(current)
            torch.testing.assert_close(current, expected, rtol=1e-12, atol=1e-12)
        for velocity, reference in (
            (frame.panel_ring_velocity_I, ref.panel_rings_I.reshape(-1, 3)),
            (frame.collocation_velocity_I, ref.collocation_I),
            (frame.leading_velocity_I, ref.leading_edge_I),
            (frame.trailing_velocity_I, ref.trailing_edge_I),
        ):
            rel = reference @ rotation.T
            expected = lin_vel + torch.cross(
                ang_vel.expand_as(rel), rel, dim=-1
            ).reshape_as(velocity)
            torch.testing.assert_close(
                velocity, expected, rtol=1e-12, atol=1e-10
            )
        # Rigid motion preserves areas exactly and rotates the normals.
        self.assertTrue(torch.equal(frame.areas, ref.areas))
        torch.testing.assert_close(
            frame.normals_I, ref.normals_I @ rotation.T, rtol=1e-12, atol=1e-12
        )

    def test_pitch_rotates_about_the_075c_axis(self):
        case = self.case
        t = 0.0  # z(0) = h; theta(0) = theta_max cos(psi)
        theta = math.radians(case.theta_max_deg) * math.cos(
            math.radians(case.phase_offset_deg)
        )
        z = case.heave_amplitude_m
        frame = self.kin.evaluate(t)
        c = case.chord_m
        # Rotation about the pivot at x/c=0.75: LE rises, TE drops.
        # world_z = z(t) - x_ref sin(theta).
        self.assertAlmostEqual(
            float(frame.leading_edge_I[:, 2].mean()), z + 0.75 * c * math.sin(theta),
            places=12,
        )
        self.assertAlmostEqual(
            float(frame.trailing_edge_I[:, 2].mean()), z - 0.25 * c * math.sin(theta),
            places=12,
        )
        # The pivot line (x_ref = 0) is the rotation axis: spanwise stations
        # keep their y exactly under pitch.
        torch.testing.assert_close(
            frame.leading_edge_I[:, 1],
            self.kin.reference_frame.leading_edge_I[:, 1],
            rtol=0.0,
            atol=1e-15,
        )

    def test_zero_pitch_heave_only_translates(self):
        # psi = 90 deg: theta(0) = theta_max*cos(90 deg) = 0, so the frame at
        # t=0 is exactly the reference translated by (0, 0, h).
        case = IZRA_CASES["IZRA-15-090"]
        kin = IzraRigidSurfaceKinematics(
            case, chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
        )
        ref = kin.reference_frame
        frame = kin.evaluate(0.0)
        shift = torch.tensor(
            [0.0, 0.0, case.heave_amplitude_m], device=DEVICE, dtype=torch.float64
        )
        torch.testing.assert_close(
            frame.panel_rings_I, ref.panel_rings_I + shift, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            frame.leading_edge_I, ref.leading_edge_I + shift, rtol=0.0, atol=0.0
        )
        # zdot(0) = 0, thetadot(0) = -theta_max omega: pure pitch rate about
        # the pivot gives v = (0, 0, -thetadot * x_ref).
        thetadot = -math.radians(case.theta_max_deg) * case.omega_rad_s
        expected = torch.stack(
            (
                torch.zeros_like(ref.panel_rings_I[..., 0]),
                torch.zeros_like(ref.panel_rings_I[..., 0]),
                -thetadot * ref.panel_rings_I[..., 0],
            ),
            dim=-1,
        )
        torch.testing.assert_close(
            frame.panel_ring_velocity_I, expected, rtol=1e-12, atol=1e-10
        )

    def test_velocities_match_finite_differences(self):
        t0 = 0.0371
        h = 1.0e-5
        frame = self.kin.evaluate(t0)
        plus = self.kin.evaluate(t0 + h)
        minus = self.kin.evaluate(t0 - h)
        for velocity, forward, backward in (
            (
                frame.panel_ring_velocity_I,
                plus.panel_rings_I,
                minus.panel_rings_I,
            ),
            (
                frame.collocation_velocity_I,
                plus.collocation_I,
                minus.collocation_I,
            ),
            (
                frame.leading_velocity_I,
                plus.leading_edge_I,
                minus.leading_edge_I,
            ),
            (
                frame.trailing_velocity_I,
                plus.trailing_edge_I,
                minus.trailing_edge_I,
            ),
        ):
            numeric = (forward - backward) / (2.0 * h)
            torch.testing.assert_close(velocity, numeric, rtol=0.0, atol=1e-5)


@unittest.skipUnless(CUDA_OK, "CUDA required")
class RigidV5MDirectDriveTest(unittest.TestCase):
    """H3: SurfaceFrame -> native V5M solver with no Q16 state bridge."""

    @classmethod
    def setUpClass(cls):
        cls.case = IZRA_CASES["IZRA-15-015"]
        cls.kin = IzraRigidSurfaceKinematics(
            cls.case, chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
        )
        cls.surface = RigidV5MSurface(
            chordwise_panels=NC, spanwise_panels=NS, device=DEVICE
        )
        cls.solver = RigidNativeV5MSolver(
            cls.surface,
            NativeV5MConfig(
                chordwise_panels=NC,
                spanwise_panels=NS,
                density=cls.case.rho_kg_m3,
                freestream=cls.case.freestream_m_s,
                aerodynamic_dt=AERO_DT,
                wake_max_rows=128,
                particle_capacity=32768,
                particle_max_age_steps=100,
                # The default 0.04/2.125 sits exactly on the frozen minimum
                # smoothing/spacing overlap and loses it to roundoff; use the
                # frozen production spacing from the Roj runner instead.
                dvm_target_spacing_chord=0.018,
                device=DEVICE,
            ),
        )
        cls.stepper = RigidV5MStepper(cls.solver, device=DEVICE)

    def test_01_protocol_and_no_structural_bridge(self):
        self.assertIsInstance(self.stepper, AerodynamicStepper)
        # The transitional Q16 bridge does not exist on the rigid path.
        self.assertFalse(hasattr(self.stepper, "set_structural_state"))
        frame = self.kin.evaluate(0.0)
        owner = self.stepper.initialize((frame,))
        topology = self.stepper.topology
        self.assertEqual(topology.total_panels, NC * NS)
        self.assertEqual(topology.surfaces[0].surface_id, "izra_wing")
        self.assertEqual(topology.surfaces[0].le_node_count, NS + 1)
        self.assertEqual(owner.state.step, 0)

    def test_02_surface_roundtrip_and_guards(self):
        frame = self.kin.evaluate(0.01)
        self.surface.set_frame(frame)
        geometry = self.surface.evaluate(None, None)
        self.assertTrue(torch.equal(geometry.rings, frame.panel_rings_I))
        self.assertTrue(torch.equal(geometry.ring_velocity, frame.panel_ring_velocity_I))
        self.assertTrue(torch.equal(geometry.collocation, frame.collocation_I))
        self.assertTrue(torch.equal(geometry.collocation_velocity, frame.collocation_velocity_I))
        self.assertTrue(torch.equal(geometry.normals, frame.normals_I))
        self.assertTrue(torch.equal(geometry.areas, frame.areas))
        self.assertTrue(torch.equal(geometry.leading_edge, frame.leading_edge_I))
        self.assertTrue(torch.equal(geometry.trailing_edge, frame.trailing_edge_I))
        # A raw structural state cannot sneak onto the rigid path.
        with self.assertRaises(TypeError):
            self.surface.evaluate(object(), None)
        with self.assertRaises(TypeError):
            self.surface.evaluate(None, object())
        fresh = RigidV5MSurface(chordwise_panels=NC, spanwise_panels=NS, device=DEVICE)
        with self.assertRaises(RuntimeError):
            fresh.evaluate(None, None)
        mismatched = build_izra_reference_grid(
            chordwise_panels=NC + 1,
            spanwise_panels=NS,
            chord_m=self.case.chord_m,
            span_m=self.case.span_m,
            pivot_fraction_chord=self.case.pivot_fraction_chord,
            device=DEVICE,
        )
        with self.assertRaises(ValueError):
            fresh.set_frame(mismatched)

    def test_03_rigid_load_assembler_rejects_structural_state(self):
        assembler = RigidAuthorLoadAssembler(density=self.case.rho_kg_m3, device=DEVICE)
        with self.assertRaises(TypeError):
            assembler.assemble(
                structural_state=object(),
                geometry=None,
                aic=None,
                gamma=None,
                external_flow=None,
                gamma_gradient=None,
                mf2_history=None,
            )

    def test_04_solve_steps_transactionally_from_frames(self):
        owner = self.stepper.initialize((self.kin.evaluate(0.0),))
        previous_digest = owner.state.digest()
        previous_wake = 0
        for step in range(1, STEPS + 1):
            frame = self.kin.evaluate(step * AERO_DT)
            proposal = self.stepper.propose(owner, (frame,), AERO_DT)
            # The solver consumed exactly this frame's geometry.
            self.assertTrue(
                torch.equal(self.solver.surface._geometry.rings, frame.panel_rings_I)
            )
            # Complete-pressure load: panel forces + a (1, 6) world wrench.
            self.assertEqual(tuple(proposal.load.panel_forces.shape), (NC * NS, 3))
            self.assertTrue(bool(torch.isfinite(proposal.load.panel_forces).all().item()))
            wrench = wp.to_torch(proposal.generalized_force)
            self.assertEqual(tuple(wrench.shape), (1, 6))
            self.assertTrue(bool(torch.isfinite(wrench).all().item()))
            parts = (
                proposal.author_load.lift1_pressure,
                proposal.author_load.wake_history_pressure,
                proposal.author_load.lift2_pressure,
                proposal.author_load.mf2_1_pressure,
            )
            for part in parts:
                self.assertTrue(bool(torch.isfinite(part).all().item()))
            torch.testing.assert_close(
                proposal.author_load.constant_pressure,
                sum(parts),
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertTrue(
                torch.allclose(
                    proposal.load.panel_forces,
                    proposal.author_load.panel_forces,
                    rtol=0.0,
                    atol=0.0,
                )
            )
            # Parent stays untouched until commit (transactional proposal).
            self.assertEqual(owner.state.digest(), previous_digest)
            rejected = self.stepper.propose(owner, (frame,), AERO_DT)
            self.assertEqual(rejected.parent_digest, proposal.parent_digest)
            self.assertEqual(owner.state.digest(), previous_digest)
            self.stepper.commit(owner, proposal)
            self.assertEqual(owner.state.step, step)
            self.assertNotEqual(owner.state.digest(), previous_digest)
            previous_digest = owner.state.digest()
            # Each step sheds exactly one full-span TEV wake row.
            self.assertEqual(owner.state.wake_gamma.numel(), previous_wake + NS)
            previous_wake = owner.state.wake_gamma.numel()
            self.assertIn("lev_release_count", owner.state.diagnostics[-1])
            self.assertTrue(owner.state.diagnostics[-1]["cuda_float64"])

    def test_05_one_way_coupling_drives_the_rigid_path(self):
        owner = WorldOwner(
            dynamic_state=WorldDynamicState(
                elastic_states={}, body_states={}, joint_states={}
            ),
            aero_state=self.stepper.initialize((self.kin.evaluate(0.0),)),
            previous_load=None,
        )
        coupling = OneWayPrescribedCoupling(self.stepper)
        for step in range(1, STEPS + 1):
            frames = (self.kin.evaluate(step * AERO_DT),)
            proposal = coupling.advance(owner, surface_frames=frames, delta_time=AERO_DT)
            self.assertEqual(owner.aero_state.state.step, step)
            self.assertTrue(bool(torch.isfinite(proposal.load.total_force).all().item()))
        self.assertEqual(owner.generation, STEPS)

    def test_06_initialize_and_propose_validate_topology(self):
        owner = self.stepper.initialize((self.kin.evaluate(0.0),))
        with self.assertRaises(ValueError):
            self.stepper.initialize(())
        with self.assertRaises(NotImplementedError):
            second = build_izra_reference_grid(
                chordwise_panels=NC,
                spanwise_panels=NS,
                chord_m=self.case.chord_m,
                span_m=self.case.span_m,
                pivot_fraction_chord=self.case.pivot_fraction_chord,
                device=DEVICE,
            )
            import dataclasses

            self.stepper.initialize(
                (self.kin.evaluate(0.0), dataclasses.replace(second, surface_id="wing_1"))
            )
        uninitialized = RigidV5MStepper(self.solver, device=DEVICE)
        with self.assertRaises(RuntimeError):
            uninitialized.propose(owner, (self.kin.evaluate(0.0),), AERO_DT)


if __name__ == "__main__":
    unittest.main()
