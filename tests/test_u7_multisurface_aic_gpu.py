"""U7-3: multi-surface global AIC with cross-surface mutual induction.

Builds the formal A16 wing pair exactly as the U4 topology test does
(right wing from the production native geometry, left wing as the y-mirror
with reversed ring winding and negated mirrored normals — a physical left
wing, not a sign-flipped right one), then assembles the GLOBAL (900, 900)
bound-vortex influence matrix with the production kernel and pins the
U7-3 contract:

- shape / dtype / device and finiteness;
- the diagonal (self-influence) blocks are bit-identical to each wing's
  own native_aic, so global assembly cannot perturb single-surface
  numerics;
- the cross blocks (left wing rings influencing right wing collocation
  points and vice versa) are nonzero — mutual induction exists;
- the mirror-pair structure: A_LL == A_RR and A_LR == A_RL exactly
  (the stored winding convention makes the mirror pair block-symmetric);
- the orientation gate: every global diagonal entry stays negative;
- cross_influence_norm quantifies |A_cross|_F / |A_self|_F in a physical
  range (nonzero, not dominant);
- a probe joint bound solve over the global system is finite, satisfies
  its own residual, stays mirror-antisymmetric (gamma_L == -gamma_R for
  the mirrored loading), and differs measurably from two independent
  single-wing solves — the number that motivates the U7-4 coupled solve.

CPU-side tests pin the validation and the solver-shell wiring with tiny
synthetic frames (the kernel is pure torch, so no CUDA is needed there).
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.v5m.multi_solver import (
    MultiSurfaceV5MSolver,
    build_cross_aic,
    build_global_aic,
    cross_influence_norm,
    mutual_induction_report,
)
from fluxvortex.aero.v5m.topology import MultiSurfaceTopology
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    freestream_vector,
    native_aic,
    Q16NativeV5MSurface,
)
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)

DTYPE = torch.float64


def _mirror_frame_y(frame: SurfaceFrame, surface_id: str) -> SurfaceFrame:
    """Mirror a frame in y (right wing -> left wing), U4 convention.

    Points and velocity vectors get y -> -y; the odd-dimensioned mirror
    reverses ring winding, so the stored normals are fully negated
    (-M n) and the production winding convention (stored = -corner-cross)
    continues to hold on the mirrored surface.
    """
    def neg_y(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[..., 1] = -out[..., 1]
        return out

    return SurfaceFrame(
        surface_id=surface_id,
        body_id=frame.body_id,
        panel_rings_I=neg_y(frame.panel_rings_I),
        panel_ring_velocity_I=neg_y(frame.panel_ring_velocity_I),
        collocation_I=neg_y(frame.collocation_I),
        collocation_velocity_I=neg_y(frame.collocation_velocity_I),
        normals_I=-neg_y(frame.normals_I),
        areas=frame.areas.clone(),
        leading_edge_I=neg_y(frame.leading_edge_I),
        trailing_edge_I=neg_y(frame.trailing_edge_I),
        leading_velocity_I=neg_y(frame.leading_velocity_I),
        trailing_velocity_I=neg_y(frame.trailing_velocity_I),
        chordwise_panels=frame.chordwise_panels,
        spanwise_panels=frame.spanwise_panels,
        topology_digest=frame.topology_digest + "|mirror-y",
    )


@unittest.skipUnless(torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required")
class U7MultiSurfaceAicGpuTest(unittest.TestCase):
    """Global AIC over the formal A16 mirror pair (2 x 450 = 900 panels)."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, _, _, _ = make_rojratsirikul2011_q16_model(
            chordwise_element_count=FORMAL_Q16_GRID[0],
            spanwise_element_count=FORMAL_Q16_GRID[1],
            case=ROJ11_A16,
        )
        reference = np.array(cls.mesh.reference_state, dtype=np.float64)
        cls.state = wp.array(
            np.ascontiguousarray(reference[None, :]),
            dtype=config.DTYPE, device=config.DEVICE,
        )
        cls.zero_velocity = wp.array(
            np.zeros((1, reference.size), dtype=np.float64),
            dtype=config.DTYPE, device=config.DEVICE,
        )
        cls.native = Q16NativeV5MSurface(
            cls.mesh,
            q16_chordwise_elements=FORMAL_Q16_GRID[0],
            q16_spanwise_elements=FORMAL_Q16_GRID[1],
            aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
            aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
            device=config.DEVICE,
        )
        cls.geometry = cls.native.evaluate(cls.state, cls.zero_velocity)
        cls.nc = FORMAL_AERO_GRID[0]
        cls.ns = FORMAL_AERO_GRID[1]
        cls.panels_per_wing = cls.nc * cls.ns                 # 450
        cls.right = SurfaceFrame.from_native_geometry(
            cls.geometry, surface_id="right_wing", body_id="body_0",
        )
        cls.left = _mirror_frame_y(cls.right, surface_id="left_wing")
        cls.frames = (cls.right, cls.left)
        cls.topology = MultiSurfaceTopology.from_surface_frames(cls.frames)
        cls.global_aic = build_global_aic(cls.frames)
        cls.single_aic = native_aic(cls.geometry, chordwise_panels=cls.nc)

    def test_global_aic_shape_dtype_device_finite(self):
        aic = self.global_aic
        self.assertEqual(tuple(aic.shape), (900, 900))
        self.assertEqual(aic.dtype, DTYPE)
        self.assertTrue(aic.is_cuda)
        self.assertTrue(bool(torch.isfinite(aic).all().item()))

    def test_self_blocks_match_single_surface_native_aic(self):
        """Diagonal blocks are bit-identical to each wing's own native_aic."""
        n = self.panels_per_wing
        aic = self.global_aic
        self.assertTrue(torch.equal(aic[:n, :n], self.single_aic))
        # The mirrored wing's self-block also lands on the same matrix
        # (checked exactly in the mirror-structure test; here it must at
        # least be a valid second diagonal block of the same scale).
        self.assertEqual(tuple(aic[n:, n:].shape), (n, n))
        torch.testing.assert_close(aic[n:, n:], self.single_aic, rtol=0, atol=0)

    def test_cross_blocks_are_nonzero(self):
        """Left wing rings DO influence right wing collocation points."""
        n = self.panels_per_wing
        aic = self.global_aic
        cross_rl = aic[:n, n:]      # right collocation <- left rings
        cross_lr = aic[n:, :n]      # left collocation <- right rings
        self.assertEqual(float(torch.count_nonzero(cross_rl).item()), cross_rl.numel())
        self.assertEqual(float(torch.count_nonzero(cross_lr).item()), cross_lr.numel())
        self_max = float(torch.max(torch.abs(self.single_aic)).item())
        for cross in (cross_rl, cross_lr):
            self.assertGreater(float(torch.max(torch.abs(cross)).item()),
                               0.05 * self_max)

    def test_mirror_pair_block_structure(self):
        """A_LL == A_RR and A_LR == A_RL exactly for the mirror pair.

        y-negation is exact in floats and every kernel op is
        sign-symmetric, so the stored winding convention (left normals
        = -M(right normals)) makes both block identities bit-exact.
        """
        n = self.panels_per_wing
        aic = self.global_aic
        self.assertTrue(torch.equal(aic[n:, n:], aic[:n, :n]))
        self.assertTrue(torch.equal(aic[:n, n:], aic[n:, :n]))

    def test_orientation_diagonal_negative(self):
        diagonal = torch.diagonal(self.global_aic)
        self.assertTrue(bool((diagonal < 0.0).all().item()))

    def test_cross_influence_norm_ratio_is_physical(self):
        report = mutual_induction_report(self.global_aic, self.topology)
        self.assertEqual(report.total_panels, 900)
        self.assertGreater(report.ratio, 1.0e-3)   # not zero: wings interact
        self.assertLess(report.ratio, 0.5)        # not dominant: self-rule holds
        self.assertEqual(report.cross_nonzero_fraction, 1.0)
        self.assertGreater(report.cross_max_abs, 0.0)
        # cross_influence_norm(frames) rebuilds and returns the same ratio.
        ratio = cross_influence_norm(self.frames)
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 1.0e-3)
        self.assertLess(ratio, 0.5)

    def test_joint_solve_quantifies_mutual_induction(self):
        """The joint bound solve differs measurably from independent solves.

        Probe loading: unit freestream at 5 deg with no motion and no wake
        gives rhs_i = -(v_inf . n_i) on every panel.  Solving the GLOBAL
        system versus each wing's own single-surface system isolates
        exactly the circulation change mutual induction would produce in
        the U7-4 coupled solve.
        """
        n = self.panels_per_wing
        aic = self.global_aic
        v_inf = freestream_vector(1.0, 5.0, aic.device)
        normals = torch.cat((self.right.normals_I, self.left.normals_I), dim=0)
        rhs = -torch.sum(v_inf[None, :] * normals, dim=1)
        scale = float(torch.max(torch.abs(rhs)).item())
        self.assertGreater(scale, 1.0e-3)  # the probe loading is non-degenerate

        gamma_joint = torch.linalg.solve(aic, rhs)
        self.assertTrue(bool(torch.isfinite(gamma_joint).all().item()))
        residual = float(torch.max(torch.abs(aic @ gamma_joint - rhs)).item())
        self.assertLess(residual, 1.0e-8 * scale)

        # Mirror-pair structure: [[S, C], [C, S]] with rhs_L = -rhs_R gives
        # gamma_L = -gamma_R (same physical lift on both wings).
        torch.testing.assert_close(
            gamma_joint[n:], -gamma_joint[:n], rtol=1.0e-8, atol=1.0e-8 * scale
        )

        # Independent per-wing solves: what the production single-surface
        # path computes.  The joint answer shifts by a measurable amount.
        gamma_independent = torch.cat(
            (
                torch.linalg.solve(aic[:n, :n], rhs[:n]),
                torch.linalg.solve(aic[n:, n:], rhs[n:]),
            ),
            dim=0,
        )
        shift = float(
            torch.max(torch.abs(gamma_joint - gamma_independent)).item()
        )
        independent_scale = float(torch.max(torch.abs(gamma_independent)).item())
        self.assertGreater(independent_scale, 0.0)
        relative_shift = shift / independent_scale
        self.assertGreater(relative_shift, 1.0e-3)   # mutual induction matters
        self.assertLess(relative_shift, 1.0)         # but is a perturbation

    def test_build_cross_aic_alias_matches(self):
        self.assertTrue(torch.equal(build_cross_aic(self.frames), self.global_aic))


# ---------------------------------------------------------------------------
# CPU contracts: tiny synthetic pairs, validation, solver shell
# ---------------------------------------------------------------------------

def _make_flat_wing(
    nc: int, ns: int, surface_id: str, *, y0: float = 0.0, chord: float = 1.0, span: float = 1.0
) -> SurfaceFrame:
    """A static flat wing: nc x ns panels on z=0 with production winding.

    Rings are wound (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1) with the
    stored normal = -normalize((c1-c0) x (c3-c0)) = -z, matching the
    production convention the orientation gate (negative AIC diagonal)
    relies on.
    """
    xs = torch.linspace(0.0, chord, nc + 1, dtype=DTYPE)
    ys = y0 + torch.linspace(0.0, span, ns + 1, dtype=DTYPE)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    corners = torch.stack(
        (
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.zeros((nc + 1) * (ns + 1), dtype=DTYPE),
        ),
        dim=-1,
    ).reshape(nc + 1, ns + 1, 3)
    rings = []
    for i in range(nc):
        for j in range(ns):
            rings.append(
                torch.stack(
                    (corners[i, j], corners[i + 1, j], corners[i + 1, j + 1], corners[i, j + 1])
                )
            )
    panel_rings = torch.stack(rings)
    collocation = panel_rings.mean(dim=1)
    normals = torch.tile(torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE), (panel_rings.shape[0], 1))
    leading = corners[0, :, :].clone()
    trailing = corners[-1, :, :].clone()
    return SurfaceFrame(
        surface_id=surface_id,
        body_id="body_0",
        panel_rings_I=panel_rings,
        panel_ring_velocity_I=torch.zeros_like(panel_rings),
        collocation_I=collocation,
        collocation_velocity_I=torch.zeros_like(collocation),
        normals_I=normals,
        areas=torch.full((panel_rings.shape[0],), 1.0 / panel_rings.shape[0], dtype=DTYPE),
        leading_edge_I=leading,
        trailing_edge_I=trailing,
        leading_velocity_I=torch.zeros_like(leading),
        trailing_velocity_I=torch.zeros_like(trailing),
        chordwise_panels=nc,
        spanwise_panels=ns,
        topology_digest=f"test:flat:{nc}x{ns}",
    )


def _stub_solver(nc: int, ns: int, device: str = "cpu") -> SimpleNamespace:
    """Just enough of a Q16NativeV5MSolver for MultiSurfaceV5MSolver wiring."""
    return SimpleNamespace(
        settings=SimpleNamespace(chordwise_panels=nc, spanwise_panels=ns, device=device)
    )


class U7MultiSurfaceAicCpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.right = _make_flat_wing(2, 2, "right_wing")
        cls.left = _mirror_frame_y(cls.right, "left_wing")
        cls.frames = (cls.right, cls.left)
        cls.topology = MultiSurfaceTopology.from_surface_frames(cls.frames)
        cls.global_aic = build_global_aic(cls.frames)

    def test_tiny_pair_global_aic(self):
        aic = self.global_aic
        self.assertEqual(tuple(aic.shape), (8, 8))
        self.assertEqual(aic.dtype, DTYPE)
        self.assertTrue(bool(torch.isfinite(aic).all().item()))
        self.assertTrue(bool((torch.diagonal(aic) < 0.0).all().item()))
        # Cross influence exists and the mirror structure holds here too.
        self.assertTrue(torch.equal(aic[4:, 4:], aic[:4, :4]))
        self.assertTrue(torch.equal(aic[:4, 4:], aic[4:, :4]))
        self.assertGreater(float(torch.max(torch.abs(aic[:4, 4:])).item()), 0.0)
        ratio = cross_influence_norm(self.frames)
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_empty_frames_rejected(self):
        with self.assertRaises(ValueError):
            build_global_aic(())

    def test_topology_frame_mismatch_rejected(self):
        # Topology surface ids must match the frames in order.
        other = _make_flat_wing(2, 2, "tail_wing")
        with self.assertRaises(ValueError):
            build_global_aic(self.frames, topology=MultiSurfaceTopology.from_surface_frames((self.right, other)))
        # Panel counts must agree with the topology's per-surface counts.
        smaller = _make_flat_wing(1, 2, "left_wing")
        with self.assertRaises(ValueError):
            build_global_aic((self.right, smaller), topology=self.topology)
        # Declared panel counts must match the tensors.
        bad = SimpleNamespace(
            surface_id="left_wing",
            chordwise_panels=3, spanwise_panels=2,
            collocation_I=self.left.collocation_I,
            normals_I=self.left.normals_I,
            panel_rings_I=self.left.panel_rings_I,
        )
        with self.assertRaises(ValueError):
            build_global_aic((self.right, bad))

    def test_solver_shell_wiring_and_validation(self):
        solver = MultiSurfaceV5MSolver(
            [_stub_solver(2, 2), _stub_solver(2, 2)], self.topology
        )
        self.assertEqual(solver.total_panels, 8)
        aic = solver.build_global_aic(self.frames)
        self.assertTrue(torch.equal(aic, self.global_aic))
        report = solver.mutual_induction(self.frames)
        self.assertEqual(report.total_panels, 8)
        self.assertGreater(report.ratio, 0.0)
        # Panel-count mismatch and empty solver lists are refused.
        with self.assertRaises(ValueError):
            MultiSurfaceV5MSolver([], self.topology)
        with self.assertRaises(ValueError):
            MultiSurfaceV5MSolver([_stub_solver(2, 2)], self.topology)
        with self.assertRaises(ValueError):
            MultiSurfaceV5MSolver([_stub_solver(2, 2), _stub_solver(3, 2)], self.topology)
        # Mixed devices are refused.
        with self.assertRaises(ValueError):
            MultiSurfaceV5MSolver(
                [_stub_solver(2, 2, "cpu"), _stub_solver(2, 2, "cuda:0")], self.topology
            )

    def test_propose_independent_needs_one_state_per_surface(self):
        solver = MultiSurfaceV5MSolver(
            [_stub_solver(2, 2), _stub_solver(2, 2)], self.topology
        )
        with self.assertRaises(ValueError):
            solver.propose_independent([object()], [object(), object()],
                                        [object(), object()], self.frames)


if __name__ == "__main__":
    unittest.main()
