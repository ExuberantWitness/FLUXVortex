import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.hirato_equations import HiratoEquationError  # noqa: E402
from claim_runtime.hirato_live_shadow import (  # noqa: E402
    HiratoLiveShadow,
    HiratoTrailingSheetShadow,
    build_bound_aic,
    build_bound_lattice,
)


def rectangular_wing(
    nc: int,
    ns: int,
    *,
    chord: float = 1.0,
    half_span: float = 1.0,
):
    """Return a stationary, flat half-wing in the runtime corner convention."""
    x = np.linspace(0.0, chord, nc + 1)
    y = np.linspace(0.0, half_span, ns + 1)
    corners = np.zeros((nc + 1, ns + 1, 3))
    corners[..., 0] = x[:, None]
    corners[..., 1] = y[None, :]
    return corners, np.zeros_like(corners)


class BoundLatticeTests(unittest.TestCase):
    def test_flat_lattice_preserves_n1_quarter_chord_geometry(self):
        corners, velocity = rectangular_wing(2, 2)
        lattice = build_bound_lattice(corners, velocity, nc=2, ns=2)

        self.assertEqual(lattice.rings.shape, (4, 4, 3))
        self.assertEqual(lattice.collocation.shape, (4, 3))
        np.testing.assert_allclose(
            lattice.normals,
            np.tile([0.0, 0.0, 1.0], (4, 1)),
        )
        np.testing.assert_allclose(lattice.chord, 1.0)
        np.testing.assert_allclose(lattice.delta_x_front, 0.5)
        np.testing.assert_allclose(lattice.rings[0, 0], [0.125, 0.0, 0.0])
        np.testing.assert_allclose(lattice.rings[0, 1], [0.125, 0.5, 0.0])
        np.testing.assert_allclose(lattice.rings[0, 3], [0.625, 0.0, 0.0])
        np.testing.assert_allclose(lattice.collocation[0], [0.375, 0.25, 0.0])

    def test_numpy_geometry_and_aic_match_frozen_warp_n1(self):
        import warp as wp

        kernel_cache = tempfile.TemporaryDirectory(
            prefix="fluxv_hirato_warp_"
        )
        self.addCleanup(kernel_cache.cleanup)
        wp.config.kernel_cache_dir = kernel_cache.name
        import _v2_robo as robo
        import diff_uvlm_unsteady_gpu as ug
        from fluxvortex.warp_fsi.config import DTYPE

        nc, ns = 2, 3
        corners, velocity = rectangular_wing(nc, ns)
        corners[..., 2] = (
            0.04 * corners[..., 0]
            + 0.03 * corners[..., 0] * corners[..., 1]
        )
        velocity[..., 0] = 0.05 * corners[..., 1]
        velocity[..., 2] = -0.2 * corners[..., 0]
        reference = build_bound_lattice(
            corners,
            velocity,
            nc=nc,
            ns=ns,
        )
        npan = nc * ns
        device = "cpu"
        cw = wp.array(
            corners.reshape(-1, 3),
            dtype=wp.vec3d,
            device=device,
        )
        vw = wp.array(
            velocity.reshape(-1, 3),
            dtype=wp.vec3d,
            device=device,
        )
        rings = wp.zeros((npan, 4), dtype=wp.vec3d, device=device)
        collocation = wp.zeros(npan, dtype=wp.vec3d, device=device)
        normals = wp.zeros(npan, dtype=wp.vec3d, device=device)
        col_velocity = wp.zeros(npan, dtype=wp.vec3d, device=device)
        wp.launch(
            ug.bound_rings_kernel,
            dim=npan,
            inputs=[cw, nc, ns],
            outputs=[rings, collocation, normals],
            device=device,
        )
        wp.launch(
            ug.colvel_kernel,
            dim=npan,
            inputs=[vw, nc, ns],
            outputs=[col_velocity],
            device=device,
        )
        np.testing.assert_allclose(reference.rings, rings.numpy(), atol=1e-14)
        np.testing.assert_allclose(
            reference.collocation,
            collocation.numpy(),
            atol=1e-14,
        )
        np.testing.assert_allclose(reference.normals, normals.numpy(), atol=1e-14)
        np.testing.assert_allclose(
            reference.collocation_velocity,
            col_velocity.numpy(),
            atol=1e-14,
        )

        for symmetry, kernel in (
            (False, ug.aic_kernel),
            (True, robo.aic_sym_kernel),
        ):
            aic = wp.zeros(
                (1, npan, npan),
                dtype=DTYPE,
                device=device,
            )
            wp.launch(
                kernel,
                dim=(npan, npan),
                inputs=[rings, collocation, normals],
                outputs=[aic],
                device=device,
            )
            np.testing.assert_allclose(
                build_bound_aic(reference, mirror_symmetry=symmetry),
                aic.numpy()[0],
                rtol=2e-13,
                atol=2e-13,
            )

    def test_warp_velocity_channels_match_numpy_oracle(self):
        import warp as wp

        kernel_cache = tempfile.TemporaryDirectory(
            prefix="fluxv_hirato_field_"
        )
        self.addCleanup(kernel_cache.cleanup)
        wp.config.kernel_cache_dir = kernel_cache.name
        from claim_runtime.hirato_shadow import (
            mirrored_ring_field,
            ring_field_velocity_lamb_oseen,
        )
        from claim_runtime.hirato_warp_backend import velocity_channels

        points = np.array(
            [
                [0.2, 0.25, 0.3],
                [0.7, 0.60, -0.1],
                [1.1, 0.90, 0.2],
            ]
        )
        base = np.array(
            [
                [
                    [0.0, 0.1, 0.0],
                    [0.0, 0.4, 0.0],
                    [0.5, 0.4, 0.1],
                    [0.5, 0.1, 0.1],
                ],
                [
                    [0.4, 0.4, 0.1],
                    [0.4, 0.8, 0.1],
                    [0.9, 0.8, 0.0],
                    [0.9, 0.4, 0.0],
                ],
            ]
        )
        bound_rings = base[:1]
        tev_rings = base[1:]
        lev_rings = base.copy()
        bound_gamma = np.array([0.2])
        tev_gamma = np.array([-0.3])
        lev_gamma = np.array([0.07, -0.11])
        u_inf = np.array([2.0, 0.0, 0.1])
        core = 0.04

        def numpy_field(rings, gamma):
            direct = ring_field_velocity_lamb_oseen(
                points,
                rings,
                gamma,
                core,
            )
            image = ring_field_velocity_lamb_oseen(
                points,
                mirrored_ring_field(rings),
                gamma,
                core,
            )
            return direct + image

        expected_bound = numpy_field(bound_rings, bound_gamma)
        expected_tev = numpy_field(tev_rings, tev_gamma)
        expected_lev = numpy_field(lev_rings, lev_gamma)
        actual = velocity_channels(
            points,
            bound_rings=bound_rings,
            bound_gamma=bound_gamma,
            tev_rings=tev_rings,
            tev_gamma=tev_gamma,
            lev_rings=lev_rings,
            lev_gamma=lev_gamma,
            u_infinity=u_inf,
            core_radius=core,
            mirror_symmetry=True,
            device="cpu",
        )
        np.testing.assert_allclose(actual.bound, expected_bound, atol=2e-14)
        np.testing.assert_allclose(actual.tev, expected_tev, atol=2e-14)
        np.testing.assert_allclose(actual.lev, expected_lev, atol=2e-14)
        np.testing.assert_allclose(
            actual.total,
            u_inf + expected_bound + expected_tev + expected_lev,
            atol=2e-14,
        )


class TrailingSheetTests(unittest.TestCase):
    def setUp(self):
        self.edges = np.array(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0]],
                [[1.0, 0.5, 0.0], [1.0, 1.0, 0.0]],
            ]
        )
        self.track = np.broadcast_to(
            np.array([2.0, 0.0, 0.0]),
            self.edges.shape,
        ).copy()

    def test_new_tev_front_is_point_three_freestream_steps_from_edge(self):
        state = HiratoTrailingSheetShadow(ns=2)
        state.shed(
            step=0,
            trailing_edges=self.edges,
            track_velocity=self.track,
            u_infinity_speed=2.0,
            dt=0.1,
            gamma_now=np.array([0.2, -0.1]),
        )
        np.testing.assert_allclose(
            state.rings[:, :2] - self.edges,
            np.broadcast_to([0.06, 0.0, 0.0], self.edges.shape),
        )
        np.testing.assert_allclose(
            state.rings[:, 2:] - state.rings[:, :2][:, ::-1],
            np.broadcast_to([0.2, 0.0, 0.0], self.edges.shape),
        )

    def test_newborn_tev_convects_and_next_row_connects_to_convected_front(self):
        state = HiratoTrailingSheetShadow(ns=2)
        state.shed(
            step=0,
            trailing_edges=self.edges,
            track_velocity=self.track,
            u_infinity_speed=2.0,
            dt=0.1,
            gamma_now=np.array([0.2, -0.1]),
        )
        state.convect_eq24(
            np.broadcast_to([1.0, 0.0, 0.0], state.rings.shape),
            0.1,
            step=0,
        )
        prior_front = state.rings[:, :2].copy()
        state.shed(
            step=1,
            trailing_edges=self.edges,
            track_velocity=self.track,
            u_infinity_speed=2.0,
            dt=0.1,
            gamma_now=np.array([0.3, -0.2]),
        )
        np.testing.assert_allclose(state.rings[-2:, 2], prior_front[:, 1])
        np.testing.assert_allclose(state.rings[-2:, 3], prior_front[:, 0])


class HiratoLiveStateTests(unittest.TestCase):
    @staticmethod
    def pitched_wing(alpha_deg: float):
        corners, velocity = rectangular_wing(2, 2)
        angle = np.radians(alpha_deg)
        relative_x = corners[..., 0] - 0.25
        corners[..., 0] = 0.25 + relative_x * np.cos(angle)
        corners[..., 2] = relative_x * np.sin(angle)
        return corners, velocity

    def test_static_flat_wing_closes_kelvin_and_convection_ledgers(self):
        corners, velocity = rectangular_wing(2, 2)
        state = HiratoLiveShadow(
            nc=2,
            ns=2,
            u_infinity=np.array([2.0, 0.0, 0.0]),
            dt=0.01,
            lesp_crit=10.0,
            core_radius=0.01,
            mirror_symmetry=False,
        )

        first = state.step(step=0, corners=corners, corner_velocity=velocity)
        second = state.step(step=1, corners=corners, corner_velocity=velocity)

        for report in (first, second):
            np.testing.assert_allclose(report.eq9_residual, 0.0, atol=0.0)
            np.testing.assert_allclose(report.a0_event, 0.0, atol=1e-14)
            self.assertFalse(np.any(report.active))
            self.assertEqual(len(report.lev_pre_convection.rings), 0)
            self.assertLessEqual(
                report.convection_ledger_max_abs_residual,
                1e-14,
            )
        self.assertEqual(first.tev_bootstrap_vertices, 2 * 4)
        self.assertEqual(second.tev_bootstrap_vertices, 2 * 4)
        np.testing.assert_allclose(
            second.gamma_tev,
            first.bound_gamma.reshape(2, 2)[-1] + first.gamma_lev,
        )
        self.assertFalse(hasattr(first, "force"))
        self.assertFalse(hasattr(first, "pressure"))

    def test_active_lev_constraint_closes_before_kelvin_advances(self):
        corners, velocity = self.pitched_wing(15.0)
        state = HiratoLiveShadow(
            nc=2,
            ns=2,
            u_infinity=np.array([2.0, 0.0, 0.0]),
            dt=0.01,
            lesp_crit=0.05,
            core_radius=0.01,
            mirror_symmetry=False,
        )
        first = state.step(step=0, corners=corners, corner_velocity=velocity)
        second = state.step(step=1, corners=corners, corner_velocity=velocity)

        self.assertTrue(np.all(first.active))
        self.assertTrue(np.all(first.new_sheet))
        self.assertTrue(np.all(second.active))
        self.assertFalse(np.any(second.new_sheet))
        self.assertEqual(len(first.lev_pre_convection.rings), 2)
        self.assertEqual(len(second.lev_pre_convection.rings), 4)
        self.assertLess(np.max(np.abs(first.lesp_active_residual)), 1e-14)
        self.assertLess(np.max(np.abs(second.lesp_active_residual)), 1e-14)
        np.testing.assert_allclose(
            second.gamma_tev,
            first.bound_gamma.reshape(2, 2)[-1] + first.gamma_lev,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(second.eq9_residual, 0.0, atol=0.0)
        self.assertEqual(first.lev_bootstrap_vertices, 2 * 4)
        # Eq.7 creates four new material vertices per strip and preserves two
        # aft vertices per old ring, hence 12 bootstrapped vertices here.
        self.assertEqual(second.lev_bootstrap_vertices, 2 * 6)

    def test_step_identity_refuses_skips_and_replays(self):
        corners, velocity = rectangular_wing(1, 1)
        state = HiratoLiveShadow(
            nc=1,
            ns=1,
            u_infinity=np.array([1.0, 0.0, 0.0]),
            dt=0.01,
            lesp_crit=10.0,
            core_radius=0.01,
            mirror_symmetry=False,
        )
        with self.assertRaises(HiratoEquationError):
            state.step(step=1, corners=corners, corner_velocity=velocity)
        state.step(step=0, corners=corners, corner_velocity=velocity)
        with self.assertRaises(HiratoEquationError):
            state.step(step=0, corners=corners, corner_velocity=velocity)


if __name__ == "__main__":
    unittest.main()
