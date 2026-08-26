"""U4: multi-surface V5M topology — global offsets and frame concatenation.

Plan §8.6/§14 U4: left+right wings share one body and contribute panels to a
single global AIC system.  These tests pin the data structure that makes that
possible:

- ``MultiSurfaceTopology.from_surface_frames``: correct panel/strip/LE-node
  offsets for the formal A16 pair (2 x 450 = 900 panels) and for asymmetric
  grids (15x30 + 10x20);
- global/local panel index round-trips and ownership boundaries;
- ``concatenate_surface_frames``: shapes and per-surface block identity;
- the U4-P parity gate: a single surface in the multi-surface container
  reproduces its own tensors bit-identically (offsets 0, no reordering);
- ``MengCaseConfig`` defaults and ``MultiSurfaceKinematics`` aggregation.

The reference frame comes from the A16 production mesh (same pattern as the
U0-P parity test); the left wing is the y-mirror of the right wing with the
spanwise node ordering swapped and ring winding reversed so the mirrored
stored normals are the mirrored normals (a physical left wing, not a
sign-flipped right one).
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.v5m.topology import (
    MultiSurfaceTopology,
    SurfaceOffsets,
    concatenate_surface_frames,
)
from fluxvortex.cases.meng2025 import MENG_PRIMARY, MengCaseConfig
from fluxvortex.cases.protocol import CaseDefinition
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.kinematics.multibody_surface import MultiSurfaceKinematics
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import Q16NativeV5MSurface
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)

DTYPE = torch.float64

_FIELD_NAMES = (
    "panel_rings_I", "panel_ring_velocity_I",
    "collocation_I", "collocation_velocity_I",
    "normals_I", "areas", "leading_edge_I", "trailing_edge_I",
    "leading_velocity_I", "trailing_velocity_I",
)

# Fields whose leading dimension is the per-surface panel count, vs the
# leading/trailing edge node count (spanwise_panels + 1).
_PANEL_FIELDS = (
    "panel_rings_I", "panel_ring_velocity_I",
    "collocation_I", "collocation_velocity_I", "normals_I", "areas",
)
_NODE_FIELDS = (
    "leading_edge_I", "trailing_edge_I",
    "leading_velocity_I", "trailing_velocity_I",
)


def _mirror_frame_y(frame: SurfaceFrame, surface_id: str) -> SurfaceFrame:
    """Mirror a frame in y (right wing -> left wing) by pure y-negation.

    Panel/strip/node indices are preserved: panel k of the mirrored wing is
    the y-mirror of panel k of the original (root stays at index 0).  Points
    and velocity vectors get y -> -y.  Because an odd-dimensioned mirror
    reverses ring winding, the stored normals are fully negated (-M n) so
    the production winding convention (stored = -normalize of the corner
    cross product) continues to hold on the mirrored surface.  Mirroring
    twice reproduces the input tensors exactly.
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


def _stored_normal_windings_agree(rings: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Dot(normal, geometric corner-cross normal) per panel.

    The production convention has stored = -normalize((c1-c0) x (c3-c0));
    this returns the dot of the STORED normal with that raw cross product so
    callers can assert the sign is unchanged under mirroring.
    """
    cross = torch.cross(rings[:, 1] - rings[:, 0], rings[:, 3] - rings[:, 0], dim=-1)
    return (normals * cross).sum(dim=-1)


@unittest.skipUnless(torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required")
class U4MultiSurfaceTopologyTest(unittest.TestCase):
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
        cls.right = SurfaceFrame.from_native_geometry(
            cls.native.evaluate(cls.state, cls.zero_velocity),
            surface_id="right_wing",
            body_id="body_0",
        )
        cls.left = _mirror_frame_y(cls.right, surface_id="left_wing")
        cls.frames = (cls.right, cls.left)
        cls.topology = MultiSurfaceTopology.from_surface_frames(cls.frames)

    # -- offsets ------------------------------------------------------------

    def test_topology_offsets_and_totals(self):
        topo = self.topology
        self.assertEqual(topo.total_panels, 900)          # 2 x 450
        self.assertEqual(topo.total_strips, 60)           # 2 x 30
        self.assertEqual(topo.total_le_nodes, 62)         # 2 x 31
        right, left = topo.surfaces
        self.assertIsInstance(right, SurfaceOffsets)
        self.assertEqual(
            (right.surface_id, right.panel_offset, right.panel_count,
             right.strip_offset, right.strip_count,
             right.le_node_offset, right.le_node_count),
            ("right_wing", 0, 450, 0, 30, 0, 31),
        )
        self.assertEqual(
            (left.surface_id, left.panel_offset, left.panel_count,
             left.strip_offset, left.strip_count,
             left.le_node_offset, left.le_node_count),
            ("left_wing", 450, 450, 30, 30, 31, 31),
        )

    def test_offsets_match_frame_counts(self):
        for surface, frame in zip(self.topology.surfaces, self.frames):
            self.assertEqual(surface.panel_count,
                             frame.chordwise_panels * frame.spanwise_panels)
            self.assertEqual(surface.strip_count, frame.spanwise_panels)
            self.assertEqual(surface.le_node_count, frame.spanwise_panels + 1)

    # -- index mapping ------------------------------------------------------

    def test_global_local_round_trip_exhaustive(self):
        topo = self.topology
        for global_panel in range(topo.total_panels):
            surface_id = topo.surface_of_panel(global_panel)
            local = topo.local_panel_index(global_panel)
            self.assertEqual(topo.global_panel_index(surface_id, local), global_panel)
        # Offsets are exact, not just monotone.
        self.assertEqual(topo.global_panel_index("right_wing", 0), 0)
        self.assertEqual(topo.global_panel_index("right_wing", 449), 449)
        self.assertEqual(topo.global_panel_index("left_wing", 0), 450)
        self.assertEqual(topo.global_panel_index("left_wing", 449), 899)

    def test_surface_of_panel_boundaries(self):
        topo = self.topology
        self.assertEqual(topo.surface_of_panel(0), "right_wing")
        self.assertEqual(topo.surface_of_panel(449), "right_wing")
        self.assertEqual(topo.surface_of_panel(450), "left_wing")
        self.assertEqual(topo.surface_of_panel(899), "left_wing")

    def test_out_of_range_and_unknown_surface_errors(self):
        topo = self.topology
        for bad in (-1, 900, 10_000):
            with self.assertRaises(IndexError):
                topo.surface_of_panel(bad)
            with self.assertRaises(IndexError):
                topo.local_panel_index(bad)
        with self.assertRaises(IndexError):
            topo.global_panel_index("right_wing", 450)
        with self.assertRaises(IndexError):
            topo.global_panel_index("left_wing", -1)
        with self.assertRaises(KeyError):
            topo.global_panel_index("tail_wing", 0)
        with self.assertRaises(KeyError):
            topo.offsets_of("tail_wing")

    def test_offsets_of_accessor(self):
        self.assertIs(self.topology.offsets_of("left_wing"), self.topology.surfaces[1])

    # -- concatenation ------------------------------------------------------

    def test_concatenate_shapes_and_block_identity(self):
        concatenated = concatenate_surface_frames(self.frames)
        self.assertEqual(len(concatenated), 10)
        rings, ring_vel, collocation, collocation_vel, normals, areas, \
            leading, trailing, leading_vel, trailing_vel = concatenated
        self.assertEqual(rings.shape, (900, 4, 3))
        self.assertEqual(ring_vel.shape, (900, 4, 3))
        self.assertEqual(collocation.shape, (900, 3))
        self.assertEqual(collocation_vel.shape, (900, 3))
        self.assertEqual(normals.shape, (900, 3))
        self.assertEqual(areas.shape, (900,))
        self.assertEqual(leading.shape, (62, 3))
        self.assertEqual(trailing.shape, (62, 3))
        self.assertEqual(leading_vel.shape, (62, 3))
        self.assertEqual(trailing_vel.shape, (62, 3))
        for tensor in concatenated:
            self.assertTrue(tensor.is_cuda)
            self.assertEqual(tensor.dtype, torch.float64)
        # Per-surface blocks are the frames' own tensors, in topology order.
        n_panels = self.right.panel_rings_I.shape[0]      # 450 per surface
        n_nodes = self.right.leading_edge_I.shape[0]      # 31 per surface
        for name, global_t in zip(_FIELD_NAMES, concatenated):
            count = n_panels if name in _PANEL_FIELDS else n_nodes
            self.assertTrue(torch.equal(global_t[:count], getattr(self.right, name)))
            self.assertTrue(torch.equal(global_t[count:], getattr(self.left, name)))

    def test_concatenate_requires_frames(self):
        with self.assertRaises(ValueError):
            concatenate_surface_frames(())

    # -- U4-P parity gate: single surface in the container ------------------

    def test_single_surface_container_matches_solo_result(self):
        topo = MultiSurfaceTopology.from_surface_frames((self.right,))
        self.assertEqual(topo.total_panels, 450)
        self.assertEqual(topo.total_strips, 30)
        self.assertEqual(topo.total_le_nodes, 31)
        self.assertEqual(topo.surfaces[0].panel_offset, 0)
        self.assertEqual(topo.surfaces[0].strip_offset, 0)
        self.assertEqual(topo.surfaces[0].le_node_offset, 0)
        self.assertEqual(topo.surface_of_panel(449), "right_wing")
        concatenated = concatenate_surface_frames((self.right,))
        for name, global_t in zip(_FIELD_NAMES, concatenated):
            self.assertTrue(torch.equal(global_t, getattr(self.right, name)))
            self.assertEqual(global_t.shape, getattr(self.right, name).shape)

    # -- mirrored wing geometry ---------------------------------------------

    def test_mirror_is_a_physical_left_wing(self):
        left = self.left
        # Occupies y <= 0, root still at node index 0 on the symmetry plane.
        self.assertLessEqual(float(left.panel_rings_I[..., 1].max()), 1e-12)
        le_y = left.leading_edge_I[:, 1]
        self.assertEqual(float(le_y[0]), 0.0)           # root LE on y = 0
        self.assertLess(float(le_y[-1]), -0.13)         # tip near -0.1375
        self.assertTrue(torch.all(le_y[1:] < le_y[:-1]))
        # Same panel counts and areas (rigid mirror).
        self.assertEqual(left.chordwise_panels, self.right.chordwise_panels)
        self.assertEqual(left.spanwise_panels, self.right.spanwise_panels)
        self.assertTrue(torch.equal(left.areas, self.right.areas))
        # Panel k of the left wing is the y-mirror of panel k of the right.
        mirrored_rings = self.right.panel_rings_I.clone()
        mirrored_rings[..., 1] = -mirrored_rings[..., 1]
        self.assertTrue(torch.equal(left.panel_rings_I, mirrored_rings))
        # The production winding convention (stored = -corner-cross) holds on
        # both wings: same sign on every panel, exactly.
        right_dots = _stored_normal_windings_agree(
            self.right.panel_rings_I, self.right.normals_I)
        left_dots = _stored_normal_windings_agree(
            left.panel_rings_I, left.normals_I)
        self.assertTrue(torch.all(right_dots < 0.0))
        self.assertTrue(torch.all(left_dots < 0.0))
        # Stored normals are the negated mirrored normals (-M n) and stay
        # unit length.
        mirrored_normals = self.right.normals_I.clone()
        mirrored_normals[:, 1] = -mirrored_normals[:, 1]
        self.assertTrue(torch.equal(left.normals_I, -mirrored_normals))
        torch.testing.assert_close(
            left.normals_I.norm(dim=-1),
            torch.ones(left.normals_I.shape[0], dtype=DTYPE, device=left.normals_I.device),
        )

    def test_mirror_is_an_involution(self):
        round_trip = _mirror_frame_y(self.left, surface_id="right_wing")
        for name in _FIELD_NAMES:
            self.assertTrue(
                torch.equal(getattr(round_trip, name), getattr(self.right, name)),
                msg=f"{name} changed under double mirror",
            )

    # -- asymmetric grids ----------------------------------------------------

    def test_asymmetric_grid_topology(self):
        small_native = Q16NativeV5MSurface(
            self.mesh,
            q16_chordwise_elements=FORMAL_Q16_GRID[0],
            q16_spanwise_elements=FORMAL_Q16_GRID[1],
            aerodynamic_chordwise_panels=10,
            aerodynamic_spanwise_panels=20,
            device=config.DEVICE,
        )
        small = SurfaceFrame.from_native_geometry(
            small_native.evaluate(self.state, self.zero_velocity),
            surface_id="tail_wing",
            body_id="body_0",
        )
        self.assertEqual((small.chordwise_panels, small.spanwise_panels), (10, 20))

        topo = MultiSurfaceTopology.from_surface_frames((self.right, small))
        self.assertEqual(topo.total_panels, 650)         # 450 + 200
        self.assertEqual(topo.total_strips, 50)          # 30 + 20
        self.assertEqual(topo.total_le_nodes, 52)        # 31 + 21
        big, little = topo.surfaces
        self.assertEqual((big.panel_offset, big.panel_count), (0, 450))
        self.assertEqual((little.panel_offset, little.panel_count), (450, 200))
        self.assertEqual((little.strip_offset, little.strip_count), (30, 20))
        self.assertEqual((little.le_node_offset, little.le_node_count), (31, 21))
        # Round trips cross the dissimilar boundary.
        self.assertEqual(topo.surface_of_panel(449), "right_wing")
        self.assertEqual(topo.surface_of_panel(450), "tail_wing")
        self.assertEqual(topo.global_panel_index("tail_wing", 199), 649)
        for global_panel in (0, 449, 450, 649):
            self.assertEqual(
                topo.global_panel_index(
                    topo.surface_of_panel(global_panel),
                    topo.local_panel_index(global_panel),
                ),
                global_panel,
            )
        concatenated = concatenate_surface_frames((self.right, small))
        rings, _, collocation, _, normals, areas, leading, _, _, _ = concatenated
        self.assertEqual(rings.shape, (650, 4, 3))
        self.assertEqual(collocation.shape, (650, 3))
        self.assertEqual(normals.shape, (650, 3))
        self.assertEqual(areas.shape, (650,))
        self.assertEqual(leading.shape, (52, 3))
        self.assertTrue(torch.equal(rings[450:], small.panel_rings_I))
        self.assertTrue(torch.equal(leading[31:], small.leading_edge_I))
        self.assertTrue(torch.equal(areas[:450], self.right.areas))


# ---------------------------------------------------------------------------
# CPU contracts: Meng case config + MultiSurfaceKinematics
# ---------------------------------------------------------------------------

def _make_reference_frame(nc: int = 2, ns: int = 2, surface_id: str = "ref_wing") -> SurfaceFrame:
    """A static (zero-velocity) reference wing: nc x ns panels on z=0."""
    xs = torch.linspace(0.0, 1.0, nc + 1, dtype=DTYPE)
    ys = torch.linspace(0.0, 1.0, ns + 1, dtype=DTYPE)
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
    panel_rings = torch.stack(rings)                      # (n, 4, 3)
    collocation = panel_rings.mean(dim=1)                 # (n, 3)
    n = panel_rings.shape[0]
    normals = torch.tile(torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE), (n, 1))
    areas = torch.full((n,), 1.0 / n, dtype=DTYPE)
    leading = corners[0, :, :].clone()                    # x = 0 edge (ns+1, 3)
    trailing = corners[-1, :, :].clone()                  # x = nc edge

    return SurfaceFrame(
        surface_id=surface_id,
        body_id="body_0",
        panel_rings_I=panel_rings,
        panel_ring_velocity_I=torch.zeros_like(panel_rings),
        collocation_I=collocation,
        collocation_velocity_I=torch.zeros_like(collocation),
        normals_I=normals,
        areas=areas,
        leading_edge_I=leading,
        trailing_edge_I=trailing,
        leading_velocity_I=torch.zeros_like(leading),
        trailing_velocity_I=torch.zeros_like(trailing),
        chordwise_panels=nc,
        spanwise_panels=ns,
        topology_digest=f"test:ref:{nc}x{ns}",
    )


def _identity_law(t: float):
    return (
        torch.zeros(3, dtype=DTYPE),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
        torch.zeros(3, dtype=DTYPE),
        torch.zeros(3, dtype=DTYPE),
    )


IDENTITY_BODY_POSE = (
    torch.zeros(3, dtype=DTYPE),
    torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
)


class U4MengCaseAndMultiSurfaceKinematicsTest(unittest.TestCase):
    def test_meng_case_config_defaults(self):
        case = MengCaseConfig()
        self.assertEqual(case.case_id, "MENG-2025")
        self.assertEqual(case.n_surfaces, 2)
        self.assertEqual(case.surface_ids, ("left_wing", "right_wing"))
        self.assertEqual(case.body_id, "body_0")
        self.assertEqual(case.frequency_hz, 2.0)
        self.assertEqual(case.half_span_m, 0.800)
        self.assertEqual(case.freestream_m_s, 8.0)

    def test_meng_config_frozen_and_registry(self):
        with self.assertRaises(Exception):
            MENG_PRIMARY.case_id = "mutated"  # type: ignore[misc]
        self.assertIsInstance(MENG_PRIMARY, MengCaseConfig)
        self.assertIsInstance(MENG_PRIMARY, CaseDefinition)
        self.assertEqual(MENG_PRIMARY.case_id, "MENG-2025")

    def test_multi_surface_kinematics_aggregates_in_order(self):
        right = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(2, 2, "right_wing"), _identity_law,
            surface_id="right_wing", body_id="body_0",
        )
        left = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(2, 2, "left_wing"), _identity_law,
            surface_id="left_wing", body_id="body_0",
        )
        multi = MultiSurfaceKinematics([right, left])
        self.assertEqual(multi.surface_ids, ("right_wing", "left_wing"))

        frames = multi.evaluate(IDENTITY_BODY_POSE, 0.25)
        self.assertEqual(len(frames), 2)
        self.assertEqual([f.surface_id for f in frames], ["right_wing", "left_wing"])
        for frame, kin in zip(frames, (right, left)):
            self.assertIsInstance(frame, SurfaceFrame)
            self.assertEqual(frame.body_id, "body_0")
            self.assertTrue(torch.equal(frame.panel_rings_I, kin.evaluate(0.25).panel_rings_I))

    def test_multi_surface_kinematics_topology_round_trip(self):
        # The aggregated frames feed straight into the U4 topology.
        right = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(3, 4, "right_wing"), _identity_law,
            surface_id="right_wing", body_id="body_0",
        )
        left = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(3, 4, "left_wing"), _identity_law,
            surface_id="left_wing", body_id="body_0",
        )
        frames = MultiSurfaceKinematics([right, left]).evaluate(IDENTITY_BODY_POSE, 0.0)
        topo = MultiSurfaceTopology.from_surface_frames(frames)
        self.assertEqual(topo.total_panels, 24)
        self.assertEqual(topo.total_strips, 8)
        self.assertEqual(topo.total_le_nodes, 10)
        self.assertEqual(topo.global_panel_index("left_wing", 0), 12)
        self.assertEqual(topo.global_panel_index("left_wing", 11), 23)
        self.assertEqual(topo.surface_of_panel(12), "left_wing")

    def test_multi_surface_kinematics_rejects_empty(self):
        with self.assertRaises(ValueError):
            MultiSurfaceKinematics([])


if __name__ == "__main__":
    unittest.main()
