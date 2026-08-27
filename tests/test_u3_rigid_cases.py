"""U3 rigid cases: prescribed rigid kinematics, one-way coupling, case configs.

The three rigid paper cases (Baik W1-W4, Yang 2025, Izraelevitz-Scherer)
migrate onto the same unified framework as the Q16 FSI path (plan §11.2-11.4,
§14 U3).  These CPU-only tests pin the new contracts:

- ``PrescribedRigidSurfaceKinematics``: rigid transform of a reference
  ``SurfaceFrame`` (identity, translation, 90-degree rotation, heave velocity);
- ``OneWayPrescribedCoupling``: exactly one propose + one commit per step and
  a single generation increment (no FSI iteration);
- the per-case unified configs and the ``CaseDefinition`` protocol;
- ``ResultStatus`` usage for rigid cases: accuracy gate is
  ``not_applicable`` without ground truth, ``passed``/``failed`` with it.
"""
from __future__ import annotations

import math

import pytest
import torch

from fluxvortex.cases import izraelevitz2017
from fluxvortex.cases.baik2011 import BAIK_CASES, BaikCaseConfig
from fluxvortex.cases.izraelevitz2017 import IZRA_CASES, IzraCaseConfig
from fluxvortex.cases.protocol import CaseDefinition
from fluxvortex.cases.rojratsirikul2011 import ROJ_A16_PRIMARY
from fluxvortex.cases.yang2025 import YANG_CASES, YangCaseConfig
from fluxvortex.coupling.one_way import OneWayPrescribedCoupling
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.runtime.result_schema import ResultStatus
from fluxvortex.state.world import WorldDynamicState, WorldOwner

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Fixtures: reference SurfaceFrame over the unit square
# ---------------------------------------------------------------------------

def _make_reference_frame(nc: int = 2, ns: int = 2) -> SurfaceFrame:
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
        surface_id="ref_wing",
        body_id="ref_body",
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
        topology_digest="test:rigid-ref",
    )


def _identity_law(t: float):
    return (
        torch.zeros(3, dtype=DTYPE),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
        torch.zeros(3, dtype=DTYPE),
        torch.zeros(3, dtype=DTYPE),
    )


# ---------------------------------------------------------------------------
# 1. PrescribedRigidSurfaceKinematics
# ---------------------------------------------------------------------------

class TestPrescribedRigidSurfaceKinematics:
    def test_identity_motion_returns_reference_exactly(self):
        ref = _make_reference_frame()
        kin = PrescribedRigidSurfaceKinematics(
            ref, _identity_law, surface_id="rigid_wing", body_id="rigid_body"
        )
        frame = kin.evaluate(0.7)

        assert isinstance(frame, SurfaceFrame)
        # Rigid identity motion: every transformed field equals the reference.
        assert torch.equal(frame.panel_rings_I, ref.panel_rings_I)
        assert torch.equal(frame.panel_ring_velocity_I, ref.panel_ring_velocity_I)
        assert torch.equal(frame.collocation_I, ref.collocation_I)
        assert torch.equal(frame.collocation_velocity_I, ref.collocation_velocity_I)
        assert torch.equal(frame.normals_I, ref.normals_I)
        assert torch.equal(frame.areas, ref.areas)
        assert torch.equal(frame.leading_edge_I, ref.leading_edge_I)
        assert torch.equal(frame.trailing_edge_I, ref.trailing_edge_I)
        assert torch.equal(frame.leading_velocity_I, ref.leading_velocity_I)
        assert torch.equal(frame.trailing_velocity_I, ref.trailing_velocity_I)
        assert frame.chordwise_panels == ref.chordwise_panels
        assert frame.spanwise_panels == ref.spanwise_panels
        assert frame.topology_digest == ref.topology_digest

    def test_surface_ids_contract(self):
        kin = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(), _identity_law, surface_id="wing_7", body_id="body_1"
        )
        assert kin.surface_ids == ("wing_7",)
        frame = kin.evaluate(0.0)
        assert frame.surface_id == "wing_7"
        assert frame.body_id == "body_1"

    def test_pure_translation_shifts_points_and_keeps_normals(self):
        ref = _make_reference_frame()
        offset = torch.tensor([2.0, -3.0, 0.5], dtype=DTYPE)
        lin_vel = torch.tensor([0.25, 0.0, -1.0], dtype=DTYPE)

        def law(t: float):
            return (
                offset,
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                lin_vel,
                torch.zeros(3, dtype=DTYPE),
            )

        frame = PrescribedRigidSurfaceKinematics(
            ref, law, surface_id="s", body_id="b"
        ).evaluate(1.3)

        for got, want in (
            (frame.panel_rings_I, ref.panel_rings_I),
            (frame.collocation_I, ref.collocation_I),
            (frame.leading_edge_I, ref.leading_edge_I),
            (frame.trailing_edge_I, ref.trailing_edge_I),
        ):
            torch.testing.assert_close(got, want + offset)
        # Normals are directions: translation leaves them unchanged.
        torch.testing.assert_close(frame.normals_I, ref.normals_I)
        # Pure translation: every point moves with the body velocity.
        torch.testing.assert_close(frame.collocation_velocity_I, lin_vel.expand_as(ref.collocation_I))
        torch.testing.assert_close(
            frame.panel_ring_velocity_I, lin_vel.expand_as(ref.panel_rings_I)
        )
        torch.testing.assert_close(frame.areas, ref.areas)

    def test_rotation_90deg_about_z(self):
        ref = _make_reference_frame()
        c = math.cos(math.pi / 4.0)
        s = math.sin(math.pi / 4.0)

        def law(t: float):
            return (
                torch.zeros(3, dtype=DTYPE),
                torch.tensor([c, 0.0, 0.0, s], dtype=DTYPE),
                torch.zeros(3, dtype=DTYPE),
                torch.tensor([0.0, 0.0, 2.0], dtype=DTYPE),
            )

        frame = PrescribedRigidSurfaceKinematics(
            ref, law, surface_id="s", body_id="b"
        ).evaluate(0.0)

        # (x, y, z) -> (-y, x, z) under +90 deg about z.  The first 2x2-grid
        # panel ring is (0,0,0), (0.5,0,0), (0.5,0.5,0), (0,0.5,0).
        ring = frame.panel_rings_I[0]
        torch.testing.assert_close(ring[0], torch.tensor([0.0, 0.0, 0.0], dtype=DTYPE))
        torch.testing.assert_close(ring[1], torch.tensor([0.0, 0.5, 0.0], dtype=DTYPE))
        torch.testing.assert_close(ring[2], torch.tensor([-0.5, 0.5, 0.0], dtype=DTYPE))
        # Trailing-edge corner (1, 1, 0) maps to (-1, 1, 0).
        torch.testing.assert_close(
            frame.trailing_edge_I[-1], torch.tensor([-1.0, 1.0, 0.0], dtype=DTYPE)
        )
        # z-normals are invariant under a z-rotation and stay unit length.
        expected_normals = ref.normals_I.clone()
        torch.testing.assert_close(frame.normals_I, expected_normals)
        torch.testing.assert_close(
            frame.normals_I.norm(dim=-1), torch.ones(frame.normals_I.shape[0], dtype=DTYPE)
        )
        # omega x r at the first collocation point: r = (-0.25, 0.25, 0),
        # omega = (0, 0, 2)  =>  v = (-0.5, -0.5, 0).
        torch.testing.assert_close(
            frame.collocation_velocity_I[0], torch.tensor([-0.5, -0.5, 0.0], dtype=DTYPE)
        )
        # Rigid rotation preserves panel areas and topology bookkeeping.
        torch.testing.assert_close(frame.areas, ref.areas)
        assert frame.chordwise_panels == ref.chordwise_panels
        assert frame.spanwise_panels == ref.spanwise_panels

    def test_heave_velocity_matches_numerical_derivative(self):
        ref = _make_reference_frame()
        amplitude = 0.3
        omega = 2.0 * math.pi * 1.5

        def law(t: float):
            return (
                torch.tensor([0.0, 0.0, amplitude * math.sin(omega * t)], dtype=DTYPE),
                torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
                torch.tensor([0.0, 0.0, amplitude * omega * math.cos(omega * t)], dtype=DTYPE),
                torch.zeros(3, dtype=DTYPE),
            )

        kin = PrescribedRigidSurfaceKinematics(ref, law, surface_id="s", body_id="b")
        t0 = 0.137
        h = 1.0e-4
        frame = kin.evaluate(t0)
        plus = kin.evaluate(t0 + h)
        minus = kin.evaluate(t0 - h)

        numeric = (plus.collocation_I - minus.collocation_I) / (2.0 * h)
        torch.testing.assert_close(frame.collocation_velocity_I, numeric, rtol=0.0, atol=1e-6)

        numeric_rings = (plus.panel_rings_I - minus.panel_rings_I) / (2.0 * h)
        torch.testing.assert_close(
            frame.panel_ring_velocity_I, numeric_rings, rtol=0.0, atol=1e-6
        )

    def test_accepts_plain_python_motion_law_outputs(self):
        """Motion laws returning lists (not tensors) are coerced."""
        ref = _make_reference_frame()

        def law(t: float):
            return ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

        frame = PrescribedRigidSurfaceKinematics(
            ref, law, surface_id="s", body_id="b"
        ).evaluate(0.0)
        torch.testing.assert_close(frame.collocation_I, ref.collocation_I)


# ---------------------------------------------------------------------------
# 2. OneWayPrescribedCoupling
# ---------------------------------------------------------------------------

class _RecordingStepper:
    """Mock AerodynamicStepper: records propose/commit calls."""

    def __init__(self) -> None:
        self.proposal = object()
        self.propose_calls = []
        self.commit_calls = []

    def propose(self, committed, surfaces, dt):
        self.propose_calls.append((committed, surfaces, dt))
        return self.proposal

    def commit(self, owner, proposal):
        self.commit_calls.append((owner, proposal))


def _make_owner(aero_state=None) -> WorldOwner:
    return WorldOwner(
        dynamic_state=WorldDynamicState(
            elastic_states={}, body_states={}, joint_states={}
        ),
        aero_state=object() if aero_state is None else aero_state,
        previous_load=None,
        generation=0,
    )


class TestOneWayPrescribedCoupling:
    def test_single_propose_commit_and_generation_increment(self):
        stepper = _RecordingStepper()
        coupling = OneWayPrescribedCoupling(stepper)
        owner = _make_owner()
        aero_state = owner.aero_state

        kin = PrescribedRigidSurfaceKinematics(
            _make_reference_frame(), _identity_law, surface_id="wing_0", body_id="body_0"
        )
        frames = (kin.evaluate(0.0),)
        dt = 5.0e-3

        returned = coupling.advance(owner, surface_frames=frames, delta_time=dt)

        # Exactly one proposal per step — no FSI iteration.
        assert stepper.propose_calls == [(aero_state, frames, dt)]
        assert len(stepper.commit_calls) == 1
        committed_owner, committed_proposal = stepper.commit_calls[0]
        assert committed_owner is owner
        assert committed_proposal is stepper.proposal
        assert returned is stepper.proposal
        # One committed step advances the owner generation exactly once.
        assert owner.generation == 1

    def test_repeated_advances_do_not_iterate(self):
        stepper = _RecordingStepper()
        coupling = OneWayPrescribedCoupling(stepper)
        owner = _make_owner()
        for step in range(4):
            coupling.advance(owner, surface_frames=(), delta_time=1.0e-3)
        assert len(stepper.propose_calls) == 4
        assert len(stepper.commit_calls) == 4
        assert owner.generation == 4


# ---------------------------------------------------------------------------
# 3. Case configs
# ---------------------------------------------------------------------------

class TestCaseConfigs:
    def test_baik_w1_through_w4(self):
        assert set(BAIK_CASES) == {"W1", "W2", "W3", "W4"}
        for case_id, config in BAIK_CASES.items():
            assert isinstance(config, BaikCaseConfig)
            assert config.case_id == case_id
            assert config.description

    def test_yang_six_installation_angles(self):
        angles = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
        assert set(YANG_CASES) == {f"YANG-{a}deg" for a in angles}
        assert sorted(c.installation_angle_deg for c in YANG_CASES.values()) == list(angles)
        for case_id, config in YANG_CASES.items():
            assert isinstance(config, YangCaseConfig)
            assert config.case_id == case_id

    def test_izra_twelve_condition_contract(self):
        # 12 unique motion conditions — the 14 experimental markers are
        # replicate observations of these conditions, never 14 cases.
        assert len(IZRA_CASES) == 12
        families: dict[float, list[float]] = {}
        for config in IZRA_CASES.values():
            assert isinstance(config, IzraCaseConfig)
            families.setdefault(config.theta_max_deg, []).append(config.phase_offset_deg)
        assert set(families) == {15.0, 25.0}
        assert sorted(families[15.0]) == [15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0]
        assert sorted(families[25.0]) == [45.0, 60.0, 75.0, 90.0, 105.0]
        for case_id, config in IZRA_CASES.items():
            assert config.case_id == case_id
            # Rectangular wing: AR = b/c = 3 and S = b*c.
            assert config.span_m / config.chord_m == pytest.approx(3.0)
            assert config.area_m2 == pytest.approx(config.span_m * config.chord_m)
            assert config.aspect_ratio == pytest.approx(3.0)
            # Pitch axis at the three-quarter chord.
            assert config.pivot_fraction_chord == pytest.approx(0.75)
            # Heave amplitude h/c = 0.6.
            assert config.heave_amplitude_m / config.chord_m == pytest.approx(0.6)
            # Forward water flow with a finite Reynolds number — not a
            # zero-freestream flapper.
            assert config.freestream_m_s > 0.0
            assert config.reynolds == pytest.approx(309676.8)
        # Ground truth: frozen CSV present with the pinned SHA-256.
        assert IZRA_CASES["IZRA-15-015"].ground_truth_path.is_file()
        assert (
            IZRA_CASES["IZRA-15-015"].ground_truth_sha256
            == "993f410c5d4857a221e57c616bf45beb5eaef5391a2deafb0b6e48e6d083b3cf"
        )
        izraelevitz2017.validate_gt()
        # The corrected scientific object: a rectangular finite wing in
        # forward flow.  The wrong prior description must stay gone.
        doc = (izraelevitz2017.__doc__ or "").lower()
        assert "elliptical" not in doc
        assert "hovering" not in doc

    def test_configs_are_frozen(self):
        for config in (
            BAIK_CASES["W1"],
            YANG_CASES["YANG-0.0deg"],
            IZRA_CASES["IZRA-15-015"],
        ):
            with pytest.raises(Exception):
                config.case_id = "mutated"  # type: ignore[misc]

    def test_configs_satisfy_case_definition_protocol(self):
        unified = [*BAIK_CASES.values(), *YANG_CASES.values(), *IZRA_CASES.values()]
        for config in unified + [ROJ_A16_PRIMARY]:
            assert isinstance(config, CaseDefinition)


# ---------------------------------------------------------------------------
# 4. ResultStatus integration for rigid cases
# ---------------------------------------------------------------------------

class TestRigidCaseResultStatus:
    def test_no_ground_truth_accuracy_not_applicable(self):
        # e.g. the Izra placeholder before the GT adapter is wired: the run
        # can complete and still declare the accuracy gate not applicable.
        status = ResultStatus(
            execution_status="completed",
            numerical_status="converged",
            physics_gate_status="passed",
            accuracy_gate_status="not_applicable",
            reproduction_status="pending",
        )
        assert status.accuracy_gate_status == "not_applicable"
        assert status.exit_code == 0
        # Without GT scoring there is no formal reproduction.
        assert not status.is_formal_reproduction

    def test_ground_truth_present_passed_is_formal_reproduction(self):
        status = ResultStatus(
            execution_status="completed",
            numerical_status="converged",
            physics_gate_status="passed",
            accuracy_gate_status="passed",
            reproduction_status="passed",
        )
        assert status.is_formal_reproduction
        assert status.exit_code == 0

    def test_ground_truth_present_failed_exits_nonzero(self):
        status = ResultStatus(
            execution_status="completed",
            numerical_status="converged",
            physics_gate_status="passed",
            accuracy_gate_status="failed",
            reproduction_status="failed",
        )
        assert status.exit_code == 2
        assert not status.is_formal_reproduction
