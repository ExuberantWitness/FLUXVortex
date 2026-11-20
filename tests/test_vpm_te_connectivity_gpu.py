"""TE wake-sheet connectivity contract (geometry red test).

Guards the two defects proven by
``research/P1_SHARED_EDGE_TIME_DISCRETE_DERIVATION_AND_REVIEW_97E7C92_20260906``
with the real 15x30 topology:

1. the previous newest row's CONVECTED front edge was still re-anchored back
   to the wing trailing edge, so the sheet never stayed connected;
2. the convected front was written into the newborn row in the same order,
   producing a crossed quadrilateral whose shared edge ran in the SAME
   direction (cosine +1) instead of the opposite one.

Required endpoint identity for the newborn row ``k`` and the next row
``k + 1``::

    r[k, 3] == r[k + 1, 0]      r[k, 2] == r[k + 1, 1]

so the shared edge vectors are exact negatives.  Finite forces, a zero
normal residual or "particles > 0" do NOT imply this geometry: at zero
incidence the wrong geometry still yields zero load and a clean gate.

This module intentionally checks geometry only; it makes no claim about
material history, circulation ledgers or steady-state accuracy.
"""

from __future__ import annotations

import pytest
import torch

import queue_roj_rigid_fig9_12_13_15 as frozen
from test_q16_joint_deposited_flow_gpu import make_case

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

FREESTREAM = 5.0
DT = frozen.DT_STAR * frozen.CHORD_M / FREESTREAM
STREAMWISE_SEGMENT = FREESTREAM * DT  # 6.88e-4 m for the frozen Roj case
ANALYTIC_NEWBORN_AREA = (
    STREAMWISE_SEGMENT * frozen.SPAN_M / frozen.AERO_GRID[1]
)


def _run(alpha: float, joint: bool, steps: int = 3):
    """Return the last proposal plus the committed owner after ``steps``."""
    solver, stepper, owner, kin = make_case(alpha, joint)
    dt = solver.settings.aerodynamic_dt
    proposal = None
    for step in range(1, steps + 1):
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        stepper.commit(owner, proposal)
    # One more proposal so the newborn row is present and inspectable.
    proposal = stepper.propose(owner, (kin.evaluate((steps + 1) * dt),), dt)
    return solver, stepper, owner, kin, proposal


def _edges(proposal):
    """Newest row's back edge, next row's front edge, and the joins."""
    ns = frozen.AERO_GRID[1]
    chain = proposal.trial_state.wake_rings.reshape(-1, ns, 4, 3)
    assert chain.shape[0] >= 2, "need at least two wake rows"
    back = chain[:-1, :, [3, 2]]  # newest rows' downstream edge (3 -> 2)
    front = chain[1:, :, :2]  # next rows' upstream edge (0 -> 1)
    new_edge = chain[0, :, 3] - chain[0, :, 2]
    old_edge = chain[1, :, 1] - chain[1, :, 0]
    cosine = (new_edge * old_edge).sum(-1) / (
        new_edge.norm(dim=-1) * old_edge.norm(dim=-1)
    )
    area = 0.5 * torch.cross(chain[0], chain[0].roll(-1, dims=1), dim=-1).sum(1)
    return {
        "chain": chain,
        "join_gap": (back - front).norm(dim=-1).max().item(),
        "cosine_min": cosine.min().item(),
        "cosine_max": cosine.max().item(),
        "newest_area_max": area.norm(dim=-1).max().item(),
    }


@pytest.mark.parametrize("alpha", [0.0, 5.0, 15.0, 19.0])
def test_joint_rows_are_endpoint_connected(alpha):
    _, _, _, _, proposal = _run(alpha, joint=True)
    e = _edges(proposal)
    # Exact endpoint coincidence: r[k,3]==r[k+1,0], r[k,2]==r[k+1,1].
    assert e["join_gap"] <= 1.0e-12, f"A{alpha:g}: sheet is disconnected"
    # Opposite traversal direction (was +1.0 with the crossed order).
    assert e["cosine_max"] <= -1.0 + 1.0e-10, (
        f"A{alpha:g}: shared edge direction is not reversed "
        f"(cos in [{e['cosine_min']}, {e['cosine_max']}])"
    )


def test_joint_newborn_polygon_area_is_physical():
    """At zero incidence the newborn ring must be the analytic rectangle.

    A crossed / coincident pair collapses to ~1e-19 m^2, which the old code
    produced while every force-based gate still passed.
    """
    _, _, _, _, proposal = _run(0.0, joint=True)
    area = _edges(proposal)["newest_area_max"]
    assert area > 0.0
    assert abs(area / ANALYTIC_NEWBORN_AREA - 1.0) <= 1.0e-10, (
        f"newborn area {area:.6e} != analytic {ANALYTIC_NEWBORN_AREA:.6e}"
    )


@pytest.mark.parametrize("alpha", [0.0, 5.0, 15.0, 19.0])
def test_joint_keeps_previous_front_convected(alpha):
    """The old front must stay ~U*dt downstream of the TE, not be re-anchored."""
    solver, _, _, _, proposal = _run(alpha, joint=True)
    ns = frozen.AERO_GRID[1]
    nc = solver.settings.chordwise_panels
    chain = proposal.trial_state.wake_rings.reshape(-1, ns, 4, 3)
    geometry = solver.surface.evaluate(None, None)
    te = geometry.rings.reshape(nc, ns, 4, 3)[-1, :, [3, 2]]
    front_to_te = (chain[1, :, :2] - te).norm(dim=-1).max().item()
    assert front_to_te > 0.5 * STREAMWISE_SEGMENT, (
        "previous row's front was re-anchored to the trailing edge "
        f"(distance {front_to_te:.3e} m, expected ~{STREAMWISE_SEGMENT:.3e})"
    )


def test_legacy_path_keeps_historical_reanchored_geometry():
    """Default (flag OFF) path must keep its historical TE re-anchor.

    Pinned as a property (front sits ON the trailing edge) so the frozen
    production numbers stay guarded even if this test's neighbours evolve.
    """
    solver, _, _, _, proposal = _run(5.0, joint=False)
    ns = frozen.AERO_GRID[1]
    nc = solver.settings.chordwise_panels
    chain = proposal.trial_state.wake_rings.reshape(-1, ns, 4, 3)
    geometry = solver.surface.evaluate(None, None)
    te = geometry.rings.reshape(nc, ns, 4, 3)[-1, :, [3, 2]]
    front_to_te = (chain[1, :, :2] - te).norm(dim=-1).max().item()
    assert front_to_te == 0.0, (
        "legacy path must keep re-anchoring the newest front to the TE; "
        f"observed {front_to_te:.3e} m"
    )
    # And its streamwise edge (v1 -> v2) is still the fixed U*dt segment.
    # Ring layout is (rear[1], rear[0], down[0], down[1]): v0->v1 and v2->v3
    # are spanwise (b/ns), v1->v2 and v3->v0 are streamwise.
    streamwise = (chain[0, :, 2] - chain[0, :, 1]).norm(dim=-1).mean().item()
    assert abs(streamwise / STREAMWISE_SEGMENT - 1.0) <= 1.0e-12, (
        f"legacy streamwise edge {streamwise:.6e} != U*dt {STREAMWISE_SEGMENT:.6e}"
    )
