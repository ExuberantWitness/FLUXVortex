"""S3a contract: caching the repeated tangent linearization per batch width is bitwise.

``_refresh_reference_tangent_body`` probes the tangent column by column in batches of
``tangent_batch_size``.  For every batch it rebuilt ``repeated_linearization`` -- the
frozen linearization tuple expanded to the batch width and made contiguous -- and the
measured cost of that rebuild is the bulk of the ~467 ms/step non-kernel part of a
refresh (S2b).  Only TWO distinct widths exist on the formal mesh (the batch width and
the last partial batch), so the rebuild runs 47 times where 2 would do.

The risk this test covers is not arithmetic but aliasing: if
``_tangent_action_linearized_prechecked`` wrote into those arrays, reusing one copy across
batches would silently feed stale scratch into later batches.  It does not -- the
linearization goes in through ``inputs=[...]`` and its own ``delta_*`` come from
``wp.zeros_like`` -- and this test proves the tangent matrix is identical bit for bit
between the two paths on the real operator and the real case geometry.
"""

from __future__ import annotations

import os

import pytest
import torch
import warp as wp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def built_runner():
    from fluxvortex.cases.rojratsirikul2011 import ROJRATSIRIKUL2011_UNIFIED_CASES
    from fluxvortex.runtime.case_runner import RojratsirikulCaseRunner

    runner = RojratsirikulCaseRunner(
        ROJRATSIRIKUL2011_UNIFIED_CASES["ROJ11-A16"], device="cuda:0"
    )
    runner.build()
    runner.run(max_aero_steps=1)
    return runner


def test_flag_defaults_on():
    from fluxvortex.warp_fsi import q16_structural_solver as module

    assert module._TANGENT_LINEARIZATION_REUSE is True


def test_reused_linearization_gives_the_identical_tangent(built_runner, monkeypatch):
    """Force two refreshes of the SAME state, once per path, and compare exactly."""

    from fluxvortex.warp_fsi import q16_structural_solver as module

    stepper = built_runner.structural
    state = built_runner.owner.state

    # Path A: cached linearization (the default).
    monkeypatch.setattr(module, "_TANGENT_LINEARIZATION_REUSE", True)
    stepper._reference_tangent_anchor_torch = None  # force the body to run
    assert stepper.refresh_reference_tangent(state) is True
    cached = stepper._reference_tangent_matrix_torch.clone()

    # Path B: rebuild the linearization on every batch.
    monkeypatch.setattr(module, "_TANGENT_LINEARIZATION_REUSE", False)
    stepper._reference_tangent_anchor_torch = None
    assert stepper.refresh_reference_tangent(state) is True
    rebuilt = stepper._reference_tangent_matrix_torch.clone()

    assert cached.shape == rebuilt.shape
    assert torch.equal(cached, rebuilt), (
        "the per-batch-width cache changed the tangent: "
        f"max|delta| = {float((cached - rebuilt).abs().max().item()):.3e}"
    )


def test_tangent_is_finite_and_square(built_runner):
    stepper = built_runner.structural
    tangent = stepper._reference_tangent_matrix_torch
    assert tangent is not None
    assert tangent.shape == (stepper.dof_count, stepper.dof_count)
    assert bool(torch.isfinite(tangent).all().item())
    assert float(tangent.abs().max().item()) > 0.0
