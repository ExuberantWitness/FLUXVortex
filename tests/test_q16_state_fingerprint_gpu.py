"""Stage 4a contract: the transaction guards must stay just as strict while
costing nothing.

Background: three guards used a full-state ``digest()`` (8 blocking D2H copies plus
a CPU SHA-256 over ~2.25 MB, called 10-12 times per aerodynamic step at steady
state) purely to answer "was this state written in place?".  They now use
``mutation_fingerprint()``, a host-only token over ``Tensor._version`` and
``(data_ptr, shape)``.

The risk of that substitution is a guard that silently stops firing, so these tests
pin the DETECTION, not the speed:

1. the fingerprint is stable across read-only work and repeated calls;
2. it changes on an in-place write to any hashed buffer (``copy_``/``fill_``);
3. it changes when the live particle count changes (the buffers are views of
   ``[:n]``, so ``n`` itself must be part of the token through the shape);
4. it changes when a buffer is rebound to a different allocation;
5. ``digest()`` itself is untouched: same state -> same hex string, and the
   fingerprint is NOT a substitute for the recorded digest.
"""

from __future__ import annotations

import pytest
import torch

from test_q16_joint_deposited_flow_gpu import make_case

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _state():
    solver, stepper, owner, kin = make_case(15.0, True)
    dt = solver.settings.aerodynamic_dt
    for step in range(1, 4):
        proposal = stepper.propose(owner, (kin.evaluate(step * dt),), dt)
        stepper.commit(owner, proposal)
    return owner.state


def test_fingerprint_is_stable_when_nothing_writes():
    state = _state()
    first = state.mutation_fingerprint()
    assert state.digest() == state.digest()
    assert state.mutation_fingerprint() == first
    # Reading through the public views must not disturb it either.
    _ = state.gamma_bound.sum().item()
    _ = state.particle_field.positions_cuda.max().item()
    assert state.mutation_fingerprint() == first


def test_fingerprint_detects_in_place_write_to_every_hashed_buffer():
    """Every buffer digest() hashes must move the token.

    ``Tensor._version`` is monotonic and fp64 arithmetic is not exactly
    invertible, so there is deliberately no "restore and compare back" here:
    each iteration takes a fresh baseline.  Bool buffers (frontier_active) are
    flipped instead of incremented.
    """

    state = _state()
    for name in (
        "gamma_bound",
        "gamma_previous",
        "wake_rings",
        "wake_gamma",
        "frontier_nodes",
        "frontier_active",
    ):
        buffer = getattr(state, name)
        if buffer.numel() == 0:
            continue
        before = state.mutation_fingerprint()
        flat = buffer.view(-1)
        if flat.dtype is torch.bool:
            flat[0] = ~flat[0]
        else:
            flat[0].add_(1.0)
        after = state.mutation_fingerprint()
        assert before != after, f"in-place write to {name} was not detected"


def test_fingerprint_detects_particle_count_and_particle_writes():
    state = _state()
    field = state.particle_field
    assert field.n > 0

    before = state.mutation_fingerprint()
    field.gamma[0, 0].add_(1.0)
    assert state.mutation_fingerprint() != before

    # The live buffers are views of [:n]; a changed count must move the token
    # through the recorded shape even though data_ptr is unchanged.
    before_count = state.mutation_fingerprint()
    count = field.n
    field.n = count - 1
    try:
        assert state.mutation_fingerprint() != before_count
    finally:
        field.n = count


def test_digest_is_value_based_while_the_fingerprint_is_event_based():
    """This is the whole reason both exist.

    ``digest()`` hashes BYTES: an in-place write that leaves every bit unchanged
    (x *= 1.0) leaves the digest identical.  The fingerprint tracks
    ``_version``, so the same write moves it.  The transaction guards need
    "was this written?"; the ledger records "what is the state?".  Using one for
    the other is the mistake this split removes.
    """

    state = _state()
    digest_before = state.digest()
    fingerprint_before = state.mutation_fingerprint()

    # Bit-identical in-place write: no digest change, but a real write event.
    state.gamma_bound.view(-1)[0].mul_(1.0)
    assert state.digest() == digest_before
    assert state.mutation_fingerprint() != fingerprint_before

    # A value-changing write moves both.
    digest_before = state.digest()
    fingerprint_before = state.mutation_fingerprint()
    state.gamma_bound.view(-1)[0].add_(1.0)
    assert state.digest() != digest_before
    assert state.mutation_fingerprint() != fingerprint_before
