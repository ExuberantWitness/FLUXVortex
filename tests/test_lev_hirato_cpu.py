"""Leading-edge vortex sheet after Hirato et al. (plan V3, stage 2): chain bookkeeping and the
coupled circulation solve.  CPU only, no solver.

The plate is the Rojratsirikul one as the solver builds it: rings from a quarter panel ahead of
the leading edge to a quarter panel behind the trailing edge, collocation at the ring centres,
with a straight wake tied to the trailing-edge unknowns as a steady state has it.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 -m pytest tests/test_lev_hirato_cpu.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from fluxvortex.warp_fsi import kernels_ring_velocity as kernel
from fluxvortex.warp_fsi import lev_hirato as h

CHORD, SPAN, NC, NS = 0.0688, 0.1375, 15, 30
SPEED, DT_STAR, WAKE_ROWS = 5.0, 0.01, 30
LESP_FACTOR, LESP_CRIT = 1.13, 0.11
AHEAD = (-0.4, 0.0, 0.6)          # free end of a test ring, in chordwise panels


def plate():
    """(rings (P,4,3), collocation (P,3), leading edge (NS+1,3), trailing edge (NS+1,3))."""
    dx, ds = CHORD / NC, SPAN / NS
    front = -0.25 * dx + torch.arange(NC, dtype=torch.float64) * dx
    back = front + dx
    y = torch.arange(NS + 1, dtype=torch.float64) * ds

    def columns(values):                               # (NC, NS, 1)
        return values.reshape(NC, 1, 1).expand(NC, NS, 1)

    left = y[:-1].reshape(1, NS, 1).expand(NC, NS, 1)
    right = y[1:].reshape(1, NS, 1).expand(NC, NS, 1)
    zero = torch.zeros(NC, NS, 1, dtype=torch.float64)
    rings = torch.cat(
        (
            torch.cat((columns(front), left, zero), dim=2),
            torch.cat((columns(front), right, zero), dim=2),
            torch.cat((columns(back), right, zero), dim=2),
            torch.cat((columns(back), left, zero), dim=2),
        ),
        dim=2,
    ).reshape(-1, 4, 3)
    collocation = torch.cat(
        ((columns(front) + columns(back)) / 2, (left + right) / 2, zero), dim=2
    ).reshape(-1, 3)
    base = torch.stack(
        (torch.zeros(NS + 1, dtype=torch.float64), y, torch.zeros(NS + 1)), dim=1
    )
    return rings, collocation, base, base + torch.tensor([CHORD, 0.0, 0.0], dtype=torch.float64)


def ring_free_end(leading: torch.Tensor) -> torch.Tensor:
    """(NS, 2, 3): a free end ahead of and above the leading edge, the same for both nodes."""
    return h.strip_pairs(leading) + torch.tensor(AHEAD, dtype=torch.float64) * CHORD / NC


def normal_influence(points, rings):
    """(points, rings): the normal component of the velocity of unit rings."""
    return kernel.native_ring_velocity_expanded(
        points, rings, core_fraction=1.0e-6, reference_length=kernel.mean_chordwise_panel(rings)
    )[..., 2]


def freestream(alpha_deg: float) -> torch.Tensor:
    alpha = math.radians(alpha_deg)
    return torch.tensor([SPEED * math.cos(alpha), 0.0, SPEED * math.sin(alpha)], dtype=torch.float64)


def trailing_wake_normal(collocation, rings, free):
    """(rows, P, NS): unit influence of every wake row, newest first, from the trailing edge."""
    rear = rings.reshape(NC, NS, 4, 3)[-1, :, 2:4]            # back-right, back-left
    step = free * (DT_STAR * CHORD) / SPEED
    rows = []
    for k in range(WAKE_ROWS):
        front, back = rear + k * step, rear + (k + 1) * step
        unit = torch.stack((front[:, 1], front[:, 0], back[:, 0], back[:, 1]), dim=1)
        rows.append(normal_influence(collocation, unit))
    return torch.stack(rows, dim=0)


def lesp_scale(rings) -> torch.Tensor:
    """The solver's `scale`: LESP = -circulation of the first row / scale."""
    first_dx = 0.5 * (
        torch.linalg.vector_norm(rings[:NS, 3] - rings[:NS, 0], dim=1)
        + torch.linalg.vector_norm(rings[:NS, 2] - rings[:NS, 1], dim=1)
    )
    theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / CHORD, -1.0, 1.0))
    return CHORD * SPEED * (theta + torch.sin(theta)) / LESP_FACTOR


def attached(alpha_deg: float) -> dict:
    """Everything a step needs before any strip sheds, for a plate in a steady wake."""
    free = freestream(alpha_deg)
    rings, collocation, leading, trailing = plate()
    aic = normal_influence(collocation, rings)
    wake = trailing_wake_normal(collocation, rings, free)     # (rows, P, NS)
    nascent_normal = wake[0]
    system = aic.clone()
    system[:, -NS:] += nascent_normal
    rhs = -free[2] * torch.ones(NC * NS, dtype=torch.float64)
    gamma = torch.linalg.solve(system, rhs)
    # the older rows of a steady state carry the trailing-edge circulation of their strip
    rhs_known = rhs - torch.einsum("kpj,j->p", wake[1:], gamma.reshape(NC, NS)[-1])
    scale = lesp_scale(rings)
    return {
        "free": free, "rings": rings, "collocation": collocation, "leading": leading,
        "trailing": trailing, "system": system, "nascent_normal": nascent_normal,
        "rhs_known": rhs_known, "gamma": gamma, "scale": scale, "lesp": -gamma[:NS] / scale,
    }


def structures(state: dict):
    """Nascent rings, pseudovortex rings and their summed unit influence (P, NS)."""
    nascent = h.nascent_row(ring_free_end(state["leading"]), h.strip_pairs(state["leading"]))
    pseudovortex = h.pseudovortex_ring(state["leading"], state["trailing"])
    influence = normal_influence(state["collocation"], nascent) + normal_influence(
        state["collocation"], pseudovortex
    )
    return nascent, pseudovortex, influence


# -- chain bookkeeping ------------------------------------------------------------------------------


def test_nascent_row_lies_on_the_leading_edge_against_the_bound_front_leg():
    rings, collocation, leading, trailing = plate()
    free = ring_free_end(leading)
    nascent = h.nascent_row(free, h.strip_pairs(leading))
    # aft corners are the leading-edge nodes, node j+1 then node j
    assert torch.allclose(nascent[:, 2], leading[1:]) and torch.allclose(nascent[:, 3], leading[:-1])
    assert torch.allclose(nascent[:, 0], free[:, 0]) and torch.allclose(nascent[:, 1], free[:, 1])
    # the bound front leg of the first row runs node j -> j+1 on the quarter-chord line; the
    # nascent aft leg runs j+1 -> j on the leading edge: opposite directions
    first_row = rings.reshape(NC, NS, 4, 3)[0]
    assert torch.all(torch.sum((first_row[:, 1] - first_row[:, 0]) * (nascent[:, 3] - nascent[:, 2]), dim=1) < 0.0)
    # the pseudovortex runs the other way on the same line, with its back edge on the trailing edge
    pseudovortex = h.pseudovortex_ring(leading, trailing)
    assert torch.allclose(pseudovortex[:, 0], leading[:-1]) and torch.allclose(pseudovortex[:, 3], trailing[:-1])
    assert torch.all(torch.sum((pseudovortex[:, 1] - pseudovortex[:, 0]) * (nascent[:, 3] - nascent[:, 2]), dim=1) < 0.0)
    del collocation


def test_split_point_is_one_third_of_the_way():
    edge = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    aft = torch.tensor([[[0.3, 0.0, 0.9], [0.3, 1.0, 0.9]]])
    assert torch.allclose(h.split_points(edge, aft), edge + (aft - edge) / 3.0)


def shed(chain, active, nascent, strength):
    """One shedding step as the solver does it: push, then record the strength as `prev`."""

    pushed, released_rings, released_gamma, released = h.push_nascent(
        chain, active, nascent, torch.full((NS,), strength)
    )
    pushed = h.LevChain(
        rings=pushed.rings, gamma=pushed.gamma, count=pushed.count,
        prev=torch.where(active, torch.full_like(active, strength, dtype=torch.float64), 0.0),
        joined=pushed.joined,
    )
    return pushed, released_rings, released_gamma, released


def test_push_keeps_every_strength_and_splits_the_old_nascent_row():
    _, _, leading, _ = plate()
    chain = h.LevChain.empty(NS, torch.device("cpu"))
    active = torch.zeros(NS, dtype=torch.bool)
    active[7] = True
    nascent = h.nascent_row(ring_free_end(leading), h.strip_pairs(leading))
    strengths = [0.3, 0.5, 0.2, 0.7]
    history = []
    for step, strength in enumerate(strengths):
        chain, released_rings, released_gamma, released = shed(
            chain, active, nascent * (1.0 + 0.1 * step), strength
        )
        history.append((bool(released[7]), float(released_gamma[7]), int(chain.count[7])))
    # three rows fit; the fourth shedding hands the oldest row over
    assert [item[2] for item in history] == [1, 2, 3, 3]
    assert [item[0] for item in history] == [False, False, False, True]
    assert history[3][1] == pytest.approx(strengths[0])
    assert [float(chain.gamma[m, 7]) for m in range(h.LEV_MAX_ROWS)] == pytest.approx(strengths[3:0:-1])
    # the other strips were never touched, and their rows carry nothing
    assert int(chain.count[:7].sum()) == 0 and int(chain.count[8:].sum()) == 0
    assert float(chain.rings[:, 8:].abs().sum()) == 0.0
    assert float(chain.rings[2, :7].abs().sum()) + float(chain.rings[2, 8:].abs().sum()) == 0.0
    # the old nascent row is now row 1 and its aft end is the new row's free end
    assert torch.allclose(chain.rings[1, 7, 2], chain.rings[0, 7, 1])
    assert torch.allclose(chain.rings[1, 7, 3], chain.rings[0, 7, 0])
    del released_rings


def test_reconnect_makes_the_chain_share_edges_exactly():
    _, _, leading, _ = plate()
    chain = h.LevChain.empty(NS, torch.device("cpu"))
    nascent = h.nascent_row(ring_free_end(leading), h.strip_pairs(leading))
    for _ in range(h.LEV_MAX_ROWS):
        chain, *_ = shed(chain, torch.ones(NS, dtype=torch.bool), nascent, 0.4)
    assert bool(torch.all(chain.joined))
    drifted = chain.rings.clone()
    drifted[1:, :, 2] += 1.0e-6                                  # a convection that did not re-pin
    pinned = h.reconnect(chain.with_rings(drifted))
    assert torch.equal(pinned.rings[1:, :, 2], pinned.rings[:-1, :, 1])
    assert torch.equal(pinned.rings[1:, :, 3], pinned.rings[:-1, :, 0])


def test_a_chain_that_stopped_shedding_is_not_joined_when_the_strip_sheds_again():
    """The paper's split presumes a sheet that keeps shedding; after a pause the new ring is a
    first ring on the edge and the old rows stay where they are."""
    _, _, leading, _ = plate()
    chain = h.LevChain.empty(NS, torch.device("cpu"))
    active = torch.ones(NS, dtype=torch.bool)
    edge = h.strip_pairs(leading)
    nascent = h.nascent_row(ring_free_end(leading), edge)
    chain, *_ = shed(chain, active, nascent, 0.4)
    # the strip did not shed at the previous step: `prev` is zero
    paused = h.LevChain(
        rings=chain.rings, gamma=chain.gamma, count=chain.count,
        prev=torch.zeros_like(chain.prev), joined=chain.joined,
    )
    assert not bool(torch.any(h.continuing(paused)))
    moved = chain.rings + 0.01                                   # the old ring convected away
    paused = paused.with_rings(moved)
    fresh = h.nascent_row(edge + 0.01, edge)                     # a first ring on the edge
    resumed, *_ = shed(paused, active, fresh, 0.6)
    assert not bool(torch.any(resumed.joined[0]))
    assert torch.equal(resumed.rings[1], moved[0])              # the old ring is untouched
    assert float(resumed.gamma[0, 3]) == pytest.approx(0.6)
    # while shedding continues the chain is joined and the old ring is split
    third, *_ = shed(resumed, active, fresh, 0.5)
    assert bool(torch.all(third.joined[0]))
    assert torch.allclose(third.rings[1, :, 2], third.rings[0, :, 1])


def test_release_edges_close_every_ring_and_merge_a_shared_node_line():
    unit = torch.eye(3, dtype=torch.float64)
    origin = torch.zeros(3, dtype=torch.float64)
    ring = torch.stack((origin, origin + unit[1], origin + unit[0] + unit[1], origin + unit[0]))
    # strips 0 and 1 share a node line; strip 2 is two panels further, with a gap in between
    rings = torch.stack((ring, ring + unit[1], ring + 3.0 * unit[1]))
    circulation = torch.tensor([0.4, 0.4, 0.6])

    starts, ends, strength, strip = h.release_edges(
        rings, circulation, torch.tensor([True, True, True])
    )
    # strip 0 loses its right leg to the shared line, strip 1 keeps its left leg with the
    # difference, strip 2 is alone: 3 + 4 + 4
    assert starts.shape[0] == 11
    assert float(torch.linalg.vector_norm(torch.sum(strength[:, None] * (ends - starts), dim=0))) < 1.0e-14
    assert int((strength == 0.0).sum()) == 1 and int(strip[strength == 0.0]) == 1

    starts, ends, strength, strip = h.release_edges(
        rings, circulation, torch.tensor([False, True, False])
    )
    assert starts.shape[0] == 4
    assert torch.allclose(strength, torch.full((4,), 0.4))
    assert float(torch.linalg.vector_norm(torch.sum(strength[:, None] * (ends - starts), dim=0))) < 1.0e-14
    del strip


def test_release_particles_subdivide_like_the_ribbon_deposit():
    unit = torch.eye(3, dtype=torch.float64)
    origin = torch.zeros(3, dtype=torch.float64)
    ring = torch.stack((origin, origin + unit[1], origin + unit[0] + unit[1], origin + unit[0]))
    positions, gamma, sigma, circulation, strip = h.release_particles(
        ring[None], torch.tensor([0.4], dtype=torch.float64), torch.tensor([True]),
        smoothing_radius=0.3, target_spacing=0.25,
    )
    # four unit legs, four segments each
    assert positions.shape == (16, 3) and torch.allclose(sigma, torch.full((16,), 0.3, dtype=torch.float64))
    # the tag is the edge circulation repeated per particle, as in the ribbon deposit; the vector
    # strength is circulation x segment, so the particles of a leg add up to circulation x edge
    assert torch.allclose(circulation, torch.full((16,), 0.4, dtype=torch.float64))
    assert float(gamma.abs().sum()) == pytest.approx(0.4 * 4.0)
    assert float(torch.linalg.vector_norm(torch.sum(gamma, dim=0))) < 1.0e-14
    assert bool(torch.all(strip == 0))


# -- the conventions, with the ring kernel ----------------------------------------------------------


def test_nascent_ring_plus_pseudovortex_is_one_closed_loop():
    """The pseudovortex's front leg cancels the nascent ring's aft leg on the leading edge.

    With that leg gone the two rings form one closed loop: free end -> leading edge -> trailing
    edge -> leading edge -> free end.  A quadrilateral cannot hold it in one ring, so the loop is
    written here as the direct quadrilateral (free end, trailing edge) plus the two triangles that
    bend it through the leading-edge nodes.  If the leading-edge legs did not cancel, an extra
    pair of anti-parallel legs would remain and the two inductions would differ.
    """
    rings, collocation, leading, trailing = plate()
    free = ring_free_end(leading)
    edge, back = h.strip_pairs(leading), h.strip_pairs(trailing)
    nascent = h.nascent_row(free, edge)
    pseudovortex = h.pseudovortex_ring(leading, trailing)
    direct = torch.stack((free[:, 0], free[:, 1], back[:, 1], back[:, 0]), dim=1)
    triangle_right = torch.stack((free[:, 1], edge[:, 1], back[:, 1], back[:, 1]), dim=1)
    triangle_left = torch.stack((back[:, 0], edge[:, 0], free[:, 0], free[:, 0]), dim=1)
    probe = collocation[::7] + torch.tensor([0.0, 0.0, 0.3]) * CHORD / NC
    reference = kernel.mean_chordwise_panel(rings)

    def velocity(rings_set, strengths):
        return (
            kernel.native_ring_velocity_expanded(
                probe, rings_set, core_fraction=1.0e-6, reference_length=reference
            )
            * strengths[None, :, None]
        ).sum(dim=1)

    strength = torch.linspace(0.02, 0.05, NS, dtype=torch.float64)
    together = velocity(nascent, strength) + velocity(pseudovortex, strength)
    loop = velocity(direct, strength) + velocity(triangle_right, strength) + velocity(triangle_left, strength)
    assert float((together - loop).abs().max() / loop.abs().max()) < 1.0e-10


def test_attached_sign_and_lesp():
    state = attached(16.0)
    # positive incidence: the leading-edge row carries one sign, and the LESP of the solver
    # (minus circulation over scale) is negative, as in production; at 16 deg the mid-span
    # strips are past the critical value and the strips next to the tips are not
    assert bool(torch.all(state["gamma"][:NS] > 0.0))
    assert bool(torch.all(state["lesp"] < 0.0))
    assert float(state["lesp"][NS // 2]) < -LESP_CRIT
    assert float(state["lesp"][0]) > -LESP_CRIT


def test_lev_loop_lowers_the_attached_lesp_but_the_bare_nascent_ring_raises_it():
    """Why the pseudovortex ring is there (the paper's Fig. 5)."""
    state = attached(16.0)
    nascent, _, both = structures(state)
    strength = 0.5 * state["gamma"][:NS]
    mid = slice(NS // 2 - 3, NS // 2 + 3)
    attached_row = state["gamma"][:NS][mid].abs()
    with_loop = torch.linalg.solve(state["system"], state["rhs_known"] - both @ strength)
    with_bare_ring = torch.linalg.solve(
        state["system"], state["rhs_known"] - normal_influence(state["collocation"], nascent) @ strength
    )
    assert bool(torch.all(with_loop[:NS][mid].abs() < attached_row)), "the loop must lower the suction"
    assert bool(torch.all(with_bare_ring[:NS][mid].abs() > attached_row)), "the bare ring raises it"


def test_coupled_solve_pins_closes_and_sheds_with_the_bound_sign():
    state = attached(16.0)
    _, _, both = structures(state)
    active = state["lesp"].abs() > LESP_CRIT
    index = torch.nonzero(active, as_tuple=False).reshape(-1)
    target = -torch.sign(state["lesp"]) * LESP_CRIT * state["scale"]
    gamma, lam, residual = h.solve_coupled(
        state["system"], state["nascent_normal"], both[:, index], state["rhs_known"], active, target
    )
    lesp = -gamma[:NS] / state["scale"]
    assert float(residual.abs().max()) < 1.0e-9 * SPEED
    # every shedding strip comes back to the critical value with the sign it had
    assert torch.allclose(lesp[active], torch.sign(state["lesp"][active]) * LESP_CRIT, atol=1.0e-9)
    # the shed vortex turns the same way as the bound leading-edge row, and the pseudovortex
    # ring adds it to the bound system over the chord: the trailing-edge row of a shedding
    # strip carries more circulation than the attached solution.  That is the lift of the
    # leading-edge vortex, which the old deposit never had.
    assert bool(torch.all(torch.sign(lam) == torch.sign(state["gamma"][:NS][index])))
    attached_rows = state["gamma"].reshape(NC, NS)
    rows = gamma.reshape(NC, NS)
    assert bool(torch.all(rows[-1][active] > attached_rows[-1][active]))
    # A strip that did not shed moves through the induction of its neighbours' leading-edge
    # vortices, here by up to 5 %; the strip next to the active ones goes from 0.107 to 0.113
    # and crosses the critical value.  That is the second decision of the paper's flowchart:
    # the active set is iterated until no strip crosses, which is the solver's job.
    assert bool(torch.all(torch.isfinite(gamma)))


def test_coupled_solve_without_active_strips_is_the_attached_solve():
    state = attached(5.0)
    gamma, lam, residual = h.solve_coupled(
        state["system"], state["nascent_normal"], torch.zeros(NC * NS, 0), state["rhs_known"],
        torch.zeros(NS, dtype=torch.bool), torch.zeros(NS),
    )
    reference = torch.linalg.solve(state["system"], state["rhs_known"])
    assert lam.shape == (0,) and torch.allclose(gamma, reference, rtol=1.0e-12, atol=0.0)
    assert float(residual.abs().max()) < 1.0e-9 * SPEED


def test_suction_force_follows_the_paper():
    chord = torch.full((NS,), CHORD)
    width = torch.full((NS,), SPAN / NS)
    forward = torch.tensor([-1.0, 0.0, 0.0]).expand(NS, 3)
    force = h.suction_force(1.208, SPEED, chord, width, forward, torch.full((NS,), 0.2))
    expected = math.pi * 1.208 * SPEED**2 * CHORD * 0.04 * SPAN / NS
    assert torch.allclose(force[:, 0], -torch.full((NS,), expected))
    assert float(force[:, 1:].abs().sum()) == 0.0
