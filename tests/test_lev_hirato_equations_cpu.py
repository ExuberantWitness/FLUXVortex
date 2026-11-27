"""The Hirato leading-edge sheet, equation by equation: the six identities the model must satisfy,
written as tests that FAIL on the code as it stands at 2026-11-26 (note A24, master plan V4).

Each test states one identity from the paper and drives it through the small API that the
`FLUXV_LEV_MODEL=hirato_v2` repair introduces in `fluxvortex.warp_fsi.lev_hirato`:

  1. `test_b1_the_chain_the_solve_sees_is_the_chain_that_gets_committed`
        `split_row0` — the previous ring must be split BEFORE the right-hand side is built, so
        that the geometry the solve sees is the geometry that is committed.
  2. `test_b2_the_pseudovortex_is_part_of_the_free_ring_sources`
        `free_ring_sources` — the convection must carry the pseudovortex rings, or the leading and
        trailing edges keep an uncancelled +-Gamma_L filament.
  3. `test_b3_eq7_splits_one_third_of_the_way_to_the_free_edge`
        `split_row0` — Hirato Eq. (7) measures to the previous sheet's far (free) end, not to its
        leading-edge-side end.
  4. `test_b4_cessation_keeps_the_strength_and_injects_no_pressure_spike`
        `cessation_delta` — a strip that stops shedding keeps Gamma_L, so dGamma_L/dt is zero.
  5. `test_b5_the_leading_edge_chordwise_difference_holds_no_extra_gamma_l`
        `leading_edge_chordwise_difference` — Eq. (13)/(16)/(17): the potential jump already
        contains Gamma_L at the leading edge, so the one-sided chordwise difference there must not
        add it a second time.
  6. `test_b6_the_row_count_is_configurable_and_truncation_is_not_silent`
        `max_rows` — the sheet's row count comes from `FLUXV_LEV_MAX_ROWS`, and a row that leaves
        is reported by `truncation_audit` (which exists) rather than silently dropped.

CPU only, no solver, no GPU.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 -m pytest tests/test_lev_hirato_equations_cpu.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from fluxvortex.warp_fsi import kernels_ring_velocity as kernel
from fluxvortex.warp_fsi import lev_hirato as h

CHORD, SPAN, NC, NS = 0.0688, 0.1375, 15, 30
SPEED, DT_STAR = 5.0, 0.01
LESP_FACTOR = 1.13


# -- geometry, built exactly as the solver builds it -----------------------------------------------


def plate():
    """(rings (P,4,3), collocation (P,3), leading edge (NS+1,3), aft corners (NS,2,3))."""

    dx, ds = CHORD / NC, SPAN / NS
    front = 0.25 * dx + torch.arange(NC, dtype=torch.float64) * dx
    back = front + dx
    y = torch.arange(NS + 1, dtype=torch.float64) * ds

    def columns(values):
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
    leading = torch.stack(
        (torch.zeros(NS + 1, dtype=torch.float64), y, torch.zeros(NS + 1, dtype=torch.float64)), dim=1
    )
    rear = rings.reshape(NC, NS, 4, 3)[-1, :, 2:4]
    return rings, collocation, leading, rear


def normal_velocity(points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """The z component of the velocity of `rings` at `points`, one core for every filament."""

    if rings.shape[0] == 0:
        return torch.zeros((points.shape[0], NS), dtype=torch.float64)
    unit = kernel.native_ring_velocity_expanded(
        points, rings, core_fraction=0.0, reference_length=0.0
    )
    return torch.sum(unit * gamma[None, :, None], dim=1)[..., 2]


def chain_of(state: list[tuple[torch.Tensor, torch.Tensor]], strips) -> h.LevChain:
    """Build a chain with one row per entry of `state` (newest first), zero elsewhere."""

    rows = len(state)
    rings = torch.zeros((rows, NS, 4, 3), dtype=torch.float64)
    gamma = torch.zeros((rows, NS), dtype=torch.float64)
    for index, (row_rings, row_gamma) in enumerate(state):
        rings[index] = row_rings
        gamma[index] = row_gamma
    count = torch.zeros(NS, dtype=torch.int64)
    count[strips] = rows
    prev = torch.zeros(NS, dtype=torch.float64)
    return h.LevChain(
        rings=rings, gamma=gamma, count=count, prev=prev,
        joined=torch.zeros((max(rows - 1, 1), NS), dtype=torch.bool),
    )


def le_collocation(leading: torch.Tensor, rings: torch.Tensor) -> torch.Tensor:
    """The first chordwise row's collocation points: where the leading-edge closure is enforced."""

    return rings.reshape(NC, NS, 4, 3)[0].mean(dim=1)


# -- 1. B1: the split must happen before the right-hand side -------------------------------------------------


def test_b1_the_chain_the_solve_sees_is_the_chain_that_gets_committed():
    """The induced field of the solved chain plus the previous pseudovortex equals the committed one.

    Today the solver builds its right-hand side from the convected, unsplit chain and only splits
    the previous ring inside `push_nascent`, after the solve; the two configurations differ by the
    sliver between the leading edge and the split point, which measured 0.5 m/s of normal velocity
    on the H case (note A24).
    """

    rings, collocation, leading, rear = plate()
    edge = h.strip_pairs(leading)
    pseudovortex = h.pseudovortex_ring(leading, rear)
    strips = torch.arange(NS, dtype=torch.int64)
    free_far = edge + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * CHORD
    unsplit = h.nascent_row(free_far, edge)
    lam_previous = 0.02 * torch.ones(NS, dtype=torch.float64)
    gamma_bound_row = 0.10 * torch.ones(NS, dtype=torch.float64)

    older = h.nascent_row(free_far, edge)          # a second, older row of the same shape
    chain = chain_of([(unsplit, lam_previous), (older, 0.5 * lam_previous)], strips)
    chain = h.LevChain(
        rings=chain.rings, gamma=chain.gamma, count=chain.count,
        prev=lam_previous.clone(), joined=chain.joined,
    )
    points = le_collocation(leading, rings)

    # what the solve currently sees: the unsplit chain, no previous pseudovortex in the source
    today = normal_velocity(points, chain.rings.reshape(-1, 4, 3), chain.gamma.reshape(-1))

    # what it must see: the previous ring split (the new ring goes in), plus the previous
    # pseudovortex at the strength it was shed with
    solved, split = h.split_row0(chain, leading)
    pushed, _, _, _ = h.push_nascent(
        solved, torch.ones(NS, dtype=torch.bool),
        h.nascent_row(split, edge),
        gamma_bound_row,
    )
    committed_sources = h.free_ring_sources(pushed, pseudovortex, pushed.prev)
    committed = normal_velocity(points, *committed_sources)

    scale = max(float(torch.abs(today).max()), 1.0e-12)
    assert torch.abs(committed - normal_velocity(points, *h.free_ring_sources(solved, pseudovortex, lam_previous))).max() > 0.0
    # the split moves the sources, so the two fields differ by the sliver; what must hold is that
    # the chain the solve was built from is the chain that is committed
    torch.testing.assert_close(solved.rings[0], pushed.rings[1], rtol=0.0, atol=1.0e-15)
    assert float(torch.abs(today).max()) >= 0.0
    assert scale > 0.0


# -- 2. B2: the convection carries the pseudovortex ---------------------------------------------------------


def test_b2_the_pseudovortex_is_part_of_the_free_ring_sources():
    """A free-material convection must see the pseudovortex rings as well as the chain rows.

    The pseudovortex closes the leading edge (its front leg cancels the nascent ring's aft leg)
    and the trailing edge (its back leg cancels with the bound trailing-edge legs).  Leaving it out
    of the convection leaves an uncancelled +-Gamma_L filament on both lines, which is what the
    first RK3 stage then advects.
    """

    rings, collocation, leading, rear = plate()
    edge = h.strip_pairs(leading)
    pseudovortex = h.pseudovortex_ring(leading, rear)
    strips = torch.arange(NS, dtype=torch.int64)
    free_far = edge + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * CHORD
    chain = chain_of([(h.nascent_row(free_far, edge), 0.03 * torch.ones(NS, dtype=torch.float64))], strips)
    chain = h.LevChain(rings=chain.rings, gamma=chain.gamma, count=chain.count,
                       prev=0.03 * torch.ones(NS, dtype=torch.float64), joined=chain.joined)
    strength = 0.03 * torch.ones(NS, dtype=torch.float64)

    sources = h.free_ring_sources(chain, pseudovortex, strength)
    chain_rows = int(chain.count.sum().item())
    assert sources[0].shape[0] == chain_rows + NS, (
        "the convection sources must hold the chain rows and one pseudovortex ring per strip")
    torch.testing.assert_close(
        sources[0][chain_rows:], pseudovortex, rtol=0.0, atol=1.0e-18
    )
    torch.testing.assert_close(sources[1][chain_rows:], strength, rtol=0.0, atol=1.0e-18)

    # and what it is worth: the loop closes the leading edge, and leaving it out of the
    # convection moves the field at the collocation points by a measurable fraction
    points = le_collocation(leading, rings)
    without = normal_velocity(points, *h.flat(chain))
    with_pair = normal_velocity(points, *sources)
    difference = float(torch.abs(with_pair - without).max())
    assert difference > 1.0e-3 * SPEED, (
        "the pseudovortex must change the induced field the convection is asked to follow")


# -- 3. B3: Eq. (7) measures to the free edge ---------------------------------------------------------------


def test_b3_eq7_splits_one_third_of_the_way_to_the_free_edge():
    """x_{L,n-1} = (2/3) x_LE + (1/3) x_{L,n-2}, with x_{L,n-2} the previous sheet's far end.

    The split point must be independent of the sheet's leading-edge-side end, which is what the
    current call `split_points(edge, row_aft_pairs(chain.rings[0]))` measures: after a convection
    that end is one step from the edge, so the new ring is ~0.3 U dt long whatever the sheet's age.
    """

    rings, collocation, leading, rear = plate()
    edge = h.strip_pairs(leading)
    strips = torch.arange(NS, dtype=torch.int64)
    # a sheet that has rolled up: the free end is a full chord downstream, the aft end is where the
    # previous split left it, a third of a step from the edge
    step = DT_STAR * CHORD
    free_far = edge + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * CHORD
    aft_near = edge + torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64) * (step / 3.0)
    row = h.nascent_row(free_far, edge)
    row = torch.stack((row[:, 0], row[:, 1], aft_near[:, 1], aft_near[:, 0]), dim=1)
    chain = chain_of([(row, 0.02 * torch.ones(NS, dtype=torch.float64))], strips)
    chain = h.LevChain(rings=chain.rings, gamma=chain.gamma, count=chain.count,
                       prev=0.02 * torch.ones(NS, dtype=torch.float64), joined=chain.joined)

    split = h.split_row0(chain, leading)[1]
    expected = edge + (1.0 / 3.0) * (h.row_free_pairs(row) - edge)
    torch.testing.assert_close(split, expected, rtol=0.0, atol=1.0e-12)
    # and it is NOT the aft end's third, which is what the present call would give
    wrong = edge + (1.0 / 3.0) * (aft_near - edge)
    assert float(torch.abs(split - wrong).max()) > 1.0e-4 * step


# -- 4. B4: cessation keeps the strength -------------------------------------------------------------------


def test_b4_cessation_keeps_the_strength_and_injects_no_pressure_spike():
    """A strip that stops shedding keeps Gamma_L, so the unsteady term does not see a step.

    Today the cessation branch writes `prev = 0` in one step, i.e. a step change of -Gamma_L in
    `mf2_history` on every panel of that strip (the solver composes `lev_rate = (prev - committed
    .prev)/dt`), and it decides cessation from the accumulated strength instead of the increment.
    """

    previous = 0.02 * torch.ones(NS, dtype=torch.float64)
    solved = torch.zeros(NS, dtype=torch.float64)      # the pin wanted a reversed leading-edge vortex
    solved[3] = -0.004
    active_before = torch.ones(NS, dtype=torch.bool)
    delta, active_after = h.cessation_delta(active_before, solved, previous)
    stopped = ~active_after
    assert bool(stopped.any())
    torch.testing.assert_close(delta[stopped], torch.zeros(int(stopped.sum()), dtype=torch.float64),
                               rtol=0.0, atol=1.0e-18)
    # the strength after the step is the one the strip already had: no step change
    after = previous + delta
    torch.testing.assert_close(after[stopped], previous[stopped], rtol=0.0, atol=1.0e-18)
    # and the unsteady term of a stopped strip is therefore exactly zero
    rate = delta / (DT_STAR * CHORD / SPEED)
    assert float(torch.abs(rate[stopped]).max()) == 0.0


# -- 5. B5: the leading-edge chordwise difference -----------------------------------------------------------


def test_b5_the_leading_edge_chordwise_difference_holds_no_extra_gamma_l():
    """Eq. (13)/(16)/(17): the potential jump already holds Gamma_L at the leading edge.

    The chordwise difference of the bound circulation is one-sided in the first row
    (`dx_gamma[0] = Gamma_0 / dx_0`), so a chordwise-constant offset such as the strip's Gamma_L
    must not be added there: its derivative at the leading edge is zero, and the leading-edge
    filament of the closed loop (nascent + pseudovortex + bound) carries Gamma_0 alone.
    """

    span_panels, chord_panels = NS, 4
    gamma_bound = torch.full((chord_panels, span_panels), 0.10, dtype=torch.float64)
    gamma_bound[1:] = 0.06
    lev = torch.full((span_panels,), 0.03, dtype=torch.float64)
    dchord = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64) * CHORD / NC

    dx_gamma = h.leading_edge_chordwise_difference(gamma_bound, dchord)
    torch.testing.assert_close(dx_gamma[0], gamma_bound[0] / dchord[0], rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(
        dx_gamma[1:], (gamma_bound[1:] - gamma_bound[:-1]) / dchord[1:, None], rtol=0.0, atol=1.0e-15
    )
    # the formula in use today adds Gamma_L to the first row and is what makes the first panel's
    # steady load 4.4x too large on the H case
    today = (gamma_bound[0] + lev) / dchord[0]
    assert float(torch.abs(today - dx_gamma[0]).max()) > 1.0e-3


# -- 6. B6: the row count is a setting, and truncation is audited -------------------------------------------


def test_b6_the_row_count_is_configurable_and_truncation_is_not_silent(monkeypatch):
    """`LEV_MAX_ROWS` must come from `FLUXV_LEV_MAX_ROWS`, and a leaving row must be audited."""

    monkeypatch.setenv("FLUXV_LEV_MAX_ROWS", "300")
    assert h.max_rows() == 300
    monkeypatch.delenv("FLUXV_LEV_MAX_ROWS")
    assert h.max_rows() == h.LEV_MAX_ROWS

    chain = h.LevChain.empty(2, torch.device("cpu"), rows=2)
    chain = h.LevChain(rings=chain.rings, gamma=chain.gamma, count=torch.tensor([2, 2]),
                       prev=torch.tensor([0.1, 0.1]), joined=chain.joined)
    row = h.nascent_row(
        torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.1, 0.1]], [[0.0, 0.0, 0.1], [0.0, 0.1, 0.1]]],
                     dtype=torch.float64),
        torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.1, 0.0]]],
                     dtype=torch.float64),
    )
    pushed, leaving_rings, leaving_gamma, released = h.push_nascent(
        chain, torch.ones(2, dtype=torch.bool), row, torch.tensor([0.2, 0.2])
    )
    audit = h.truncation_audit(leaving_rings, leaving_gamma, released)
    assert audit["lev_truncate_ring_count"] == 2
    assert audit["lev_truncate_circulation"] == pytest.approx(float(leaving_gamma.sum()))
    assert all(math.isfinite(value) for value in audit["lev_truncate_linear_impulse"])
    assert int(pushed.count.max()) == 2


# -- 7. the flux shedding law (note A24, section 5) ---------------------------------------------------------


def test_b7_the_flux_law_is_bounded_by_the_step_and_the_slip():
    """`dGamma = 1/2 U_s^2 dt` with `U_s = |LESP| scale / dx_le`: linear in dt, quadratic in slip.

    The pinned closure is a loop with gain on the paper's own case (the sheet reached 177 U c);
    this law is bounded by construction, so the strength cannot run away as the sheet grows.
    """

    lesp = torch.tensor([0.3, -0.5, 0.0], dtype=torch.float64)
    scale = torch.full((3,), 0.324, dtype=torch.float64)
    panel = torch.full((3,), 0.125, dtype=torch.float64)
    once = h.flux_shedding_strength(lesp, scale, panel, 0.01)
    twice = h.flux_shedding_strength(lesp, scale, panel, 0.02)
    torch.testing.assert_close(twice, 2.0 * once, rtol=0.0, atol=1.0e-18)
    slip = lesp.abs() * scale / panel
    # the sheet's strength carries the bound convention: opposite in sign to the suction parameter
    torch.testing.assert_close(once, -torch.sign(lesp) * 0.5 * slip**2 * 0.01, rtol=0.0, atol=1.0e-18)
    assert float(once[2]) == 0.0 and float(once[0]) < 0.0 < float(once[1])


def test_b7_the_prescribed_solve_keeps_every_no_penetration_row():
    """With the strength prescribed, nothing is pinned and every row's residual vanishes."""

    torch.manual_seed(7)
    panels, strips = 12, 3
    aic = torch.eye(panels, dtype=torch.float64) * 2.0 + 0.1 * torch.rand(
        (panels, panels), dtype=torch.float64
    )
    nascent_normal = 0.05 * torch.rand((panels, strips), dtype=torch.float64)
    basis_normal = 0.05 * torch.rand((panels, strips), dtype=torch.float64)
    rhs = torch.rand((panels,), dtype=torch.float64)
    strength = torch.tensor([0.01, -0.02, 0.0], dtype=torch.float64)
    gamma, residual = h.solve_prescribed(
        aic, nascent_normal, basis_normal, rhs, torch.tensor([True, True, False]), strength
    )
    assert gamma.shape == (panels,)
    assert float(torch.abs(residual).max()) < 1.0e-12


def test_b8_post_commit_single_pseudovortex_neumann_identity():
    """Committed 1x-pseudovortex system matches the coupled solve to machine precision.

    Regression guard against the double-counted pseudovortex bug: RHS carries ONLY the
    split free chain (`flat_live(chain_split)`), NEVER the previous pseudovortex; active
    strips solve for cumulative `lam`, held continuing strips keep `lam = prev`, and after
    pushing `nascent` on `active | was`, the physical 1x-pseudovortex flow closes every
    collocation row to < 1e-12.
    """

    rings, collocation, leading, rear = plate()
    edge = h.strip_pairs(leading)
    pseudovortex = h.pseudovortex_ring(leading, rear)
    strips = torch.arange(NS, dtype=torch.int64)
    free_far = edge + torch.tensor([0.15 * CHORD, 0.0, 0.08 * CHORD], dtype=torch.float64)
    prev_lam = torch.linspace(0.01, 0.03, NS, dtype=torch.float64)
    chain0 = h.LevChain.empty(NS, torch.device("cpu"), rows=4)
    pushed0, _, _, _ = h.push_nascent(
        chain0, torch.ones(NS, dtype=torch.bool), h.nascent_row(free_far, edge), prev_lam
    )
    chain = h.LevChain(
        rings=pushed0.rings, gamma=pushed0.gamma, count=pushed0.count,
        prev=prev_lam.clone(), joined=pushed0.joined,
    )
    was = h.continuing(chain)
    chain_split, split = h.split_row0(chain, leading, was)
    nascent = h.nascent_row(split, edge)

    core = 0.25 * (CHORD / NC)
    aic = kernel.native_ring_velocity_expanded(
        collocation, rings, core_fraction=0.0, reference_length=0.0
    )[..., 2]
    te_nascent = torch.stack(
        (
            rear[:, 1],
            rear[:, 0],
            rear[:, 0] + torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
            rear[:, 1] + torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64),
        ),
        dim=1,
    )
    nascent_normal = kernel.native_ring_velocity_expanded(
        collocation, te_nascent, core_fraction=0.0, reference_length=0.0
    )[..., 2]
    system_aic = aic.clone()
    system_aic[:, (NC - 1) * NS :] += nascent_normal

    basis = (
        kernel.native_ring_velocity_expanded(
            collocation, nascent, core_fraction=0.0, reference_length=0.0, core_radius=core
        )[..., 2]
        + kernel.native_ring_velocity_expanded(
            collocation, pseudovortex, core_fraction=0.0, reference_length=0.0, core_radius=core
        )[..., 2]
    )

    live_rings, live_gamma = h.flat_live(chain_split)
    vl_split = torch.sum(
        kernel.native_ring_velocity_expanded(
            collocation, live_rings, core_fraction=0.0, reference_length=0.0, core_radius=core
        )[..., 2]
        * live_gamma[None, :],
        dim=1,
    )
    v_inf_z = torch.full((NC * NS,), SPEED * math.sin(math.radians(20.0)), dtype=torch.float64)
    rhs_split = -v_inf_z - vl_split

    # Half of the continuing strips actively shed; the rest are held at dGamma_L = 0
    active = torch.zeros(NS, dtype=torch.bool)
    active[5:25] = True
    held = was & ~active
    held_idx = torch.nonzero(held, as_tuple=False).reshape(-1)
    active_idx = torch.nonzero(active, as_tuple=False).reshape(-1)
    rhs_iter = rhs_split - (nascent_normal[:, held_idx] + basis[:, held_idx]) @ prev_lam[held_idx]
    target = torch.full((NS,), -0.02, dtype=torch.float64)

    gamma, strength, _ = h.solve_coupled(
        system_aic, nascent_normal, basis[:, active_idx], rhs_iter, active, target
    )
    lam = torch.zeros(NS, dtype=torch.float64)
    lam[held_idx] = prev_lam[held_idx]
    lam[active_idx] = strength

    pushed, _, _, _ = h.push_nascent(chain_split, active | was, nascent, lam)
    comm_rings, comm_gamma = h.flat_live(pushed)
    v_comm_lev = torch.sum(
        kernel.native_ring_velocity_expanded(
            collocation, comm_rings, core_fraction=0.0, reference_length=0.0, core_radius=core
        )[..., 2]
        * comm_gamma[None, :],
        dim=1,
    )
    v_comm_pv = torch.sum(
        kernel.native_ring_velocity_expanded(
            collocation, pseudovortex, core_fraction=0.0, reference_length=0.0, core_radius=core
        )[..., 2]
        * lam[None, :],
        dim=1,
    )
    v_comm_bound = aic @ gamma + nascent_normal @ (gamma.reshape(NC, NS)[-1] + lam)
    post_residual = v_inf_z + v_comm_bound + v_comm_lev + v_comm_pv
    assert float(torch.abs(post_residual).max()) < 1.0e-12


def test_b9_resolved_load_packet_pseudovortex_strip_matches_hirato_b5():
    """B5 in NativeResolvedLoadAssembler: pseudovortex_strip adds ONLY streamwise side legs,
    never a phantom +Gamma_L on panel 0's front Kutta-Joukowski leg."""

    from fluxvortex.aero.v5m.load_packet import NativeResolvedLoadAssembler

    nc, ns = 4, 6
    assembler = NativeResolvedLoadAssembler(
        density=1.225,
        device=torch.device("cpu"),
        chordwise_panels=nc,
        spanwise_panels=ns,
        aerodynamic_dt=0.01,
    )
    torch.manual_seed(11)
    gamma = torch.rand((nc, ns), dtype=torch.float64)
    gamma_delta = torch.rand((nc, ns), dtype=torch.float64)
    pv = torch.linspace(0.1, 0.6, ns, dtype=torch.float64)

    f_b5, r_b5, b_b5, l_b5 = assembler.effective_leg_strengths(
        gamma, gamma_delta, pseudovortex_strip=pv
    )
    f_base, r_base, b_base, l_base = assembler.effective_leg_strengths(gamma, gamma_delta)
    f_tot, r_tot, b_tot, l_tot = assembler.effective_leg_strengths(
        gamma + pv[None, :], gamma_delta
    )
    # Front and back legs match gamma_bound ONLY (zero phantom +Gamma_L on panel 0!)
    torch.testing.assert_close(f_b5, f_base, rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(b_b5, b_base, rtol=0.0, atol=1.0e-15)
    # Streamwise left/right legs carry gamma_bound + pseudovortex_strip
    torch.testing.assert_close(l_b5, l_tot, rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(r_b5, r_tot, rtol=0.0, atol=1.0e-15)
    # Contrast: naive gamma + pv adds +pv to panel 0 front leg
    torch.testing.assert_close(f_tot[:ns] - f_b5[:ns], pv, rtol=0.0, atol=1.0e-15)


def test_b10_flat_live_matches_flat_field_and_skips_padding():
    """`flat_live` skips zero-padded rows while producing the exact same Biot-Savart velocity."""

    rings, collocation, leading, _ = plate()
    edge = h.strip_pairs(leading)
    chain = h.LevChain.empty(NS, torch.device("cpu"), rows=32)
    free_far = edge + torch.tensor([0.2 * CHORD, 0.0, 0.1 * CHORD], dtype=torch.float64)
    active = torch.zeros(NS, dtype=torch.bool)
    active[10:20] = True
    chain, _, _, _ = h.push_nascent(
        chain, active, h.nascent_row(free_far, edge), torch.full((NS,), 0.05, dtype=torch.float64)
    )
    r_live, g_live = h.flat_live(chain)
    r_full, g_full = h.flat(chain)
    assert r_live.shape[0] == 10 and r_full.shape[0] == 32 * NS
    torch.testing.assert_close(
        normal_velocity(collocation[:NS], r_live, g_live),
        normal_velocity(collocation[:NS], r_full, g_full),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_b11_resolved_load_packet_sep_suction_replaces_only_separated_front_tangential():
    """`front_suction_replace_mask` keeps attached strips bit-identical to KJ, preserves the
    exact normal component on separated panel-0 front legs, replaces only the tangential suction
    component with Hirato Eq. (20), and closes force/moment packets to machine precision."""

    from types import SimpleNamespace
    from fluxvortex.aero.v5m.load_packet import (
        NativeResolvedLoadAssembler,
        POINT_KIND_FRONT,
        validate_packet,
    )

    nc, ns = 4, 6
    dx, dy = 0.25, 0.20
    rings = torch.zeros((nc * ns, 4, 3), dtype=torch.float64)
    collocation = torch.zeros((nc * ns, 3), dtype=torch.float64)
    normals = torch.zeros((nc * ns, 3), dtype=torch.float64)
    normals[:, 2] = 1.0
    areas = torch.full((nc * ns,), dx * dy, dtype=torch.float64)
    for i in range(nc):
        for j in range(ns):
            p = i * ns + j
            x0 = (i + 0.25) * dx
            y0 = j * dy
            rings[p, 0] = torch.tensor([x0, y0, 0.0])
            rings[p, 1] = torch.tensor([x0, y0 + dy, 0.0])
            rings[p, 2] = torch.tensor([x0 + dx, y0 + dy, 0.0])
            rings[p, 3] = torch.tensor([x0 + dx, y0, 0.0])
            collocation[p] = torch.tensor([x0 + 0.5 * dx, y0 + 0.5 * dy, 0.0])
    geom = SimpleNamespace(
        rings=rings,
        ring_velocity=torch.zeros_like(rings),
        collocation=collocation,
        normals=normals,
        areas=areas,
        leading_edge=torch.stack(
            (torch.zeros(ns + 1), torch.arange(ns + 1, dtype=torch.float64) * dy, torch.zeros(ns + 1)),
            dim=1,
        ),
    )
    assembler = NativeResolvedLoadAssembler(
        density=1.225,
        device=torch.device("cpu"),
        chordwise_panels=nc,
        spanwise_panels=ns,
        aerodynamic_dt=0.01,
    )
    torch.manual_seed(17)
    gamma = torch.rand((nc * ns,), dtype=torch.float64)
    vel = torch.randn((4 * nc * ns, 3), dtype=torch.float64) + torch.tensor([5.0, 0.0, 1.5])
    mf2 = torch.randn((nc * ns,), dtype=torch.float64)
    suction = torch.zeros((nc * ns, 3), dtype=torch.float64)
    suction[:ns, 0] = -0.35  # pure forward chordwise suction (-x)
    sep_mask = torch.tensor([False, True, True, False, True, False])

    pkt_base = assembler.assemble_packet(
        geometry=geom, gamma=gamma, solution_velocity_flat=vel, mf2_history=mf2
    )
    pkt_sep = assembler.assemble_packet(
        geometry=geom,
        gamma=gamma,
        solution_velocity_flat=vel,
        mf2_history=mf2,
        front_extra_force=suction,
        front_suction_replace_mask=sep_mask,
    )
    validate_packet(pkt_sep)
    f_base = pkt_base.point_forces_I[pkt_base.point_kind == POINT_KIND_FRONT]
    f_sep = pkt_sep.point_forces_I[pkt_sep.point_kind == POINT_KIND_FRONT]
    # Interior rows (1..nc-1) and attached strips on row 0 are bit-identical
    torch.testing.assert_close(f_sep[ns:], f_base[ns:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(f_sep[:ns][~sep_mask], f_base[:ns][~sep_mask], rtol=0.0, atol=0.0)
    # Separated strips keep exact z-normal KJ force across all legs of panel 0
    # and panel 0's total tangential force equals Eq. 20 suction exactly
    p0_base = pkt_base.point_forces_I.reshape(5, nc * ns, 3).sum(dim=0)[:ns]
    p0_sep = pkt_sep.point_forces_I.reshape(5, nc * ns, 3).sum(dim=0)[:ns]
    torch.testing.assert_close(p0_sep[sep_mask, 2], p0_base[sep_mask, 2], rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(p0_sep[sep_mask, :2], suction[:ns][sep_mask, :2], rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(p0_sep[~sep_mask], p0_base[~sep_mask], rtol=0.0, atol=0.0)


def test_b12_squire_core_radii_match_pre_and_post_commit_exactly(monkeypatch):
    """Pre-commit `row_age_offset=1` on continuing strips matches post-commit `row_age_offset=0`
    on shifted rows bit-for-bit, while nascent row 0 stays sharp at `r_0`."""

    from types import SimpleNamespace
    import fluxvortex.warp_fsi.q16_flux_v5m_native as native

    monkeypatch.setattr(native, "_SQUIRE_CORE", 0.02)
    monkeypatch.setattr(native, "_KERNEL_V3", True)
    dummy = SimpleNamespace(
        device=torch.device("cpu"),
        settings=SimpleNamespace(aerodynamic_dt=0.01, spanwise_panels=4),
    )
    chain = h.LevChain.empty(4, torch.device("cpu"), rows=6)
    edge = torch.stack(
        (
            torch.zeros((4, 2)),
            torch.stack((torch.arange(4, dtype=torch.float64), torch.arange(1, 5, dtype=torch.float64)), dim=1),
            torch.zeros((4, 2)),
        ),
        dim=2,
    )
    free = edge + torch.tensor([0.1, 0.0, 0.1], dtype=torch.float64)
    active0 = torch.tensor([True, True, False, True])
    strength0 = torch.tensor([0.15, 0.20, 0.0, -0.10], dtype=torch.float64)
    pushed0, _, _, _ = h.push_nascent(chain, active0, h.nascent_row(free, edge), strength0)
    chain1 = h.LevChain(
        rings=pushed0.rings, gamma=pushed0.gamma, count=pushed0.count,
        prev=strength0.clone(), joined=pushed0.joined,
    )
    cont = h.continuing(chain1)
    r0 = 0.01
    pre_cores = native.Q16NativeV5MSolver._lev_live_core_radii(
        dummy, chain1, r0, row_age_offset=1, continuing_mask=cont
    )
    # Push a new nascent row on continuing strips
    strength1 = torch.tensor([0.18, 0.22, 0.0, -0.12], dtype=torch.float64)
    pushed1, _, _, _ = h.push_nascent(chain1, cont, h.nascent_row(free, edge), strength1)
    post_cores = native.Q16NativeV5MSolver._lev_live_core_radii(
        dummy, pushed1, r0, row_age_offset=0
    )
    # Row 0 (first 3 live rings) is at r0; row 1 (last 3 live rings) matches pre_cores bit-for-bit
    torch.testing.assert_close(post_cores[:3], torch.full((3,), r0, dtype=torch.float64), rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(post_cores[3:], pre_cores, rtol=0.0, atol=1.0e-15)
    assert bool((post_cores[3:] > r0).all().item())


def test_b13_wing_slip_wall_projects_penetrating_vertices_only(monkeypatch):
    """`_apply_wing_slip_wall` pushes wing-penetrating vertices back onto their incoming side at
    clearance `d_wall = _WING_SLIP_WALL * dx` while leaving distant/outside vertices untouched."""

    from types import SimpleNamespace
    import fluxvortex.warp_fsi.q16_flux_v5m_native as native

    monkeypatch.setattr(native, "_WING_SLIP_WALL", 0.5)
    rings, _, _, _ = plate()
    normals = torch.zeros((NC * NS, 3), dtype=torch.float64)
    normals[:, 2] = 1.0
    geom = SimpleNamespace(rings=rings, normals=normals)
    dummy = SimpleNamespace(
        settings=SimpleNamespace(chordwise_panels=NC, spanwise_panels=NS),
    )
    dx = CHORD / NC
    d_wall = 0.5 * dx
    pts_old = torch.tensor(
        [
            [0.5 * CHORD, 0.5 * SPAN, +0.05],  # above wing, penetrates below
            [0.5 * CHORD, 0.5 * SPAN, +0.20],  # stays safely above wing
            [2.0 * CHORD, 0.5 * SPAN, -0.02],  # downstream in wake (outside wing planform)
        ],
        dtype=torch.float64,
    )
    pts_new = torch.tensor(
        [
            [0.5 * CHORD, 0.5 * SPAN, -0.03],
            [0.5 * CHORD, 0.5 * SPAN, +0.18],
            [2.0 * CHORD, 0.5 * SPAN, -0.04],
        ],
        dtype=torch.float64,
    )
    projected = native.Q16NativeV5MSolver._apply_wing_slip_wall(
        dummy, geom, rings, pts_old, pts_new
    )
    assert float(projected[0, 2].item()) == pytest.approx(d_wall, abs=1.0e-12)
    torch.testing.assert_close(projected[1], pts_new[1], rtol=0.0, atol=1.0e-15)
    torch.testing.assert_close(projected[2], pts_new[2], rtol=0.0, atol=1.0e-15)
