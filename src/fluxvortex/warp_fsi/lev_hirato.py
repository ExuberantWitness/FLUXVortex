"""Leading-edge vortex sheet as a chain of full-strength rings.

Hirato, G., et al., "Vortex-Sheet Representation of Leading-Edge Vortex Shedding from Finite
Wings", Journal of Aircraft, doi:10.2514/1.C035124, Sec. II.D (plan V3, stage 2).  The paper
states that its method applies to wings of aspect ratio four and higher; the wings here are below
that.

Per span strip the sheet is a chain of at most `LEV_MAX_ROWS` ring rows, every one an exact
ring.  A row is stored in the solver's corner order, (front-left, front-right, back-right,
back-left) = (free_j, free_{j+1}, aft_{j+1}, aft_j): the free end points away from the wing,
the aft end towards it.  Row 0 is the nascent ring while the strip sheds; its aft corners are then the
leading-edge nodes.  Along that edge the nascent ring runs j+1 -> j, against the bound front leg
of the first ring row (j -> j+1), and the pseudovortex ring across the chord has its front edge
on the same line with the same strength, so the leading-edge filament cancels and the first span
filament in front of the edge carries the difference between this step's and last step's
strength.

When a strip sheds again at the next step, the previous row 0 is split one third of the way
from the leading edge to its own aft end (the paper's Eq. 7): that aft end moves to the split
point and the new nascent ring takes the room next to the leading edge.  Every row keeps the
strength it had when it was shed.  A row that leaves the buffer is dropped, and its circulation
and impulse are handed back to the caller for its ledger.

Why the rows are not handed to the particle field.  A first version kept 3 rows as rings and
turned the rest into Gaussian particles with the smoothing radius of the old deposit.  On the
rigid plate at 16 deg that gave a thrust of its own kind of trouble: Cn 0.165 against 0.633 for
the old deposit, never settling (note A23, 6b).  In continuous shedding a row is 1-3 U dt long
and the smoothing radius was 4 U dt, so the front and back edges of a released ring cancelled
within it, and the streamwise legs of the sheet, once particles, acted on the next strip's
collocation points at 13 % of the filament law: the far sheet was smoothed away.  The sheet has
to stay filaments.

The paper's split presumes a sheet that keeps shedding.  A strip that stopped shedding has a
chain that convected away from the edge; when it sheds again its new ring is a first ring on the
edge and nothing is joined to the old rows (they stay in the chain until they leave it).  The
rule is "joined only if the strip shed at the previous step too"; it has no constant of its own.

The first ring of a strip is placed by Ramesh's rule, U A0 dt (sin a, cos a) / sqrt 2 in the
strip's chord/normal frame (platform/claim_runtime/hirato_equations.py, which records that the
embedding into a finite-wing strip is an adapter hypothesis).

Everything here is a pure function of its tensor arguments; nothing in this module knows the
solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Ring rows kept per strip.  In continuous shedding a row is about one U dt long, so this is
#: about one chord of sheet on the Rojratsirikul wing (dt* = 0.01); the trailing-edge wake keeps
#: three.  A row that leaves is dropped and audited, as the trailing-edge wake's are.
LEV_MAX_ROWS = 96
#: the paper's Eq. (7): the split point is this fraction of the way from the leading edge
SPLIT_FRACTION = 1.0 / 3.0


def strip_pairs(nodes: torch.Tensor) -> torch.Tensor:
    """(N+1, 3) node line -> (N, 2, 3): the two nodes of every strip, left then right."""

    return torch.stack((nodes[:-1], nodes[1:]), dim=1)


def row_free_pairs(row: torch.Tensor) -> torch.Tensor:
    """(N, 4, 3) rings -> (N, 2, 3): the free-end corners, node j then node j+1."""

    return torch.stack((row[:, 0], row[:, 1]), dim=1)


def row_aft_pairs(row: torch.Tensor) -> torch.Tensor:
    """(N, 4, 3) rings -> (N, 2, 3): the aft corners, node j then node j+1."""

    return torch.stack((row[:, 3], row[:, 2]), dim=1)


def nascent_row(free_pairs: torch.Tensor, leading_edge_pairs: torch.Tensor) -> torch.Tensor:
    """(N, 4, 3): the nascent ring of every strip, aft corners on the leading edge."""

    return torch.stack(
        (
            free_pairs[:, 0],
            free_pairs[:, 1],
            leading_edge_pairs[:, 1],
            leading_edge_pairs[:, 0],
        ),
        dim=1,
    )


def pseudovortex_ring(leading_edge: torch.Tensor, rear: torch.Tensor) -> torch.Tensor:
    """(N, 4, 3): the ring across the chord of every strip.

    Front edge on the geometric leading edge, where it cancels the aft leg of the nascent
    leading-edge ring.  Aft edge on the aft edge of the rearmost bound ring of the strip (the
    paper's Fig. 5): `rear` (N, 2, 3) holds that strip's back-right corner (node j+1) and
    back-left corner (node j), the line where the trailing-edge wake row attaches.  There its
    back leg, the bound back legs and the nascent wake row's front leg are on one line and
    cancel exactly.  Same corner order as a bound ring, so its circulation is in the bound
    convention.

    A first version put the aft edge on the geometric trailing edge, a quarter panel ahead of
    that line.  What stays behind is a dipole of strength lambda with the arm of a quarter panel
    right at the last collocation point, -lambda / (pi dx) there: on the rigid plate at 16 deg
    it drove the bound trailing-edge row negative and the step-to-step force between -0.16 and
    +1.07 (note A23, 6c).
    """

    return torch.stack(
        (leading_edge[:-1], leading_edge[1:], rear[:, 0], rear[:, 1]), dim=1
    )


def bound_rear(rings: torch.Tensor, chordwise_panels: int) -> torch.Tensor:
    """(N, 2, 3): back-right and back-left corners of the last chordwise row of bound rings."""

    return rings.reshape(chordwise_panels, -1, 4, 3)[-1, :, 2:4]


def first_free_points(
    leading_edge_pairs: torch.Tensor,
    chord_tangent: torch.Tensor,
    normal: torch.Tensor,
    speed: float,
    delta_time: float,
    lesp: torch.Tensor,
    sin_alpha: torch.Tensor,
    cos_alpha: torch.Tensor,
) -> torch.Tensor:
    """(N, 2, 3): free end of the first ring of a strip, by Ramesh's rule.

    `chord_tangent` and `normal` are unit vectors of the strip (LE -> TE, suction side), `lesp`
    is the strip's leading-edge suction parameter with its sign, and `sin_alpha`, `cos_alpha`
    those of the strip's geometric angle of attack.
    """

    scale = speed * lesp * delta_time / math.sqrt(2.0)
    displacement = scale[:, None] * (
        sin_alpha[:, None] * chord_tangent + cos_alpha[:, None] * normal
    )
    return leading_edge_pairs + displacement[:, None, :]


def split_points(leading_edge_pairs: torch.Tensor, aft_pairs: torch.Tensor) -> torch.Tensor:
    """(N, 2, 3): the paper's Eq. (7), one third of the way from the edge to the aft end."""

    return leading_edge_pairs + SPLIT_FRACTION * (aft_pairs - leading_edge_pairs)


@dataclass(frozen=True)
class LevChain:
    """The ring chains of every strip.

    `rings` (rows, N, 4, 3) and `gamma` (rows, N): row m of strip j is `rings[m, j]` with
    circulation `gamma[m, j]`.  Row 0 is the end nearest the leading edge.  Rows at or beyond
    `count[j]` are zero and carry no strength.  `prev` (N,) is the pseudovortex strength each
    strip carried at the previous step, zero where it was not shedding.  `joined` (rows-1, N):
    the aft end of row m+1 of a strip is the free end of row m, exactly.

    A strip joins its new ring to its chain only when it was shedding at the previous step
    too: that is when the paper's split applies, the last ring being the one it just shed.  A
    chain that stopped shedding has left the edge; when the strip sheds again the new ring is
    a first ring on the edge and the old rows stay where they are, unjoined.
    """

    rings: torch.Tensor
    gamma: torch.Tensor
    count: torch.Tensor
    prev: torch.Tensor
    joined: torch.Tensor

    @classmethod
    def empty(
        cls, span_panels: int, device: torch.device, rows: int = LEV_MAX_ROWS
    ) -> "LevChain":
        if rows < 1:
            raise ValueError("a chain needs at least one row")
        return cls(
            rings=torch.zeros((rows, span_panels, 4, 3), dtype=torch.float64, device=device),
            gamma=torch.zeros((rows, span_panels), dtype=torch.float64, device=device),
            count=torch.zeros((span_panels,), dtype=torch.int64, device=device),
            prev=torch.zeros((span_panels,), dtype=torch.float64, device=device),
            joined=torch.zeros((rows - 1, span_panels), dtype=torch.bool, device=device),
        )

    def clone(self) -> "LevChain":
        return LevChain(
            rings=self.rings.clone(), gamma=self.gamma.clone(), count=self.count.clone(),
            prev=self.prev.clone(), joined=self.joined.clone(),
        )

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.rings, self.gamma, self.count, self.prev, self.joined)

    def with_rings(self, rings: torch.Tensor) -> "LevChain":
        """The same chain with other corner positions (after a convection)."""

        return LevChain(
            rings=rings, gamma=self.gamma, count=self.count, prev=self.prev, joined=self.joined
        )


def _mask_invalid(chain: LevChain, count: torch.Tensor) -> LevChain:
    rows = torch.arange(chain.rings.shape[0], device=chain.rings.device)[:, None]
    valid = rows < count[None, :]                              # (rows, N)
    return LevChain(
        rings=torch.where(valid[:, :, None, None], chain.rings, torch.zeros_like(chain.rings)),
        gamma=torch.where(valid, chain.gamma, torch.zeros_like(chain.gamma)),
        count=count,
        prev=chain.prev,
        joined=chain.joined & valid[:-1] & valid[1:],
    )


def continuing(chain: LevChain) -> torch.Tensor:
    """(N,) bool: the strips that have a ring and were shedding at the previous step."""

    return (chain.count > 0) & (chain.prev != 0.0)


def push_nascent(
    chain: LevChain, active: torch.Tensor, nascent: torch.Tensor, strength: torch.Tensor
) -> tuple[LevChain, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shed: the new nascent rows go in, a continuing chain is split, a row may leave.

    `active` (N,) bool: the strips shedding this step; `nascent` (N, 4, 3) and `strength` (N,)
    are theirs (ignored where not active).  For a strip that continues its chain the aft
    corners of its previous row 0 become the free corners of the new one (the split point) and
    the two are joined; for a strip that starts again after a pause the old rows are left as
    they are.  Returns the new chain and, for the strips whose chain was already full, the row
    that left: rings (N, 4, 3), circulation (N,), mask (N,).  The caller audits it.
    """

    rows = chain.rings.shape[0]
    active = active.to(torch.bool)
    split_here = active & continuing(chain)
    shifted_rings = torch.zeros_like(chain.rings)
    shifted_gamma = torch.zeros_like(chain.gamma)
    shifted_rings[1:] = chain.rings[:-1]
    shifted_gamma[1:] = chain.gamma[:-1]
    shifted_joined = torch.zeros_like(chain.joined)
    shifted_joined[1:] = chain.joined[:-1]
    if rows > 1:
        # the old row 0, now row 1: for a continuing chain its aft end is the split point
        split_rings = shifted_rings[1].clone()
        split_rings[:, 2] = nascent[:, 1]
        split_rings[:, 3] = nascent[:, 0]
        shifted_rings[1] = torch.where(split_here[:, None, None], split_rings, shifted_rings[1])
        shifted_joined[0] = split_here
    shifted_rings[0] = nascent
    shifted_gamma[0] = strength
    rings = torch.where(active[None, :, None, None], shifted_rings, chain.rings)
    gamma = torch.where(active[None, :], shifted_gamma, chain.gamma)
    joined = torch.where(active[None, :], shifted_joined, chain.joined)
    count = torch.where(active, (chain.count + 1).clamp(max=rows), chain.count)
    released_mask = active & (chain.count >= rows)
    return (
        _mask_invalid(
            LevChain(rings=rings, gamma=gamma, count=count, prev=chain.prev, joined=joined),
            count,
        ),
        chain.rings[rows - 1],
        chain.gamma[rows - 1],
        released_mask,
    )


def reconnect(chain: LevChain) -> LevChain:
    """The aft corners of a joined row are the free corners of the row in front, exactly.

    After a convection the two coincide up to rounding; this makes them one point again so
    the chain shares its edges.  An unjoined row keeps its own corners.
    """

    rings = chain.rings.clone()
    if chain.rings.shape[0] > 1:
        joined = chain.joined[:, :, None]
        rings[1:, :, 2] = torch.where(joined, chain.rings[:-1, :, 1], chain.rings[1:, :, 2])
        rings[1:, :, 3] = torch.where(joined, chain.rings[:-1, :, 0], chain.rings[1:, :, 3])
    return chain.with_rings(rings)


def flat(chain: LevChain) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows*N, 4, 3) rings and (rows*N,) strengths, for the ring kernel."""

    return chain.rings.reshape(-1, 4, 3), chain.gamma.reshape(-1)


def active_rows(chain: LevChain) -> torch.Tensor:
    """(rows*N,) bool: the entries of `flat` that carry a row."""

    rows = chain.rings.shape[0]
    index = torch.arange(rows, device=chain.rings.device)[:, None]
    return (index < chain.count[None, :]).reshape(-1)


def truncation_audit(
    rings: torch.Tensor, circulation: torch.Tensor, left: torch.Tensor
) -> dict[str, object]:
    """What the rows that left the chains took with them (retention ledger).

    Count, signed circulation, and the closed-filament linear impulse
    0.5 * Gamma * sum_i (x_i x x_{i+1}) of every ring that left, as the trailing-edge wake's
    truncation is audited.  Circulation that leaves the model is never dropped silently.
    """

    kept = left.to(torch.bool)
    count = int(torch.count_nonzero(kept).item())
    if count == 0:
        return {"lev_truncate_ring_count": 0, "lev_truncate_circulation": 0.0,
                "lev_truncate_linear_impulse": [0.0, 0.0, 0.0]}
    corners = rings[kept].reshape(-1, 4, 3)
    cross_sum = torch.cross(corners, torch.roll(corners, -1, dims=1), dim=2).sum(dim=1)
    impulse = 0.5 * circulation[kept][:, None] * cross_sum
    return {
        "lev_truncate_ring_count": count,
        "lev_truncate_circulation": float(circulation[kept].sum().item()),
        "lev_truncate_linear_impulse": [float(value) for value in impulse.sum(dim=0).tolist()],
    }


def suction_force(
    density: float, speed: float, chord: torch.Tensor, width: torch.Tensor,
    forward: torch.Tensor, lesp: torch.Tensor,
) -> torch.Tensor:
    """(N, 3): the leading-edge suction of every strip, the paper's Eq. (20).

    F = pi rho c U^2 A0^2 per unit span, along the chord towards the leading edge (`forward`,
    a unit vector per strip), on the strip's width.
    """

    magnitude = math.pi * density * speed * speed * chord * lesp * lesp * width
    return magnitude[:, None] * forward


def solve_coupled(
    aic: torch.Tensor,
    nascent_normal: torch.Tensor,
    basis_normal: torch.Tensor,
    rhs: torch.Tensor,
    active: torch.Tensor,
    pin_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The circulation solve with the nascent leading-edge vortex of the active strips unknown.

    Unknowns: the bound circulation gamma (P) and the nascent leading-edge strength lam (k), k
    the number of active strips.  Equations:
      * no penetration on every row that is not the leading-edge row of an active strip: the
        bound sheet, the nascent trailing-edge row and the nascent leading-edge structures;
      * the leading-edge row of an active strip is pinned: its circulation is such that the
        suction parameter equals the critical value (`pin_target`, in circulation units);
      * one closure row per active strip: the no-penetration equation of that leading-edge row.
    The trailing-edge row of an active strip carries gamma_TE + lam, so lam enters through
    `nascent_normal` as well.  `aic` already carries the nascent trailing-edge row in the
    trailing-edge columns (the implicit Kutta row).  Returns gamma, lam, and the residual of all
    P no-penetration rows, the pinned ones included.
    """

    size = aic.shape[0]
    index = torch.nonzero(active.to(torch.bool), as_tuple=False).reshape(-1)
    count = index.shape[0]
    full = torch.cat((aic, nascent_normal[:, index] + basis_normal), dim=1)
    matrix = torch.zeros((size + count, size + count), dtype=rhs.dtype, device=rhs.device)
    right = torch.zeros(size + count, dtype=rhs.dtype, device=rhs.device)
    matrix[:size] = full
    right[:size] = rhs
    if count:
        matrix[size:] = full[index]
        right[size:] = rhs[index]
        matrix[index] = 0.0
        matrix[index, index] = 1.0
        right[index] = pin_target[index]
    solution = torch.linalg.solve(matrix, right)
    return solution[:size], solution[size:], full @ solution - rhs


__all__ = [
    "LEV_MAX_ROWS",
    "SPLIT_FRACTION",
    "LevChain",
    "active_rows",
    "continuing",
    "flat",
    "first_free_points",
    "nascent_row",
    "pseudovortex_ring",
    "push_nascent",
    "reconnect",
    "row_aft_pairs",
    "row_free_pairs",
    "solve_coupled",
    "split_points",
    "strip_pairs",
    "suction_force",
    "truncation_audit",
]
