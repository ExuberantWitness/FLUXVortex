"""Native CUDA FLUX-V5M aerodynamic owner for Q16 shell coupling.

The scientific data plane is Q16 surface interpolation -> vortex geometry ->
LEV/TEV/free-wake solve -> panel load -> exact Q16 transpose.  Every evolving
numerical array is CUDA float64; Python owns only topology and transactions.
"""
from __future__ import annotations

import os

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch
import warp as wp

from fluxvortex.aero.v5m.loads import LOAD_HISTORY_MODELS
from fluxvortex.aero.v5m.separation import reconcile_release_mask, unified_separation_mask
from fluxvortex.q16_ancf_mesh import Q16Mesh
from fluxvortex.q16_work_conjugate_transfer import Q16SurfaceTransferMap

from . import config as warp_config
from . import lev_hirato
from .kernels_q16_transfer import Q16CudaSurfaceTransfer
from .kernels_ring_velocity import (
    _KERNEL_V3,
    _MIXED_PRECISION as _RING_MIXED_PRECISION,
    mean_chordwise_panel,
    native_ring_velocity_expanded,
    native_ring_velocity_fused,
)

from ldvm_source_bank_gpu import CudaLDVMSourceBank
from pfield_torch_gpu import CudaParticleField


NATIVE_V5M_CONTRACT = "q16-flux-v5m-native-cuda-float64-v1"
LESP_FACTOR = 1.13
_FOUR_PI = 4.0 * math.pi
_EPS_V = 1.0e-9

#: Leading-edge pin form on the legacy (production) separated path (2026-11-24).
#:
#: Two LESP sign conventions meet at the pin.  The 2-D source bank follows Ramesh
#: (LESP == A0, A0 > 0 for positive suction; it returns A0 = +lesp_crit on a shed strip).
#: The 3-D side defines lesp_pre = -gamma_pre/scale, i.e. with one extra minus sign, and
#: the legacy pin then applies Gamma_LE = -A0_bank*scale -- so the 3-D minus sign is
#: applied to a quantity (the bank's A0) that never carried it.  Measured at runtime on
#: A16 (12 steps, 30/30 strips): bank A0 = +0.110, 3-D lesp_pre = -0.35, attached
#: gamma_pre = +0.004, pinned Gamma_LE < 0 -- the leading-edge row ends up with the
#: OPPOSITE sign to the attached circulation it is supposed to bound.  The separation mask
#: is |LESP| > crit, which is sign-symmetric, so this never changed WHICH strips separate;
#: it only changed what they are pinned to.
#:
#: The in-tree joint path already lands on the attached sign: it uses the SAME 3-D
#: definition (lesp_pre = -gamma_pre/scale) but writes its target as
#: -sign(lesp_pre)*crit*scale, which resolves to +sign(gamma_pre)*crit*scale.  And the
#: literature agrees: Hirato et al. 2019 (J. Aircraft, eq. 6) has A0 proportional to
#: Gamma_1 with no minus sign, so A0 and Gamma_1 share a sign.
#:
#:   legacy      Gamma_LE = -A0_bank*scale              (the historical behaviour)
#:   signed      Gamma_LE = +A0_bank*scale              (literature convention)
#:   signed_mag  Gamma_LE = +sign(gamma_pre)*|A0|*scale (sign from the 3-D solution itself)
#:
#: The choice between them is made ONLY by the error against the digitised experiment
#: (A16 Cn 0.92-0.95, zmax/c 0.043 on the converged window), not by argument.
_LE_PIN = os.environ.get("FLUXV_LE_PIN", "legacy").strip().lower()
if _LE_PIN not in {"legacy", "signed", "signed_mag"}:
    raise ValueError(
        f"FLUXV_LE_PIN must be 'legacy', 'signed' or 'signed_mag', got {_LE_PIN!r}"
    )

#: Where the NEWBORN leading-edge sheet sits relative to the leading edge.
#:
#: The deposit is a closed ribbon anchored on the physical leading-edge line, so its
#: first-row induction is set by how far AFT its return leg reaches -- closing the
#: sheet at the leading edge makes it a sliver whose two legs cancel.  Measured on the
#: real A16 geometry (platform/warp_vpm/diagnose_lev_ribbon_cancellation.py, note A14),
#: as a fraction of the monopole an open filament of the same strength induces at the
#: first collocation row:
#:
#:     chordwise d/c   0.001   0.005   0.01    0.02    1/30    0.05
#:     survives         2.1%   11.5%   24.8%   56.0%  103.6%  162.2%
#:     normal   d/c     0.0%    0.4%    1.5%    5.7%   14.8%   29.4%
#:
#: The production arm sits at 5.8% of the field average, i.e. in the chordwise-sliver
#: column.  Extending the return leg along the NORMAL direction does not help -- the
#: first collocation row is 0.5*dx AFT of the edge, so only chordwise extent changes
#: the distance difference between the two legs.
#:
#:   off    frontier = LE + reference_node_chord * (2-D birth displacement).  The
#:          historical behaviour: the sheet is a chordwise sliver, 0.08 panels deep
#:          on A16 (diagnose_roj_lev_feed.py prints it).
#:   chord  the same, with the CHORDWISE reach replaced by FLUXV_LE_SHEET_EXTENT_PANELS
#:          panels along the chord.  The normal offset of the 2-D birth point is kept.
#:
#: `chord` is a controlled sweep of ONE geometric quantity of the existing deposit.  It
#: is NOT the leading-edge vortex of Hirato et al. (Journal of Aircraft, doi:10.2514/1.C035124,
#: Sec. II.D), and an earlier version of this comment wrongly said it was.  That paper
#: does not widen a per-step loop: its new ring goes one third of the way from the
#: leading edge to the LAST SHED RING (x_n = 2/3 x_LE + 1/3 x_{n-2}, its Eq. 7), the
#: ring next to the edge is split, every ring keeps its FULL shedding-time strength,
#: and a pseudovortex ring spanning the chord cancels the filament on the edge.  The
#: steady sheet is therefore one loop from the edge to the rolled-up core, not a stack
#: of slivers.  A widened loop lying in the surface plane is bound vorticity, so
#: `chord` cannot be expected to produce a recirculation region; it exists to measure
#: how far the deposit's own extent carries, and to show that geometry alone is not the
#: fix.  Selection is by the error against the digitised experiment, never by
#: self-consistency.
_LE_SHEET_EXTENT = os.environ.get("FLUXV_LE_SHEET_EXTENT", "off").strip().lower()
if _LE_SHEET_EXTENT not in {"off", "chord"}:
    raise ValueError(
        "FLUXV_LE_SHEET_EXTENT must be 'off' or 'chord', "
        f"got {_LE_SHEET_EXTENT!r}"
    )
#: Chordwise reach of the `chord` arm in PANEL units (divided by chordwise_panels
#: where it is applied: `reference_node_chord` is the FULL chord, see the trap noted in
#: `_lev_sheet_frontier`).
_LE_SHEET_EXTENT_PANELS = float(os.environ.get("FLUXV_LE_SHEET_EXTENT_PANELS", "2.0"))

#: How the new trailing-edge wake row meets the circulation solve (plan V3, P0-B).
#:
#: The legacy order inside one step is: convect the wake, solve the bound circulation
#: against that convected wake, THEN insert the new row [rear, rear + U dt] and re-rivet the
#: previous newest row's front back to the trailing edge.  Two things follow, both measured
#: (diagnose_ring_lattice_steady_cpu.py --wake-layout, FLUX_V5M_PLAN_AUTHOR_REPLY_AND_V3
#: _20261125.md question 3):
#:
#:   * at the moment of the solve there is NO wake filament at the trailing edge: the bound
#:     trailing-edge leg stands uncancelled with a 1*U*dt gap behind it (layout "gap1",
#:     which alone is worth -35% of Cn at alpha = 5 deg once the kernel constant is gone);
#:   * after insertion rows 0 and 1 both start at the trailing edge (layout "overlap"):
#:     the new row is a closed ring one U*dt wide whose self-induced velocity throws its
#:     front corners off the edge (the "singular undershoot" the 1e-9 constant was hiding).
#:
#:   legacy  the order above.  The behaviour before 2026-11-25; now reached only by setting
#:           FLUXV_TE_WAKE=legacy.
#:   v3      the new row spans [trailing edge, convected front of the previous newest row]
#:           and is part of the linear solve: its induction enters the columns of the
#:           trailing-edge row (its strength is the trailing-edge circulation plus the LEV
#:           release), the part of it that is known before the solve (the LEV release)
#:           enters the right-hand side, and every load evaluation sees the same vortex
#:           system.  The insertion then simply stores the row that was solved for, so the
#:           wake is one continuous sheet and the trailing-edge legs cancel exactly.
#:           Default since 2026-11-25 (the owner's decision, with FLUXV_RING_EPS_V = 1e-12):
#:           on the rigid plate the legacy order at a small constant leaves Cn 15-25 % below
#:           the lattice (note A22, sections 1 and 6b).
#:
#: Either the reopening alone or the implicit row alone makes the result worse than legacy
#: (reopening alone: 0.1225 against 0.1364 at eps = 1e-11, plan V3 question 3), so v3
#: always does both.
_TE_WAKE_CHOICE = os.environ.get("FLUXV_TE_WAKE", "v3").strip().lower()
if _TE_WAKE_CHOICE not in {"legacy", "v3", "v4"}:
    raise ValueError(
        f"FLUXV_TE_WAKE must be 'legacy', 'v3' or 'v4', got {_TE_WAKE_CHOICE!r}"
    )
_TE_WAKE_V4 = _TE_WAKE_CHOICE == "v4"
_TE_WAKE_V3 = _TE_WAKE_CHOICE in {"v3", "v4"}
#: Connected Katz-Plotkin Sec. 13.12 split fraction under FLUXV_TE_WAKE=v4:
#: split_te = te_edge + beta * (convected_front - te_edge), keeping wake_rings[0]
#: CONNECTED to split_te (zero gaps, exact div(omega) = 0, exact G-VLM).
_TE_SPLIT_BETA = float(os.environ.get("FLUXV_TE_SPLIT_BETA", "0.3"))
if not (math.isfinite(_TE_SPLIT_BETA) and 0.0 < _TE_SPLIT_BETA <= 1.0):
    raise ValueError(
        f"FLUXV_TE_SPLIT_BETA must be in (0, 1], got {_TE_SPLIT_BETA!r}"
    )
#: Plan V4 Track A-1: synchronized t^{n-1} convection using bound_rings_previous,
#: unique structured wake vertices (3.9x fewer targets), live LEV compaction,
#: unified wake+LEV+pseudovortex velocity field, and non-frozen stage-2 positions.
_CONVECTION_V4 = os.environ.get("FLUXV_CONVECTION_V4", "0").strip().lower() in {
    "1", "true", "on", "yes",
}
#: Stage-2 RK2 TE/LE closure bridge under FLUXV_CONVECTION_V4=1 (default ON when v4 is
#: enabled; production default has _CONVECTION_V4=False so bit-identical).  Prevents the
#: uncancelled -Gamma_TE / +Gamma_LE filaments torn open when w_pred/l_pred advance away
#: from frozen t^{n-1} bound_src + pseudovortex at RK2 stage 2.
_CONVECTION_V4_STAGE2_BRIDGE = os.environ.get(
    "FLUXV_CONVECTION_V4_STAGE2_BRIDGE", "1"
).strip().lower() in {"1", "true", "on", "yes"}
#: Matched-horizon gauge-shift rebasing for Hirato v2 LEV chains during multi-cycle
#: asymmetric lift-producing flapping (default OFF, bit-identical production).  When enabled
#: (`FLUXV_LEV_GAUGE_REBASE=1`), once a strip's LEV chain reaches `wake_max_rows`, its
#: truncated tail circulation `g_drop` is subtracted simultaneously from the remaining LEV
#: chain, pseudovortex (`chain.prev`), and TE wake of that strip — preserving exact Kelvin
#: / LE / TE cancellations while eliminating secular cross-cycle `chain.prev` ratcheting.
_LEV_GAUGE_REBASE = os.environ.get("FLUXV_LEV_GAUGE_REBASE", "0").strip().lower() in {
    "1", "true", "on", "yes",
}
#: Optional tapered-wingtip nascent LEV split cap in local leading-edge panel lengths dx_0(j)
#: (default 0.0 = OFF for bit-identical Hirato Eq. 7 on rectangular wings).  On sharply tapered
#: or rounded bird wingtips where U_rel * dt / 3 > 0.75 * dx_0(j), capping the split length at
#: e.g. 0.40 * dx_0(j) keeps the nascent LEV filament strictly ahead of panel 0's collocation
#: point while remaining on the exact (LE, old_free) segment for exact Helmholtz cancellation.
_LEV_SPLIT_CAP_DX = float(os.environ.get("FLUXV_LEV_SPLIT_CAP_DX", "0.0"))
if not (math.isfinite(_LEV_SPLIT_CAP_DX) and _LEV_SPLIT_CAP_DX >= 0.0):
    raise ValueError(
        f"FLUXV_LEV_SPLIT_CAP_DX must be finite and >= 0, got {_LEV_SPLIT_CAP_DX!r}"
    )
#: Plan V4 Track A-2: optional wake core multiplier at wing collocation points
#: (0 / false = uncored for exact G-VLM cancellation at split_te; 1 / true = full
#: _core_radius; or a positive float fraction like 0.25).
_WAKE_CORE_AT_COLLOCATION_RAW = os.environ.get(
    "FLUXV_WAKE_CORE_AT_COLLOCATION", "0"
).strip().lower()
if _WAKE_CORE_AT_COLLOCATION_RAW in {"0", "false", "off", "no", ""}:
    _WAKE_CORE_AT_COLLOCATION = 0.0
elif _WAKE_CORE_AT_COLLOCATION_RAW in {"1", "true", "on", "yes"}:
    _WAKE_CORE_AT_COLLOCATION = 1.0
else:
    _WAKE_CORE_AT_COLLOCATION = float(_WAKE_CORE_AT_COLLOCATION_RAW)
if not (math.isfinite(_WAKE_CORE_AT_COLLOCATION) and _WAKE_CORE_AT_COLLOCATION >= 0.0):
    raise ValueError(
        f"FLUXV_WAKE_CORE_AT_COLLOCATION must be finite and >= 0, got {_WAKE_CORE_AT_COLLOCATION!r}"
    )
#: Multiplier on _core_radius(geometry) for LEV chain + pseudovortex at wing
#: collocation points and load midpoints under FLUXV_LEV_MODEL=hirato_v2.
#: 1.0 = same uniform v3 core as convection; 0.0 = singular (unregularized).
_LEV_CORE_RHS_FRACTION = float(os.environ.get("FLUXV_LEV_CORE_RHS_FRACTION", "1.0"))
if not (math.isfinite(_LEV_CORE_RHS_FRACTION) and _LEV_CORE_RHS_FRACTION >= 0.0):
    raise ValueError(
        f"FLUXV_LEV_CORE_RHS_FRACTION must be finite and >= 0, got {_LEV_CORE_RHS_FRACTION!r}"
    )
#: Wing-impenetrability slip-wall clearance in chordwise panel lengths dx (Hirato PhD
#: dissertation Sec. 4.8.3 / Ramesh 2014 / Bird 2021).  0.0 = disabled (default, bit-identical
#: production path); > 0 (e.g. 0.25) projects any free wake/LEV/particle vertex that crosses
#: or approaches within `clearance * dx` of the solid wing surface back onto its incoming side.
_WING_SLIP_WALL = float(os.environ.get("FLUXV_WING_SLIP_WALL", "0.0"))
if not (math.isfinite(_WING_SLIP_WALL) and _WING_SLIP_WALL >= 0.0):
    raise ValueError(
        f"FLUXV_WING_SLIP_WALL must be finite and >= 0, got {_WING_SLIP_WALL!r}"
    )
#: Stage 3 opt-in rigid-solver load deduplication: in bound_rate mode, skip the unused
#: material_ring_velocity_derivative + mf2_1 LU + discarded super().propose() packet pass
#: that RigidNativeV5MSolver immediately replaces.  Default OFF (bit-identical production).
_FAST_RIGID_LOADS = os.environ.get("FLUXV_FAST_RIGID_LOADS", "0").strip().lower() in {
    "1", "true", "on", "yes",
}
#: Stage 2B Squire/Ramasamy-Leishman dimensionless turbulent core growth coefficient
#: c_sq = 4 * alpha_LO * a_1: r_c(age)^2 = r_0^2 + c_sq * |Gamma| * age_s.
#: Scale-invariant across wing sizes and Re; row 0 (age=0) stays sharp at r_0 so
#: nascent LESP shedding and G-VLM steady lift are untouched while older rolled-up
#: LEV/wake cores spread physically instead of suffering K-H filament-stretching chaos.
#: Default 0.0 (OFF, bit-identical production).
_SQUIRE_CORE = float(os.environ.get("FLUXV_SQUIRE_CORE", "0.0"))
if not (math.isfinite(_SQUIRE_CORE) and _SQUIRE_CORE >= 0.0):
    raise ValueError(
        f"FLUXV_SQUIRE_CORE must be finite and >= 0, got {_SQUIRE_CORE!r}"
    )
#: Galilean-invariant leading-edge relative speed U_rel = |v_inf - v_LE| in LESP scaling
#: (Ramesh 2014 Eq. 2.11 / Izraelevitz 2017): reduces identically to U_inf on stationary
#: wings and remains exact under large plunge/flapping velocity and hover.  Default OFF.
_LESP_UREL = os.environ.get("FLUXV_LESP_UREL", "0").strip().lower() in {
    "1", "true", "on", "yes",
}
#: Hirato et al. (2019) Eqs. (16)-(20) Bernoulli surface pressure + Eq. (20) LESP-limited
#: leading-edge suction on RigidNativeV5MSolver (eliminates LEV-induced phantom suction on
#: panel 0's Kutta-Joukowski front filament during separated flow).  Default OFF.
_HIRATO_PRESSURE_LOADS = os.environ.get(
    "FLUXV_HIRATO_PRESSURE_LOADS", "0"
).strip().lower() in {"1", "true", "on", "yes"}
#: First-principles LESP-consistent Kutta-Joukowski loads (default OFF): uses exact
#: Kutta-Joukowski filament forces everywhere the leading edge is attached, and on strips
#: with an active/continuing LEV sheet (`lev_chain.prev != 0`) replaces panel 0's front
#: filament tangential component with Hirato Eq. (20) LESP-limited leading-edge suction
#: while keeping its normal KJ component (eliminates phantom near-LE LEV suction without
#: losing 1/4-panel filament accuracy on attached/interior panels; exact packet closure).
_LEV_SEP_SUCTION = os.environ.get(
    "FLUXV_LEV_SEP_SUCTION", "0"
).strip().lower() in {"1", "true", "on", "yes"}

#: Core radius of the free-wake filaments, in chordwise panel lengths.  Used only under
#: FLUXV_RING_KERNEL=v3, and only where free vorticity is evaluated close to filaments: the
#: convection of free material points (wake nodes, particles), where it acts on every
#: filament they see, bound sheet included, and the wake filaments of the load evaluation.
#: The influence matrix is never cored.  Tied to the panel size, never to U*dt and never to
#: a fixed number of metres, so that it scales with the wing.  The plan's range is 0.25-0.5
#: panel (G-Flap chooses within it); 0 switches the core off.  Why the bound sheet needs it
#: at the trailing-edge junction: `Q16NativeV5MSolver._core_radius`.
_WAKE_CORE_PANELS = float(os.environ.get("FLUXV_WAKE_CORE_PANELS", "0.25"))
if not (math.isfinite(_WAKE_CORE_PANELS) and _WAKE_CORE_PANELS >= 0.0):
    raise ValueError(
        f"FLUXV_WAKE_CORE_PANELS must be finite and non-negative, got {_WAKE_CORE_PANELS!r}"
    )

#: Core of the free wake under the legacy kernel (plan V3, Bug-4).  That kernel's core radius
#: is `core_fraction * max(longest edge of the ring, the wing's chordwise panel)`; the author's
#: code reaches it only through `rough=True` (0.1), which nothing ever passed, so with
#: core_fraction 1e-6 the free wake had no core and the 1e-9 constant was doing that job.
#:
#: Measured 2026-11-25 on the flapping wing (Izraelevitz IZRA-15-015, 8 x 24 panels, U dt =
#: 4.8 mm) with the constant at 1e-12 and the implicit trailing-edge row and no core: three
#: cycles periodic and sane (suction parameter up to 0.36), then at step 329 it goes
#: 0.17 -> 22.7 -> 459 and the force with it.  The wing passes through its own wake; a filament
#: sweeping across the surface passes through every distance, and at 1e-12 the constant only
#: tames those below 0.03 mm (0.2 mm at 1e-9).  With this core a filament 0.1 mm from a point
#: is seen at 0.6 % of the raw law.  0 switches the core off.
#:
#: It acts wherever free material points are convected (every filament they see, the bound
#: sheet included, so that the trailing-edge legs still cancel each other) and wherever the
#: wake is evaluated at the collocation points and at the load midpoints.  Never in the
#: influence matrix.
_FREE_WAKE_CORE_FRACTION = float(os.environ.get("FLUXV_FREE_WAKE_CORE_FRACTION", "0.1"))
if not (math.isfinite(_FREE_WAKE_CORE_FRACTION) and _FREE_WAKE_CORE_FRACTION >= 0.0):
    raise ValueError(
        "FLUXV_FREE_WAKE_CORE_FRACTION must be finite and non-negative, got "
        f"{_FREE_WAKE_CORE_FRACTION!r}"
    )

#: Core radius of the free wake taken from the time step instead of from the panel size
#: (2026-11-26, default OFF: 0).  Positive values are in U_inf * aerodynamic_dt, and the value
#: replaces `FLUXV_WAKE_CORE_PANELS` wherever the v3 kernel cores the free wake.
#:
#: The two papers that state a core for a shed sheet both scale it with the step, not with the
#: wing's panel:
#:   * Li et al., JFM 972, A30, Sec. 3: every free vortex carries the Vatistas core of radius
#:     r_core = U_inf * dt; only the vortices fixed on the plate use Rankine.
#:   * Hirato et al., Eq. (25): the Lamb-Oseen radius "is set to a small fraction, typically
#:     less than 0.5, of the anticipated smallest vortex-ring dimension"; a wake whose rows are
#:     one convection step apart has 0.5 U_inf dt as that bound.
#: The panel rule (`FLUXV_WAKE_CORE_PANELS` = 0.25, and the 0.1 of `FLUXV_FREE_WAKE_CORE_FRACTION`
#: on the legacy kernel) is a fraction of the panel and says nothing about how far the wake moves
#: per step.  On the published flapping cases the step is 0.88 chordwise panels (U = 11 m/s,
#: dt = 1 ms, c = 0.1 m, 8 panels), so the panel rule gives 0.11-0.28 U_inf dt against Li's 1.0
#: and Hirato's 0.5, and the tip vortex then winds up inside its own filaments: K2 at alpha = 0
#: with the panel rule diverges at step 102 (note A24, section 4.1), and with a core of
#: 0.5 * U_inf dt it is periodic for ten cycles.  0 keeps the panel rule, bit for bit.
_FREE_WAKE_CORE_STEPS = float(os.environ.get("FLUXV_FREE_WAKE_CORE_STEPS", "0.0"))
if not (math.isfinite(_FREE_WAKE_CORE_STEPS) and _FREE_WAKE_CORE_STEPS >= 0.0):
    raise ValueError(
        "FLUXV_FREE_WAKE_CORE_STEPS must be finite and non-negative, got "
        f"{_FREE_WAKE_CORE_STEPS!r}"
    )

#: The shed-vortex distance of the wake, in U_inf * dt (2026-11-26, default OFF: 0).
#:
#: Both Kumar papers state it in so many words and both cite Katz and Plotkin for it:
#:   * Kumar et al., arXiv 2309.14832, Sec. 3.1: "At each time step, a wake vortex is shed at a
#:     distance of 0.3 U_inf dt from the wing tip".
#:   * Kumar et al., Sadhana 51:103 (2026), Sec. 3: "with a wake vortex being released at a
#:     distance of 0.3 U_inf dt from the wing tip at each time step [12]".
#: A positive value sheds the newborn row as its own ring, from the trailing edge downstream by
#: that distance, with the rows no longer sharing their corners: the classical discrete wake,
#: in which every ring is a separate vortex element that drifts with the flow.  The connected
#: material sheet (rows sharing an edge exactly, the front one anchored on the wing) is what this
#: solver does by default; the flag is here to measure the difference, and it is NOT the
#: production path.
_SHED_DISTANCE_FRACTION = float(os.environ.get("FLUXV_SHED_DISTANCE_FRACTION", "0.0"))
if not (math.isfinite(_SHED_DISTANCE_FRACTION) and _SHED_DISTANCE_FRACTION >= 0.0):
    raise ValueError(
        "FLUXV_SHED_DISTANCE_FRACTION must be finite and non-negative, got "
        f"{_SHED_DISTANCE_FRACTION!r}"
    )

#: The leading-edge vortex (plan V3, stage 2).
#:
#:   legacy   the 2-D source bank gives the strength of every strip's release, the 3-D
#:            leading-edge row is pinned, and a closed ribbon 0.08 panels deep carries the
#:            increment into the particle field.  Its induction at the first collocation row is
#:            5.8 % of an open filament's (note A14): the wing never feels its own leading-edge
#:            vortex, and the strength does not converge with the time step.
#:   hirato   the sheet of Hirato et al. (lev_hirato.py): a chain of full-strength rings per
#:            strip kept as exact rings (no particles), the nascent ring on the geometric
#:            leading edge, a pseudovortex ring across the chord, the new ring at one third of
#:            the way to the last one, and the strength
#:            of the nascent ring solved in the 3-D system together with the bound circulation:
#:            the leading-edge row is pinned at the critical suction parameter and no
#:            penetration holds everywhere else and at that leading-edge row too.  The 2-D
#:            source bank is not used.  The paper states that its method applies to wings of
#:            AR 4 and higher; the wings here are below that.
#:
#: Default hirato since 2026-11-25 (the owner's decision).
#:   hirato_v2  the same sheet with the five defects of note A24 section 5 repaired, each one
#:            written as an identity in `tests/test_lev_hirato_equations_cpu.py`:
#:              B1  the previous ring is split BEFORE the right-hand side is built, and the
#:                  previous step's pseudovortex rings are in that right-hand side and in the
#:                  convection (`lev_hirato.split_row0`, `free_ring_sources`), so the geometry the
#:                  solve sees is the geometry that gets committed;
#:              B2  the pseudovortex rings are part of every free-material convection
#:                  (`free_ring_sources`), instead of leaving an uncancelled +-Gamma_L filament
#:                  on the leading and trailing edges;
#:              B3  Eq. (7) measures one third of the way to the sheet's FREE end
#:                  (`split_row0`), not to its leading-edge-side end, which made every new ring a
#:                  third of one convection step long whatever the sheet's age;
#:              B4  cessation is decided on the increment and adds nothing, so the strip keeps its
#:                  Gamma_L and the unsteady term sees no step (`cessation_delta`);
#:              B5  the leading-edge chordwise difference carries Gamma_0 and not Gamma_0 + Gamma_L
#:                  (`leading_edge_chordwise_difference`);
#:              B6  the row count comes from FLUXV_LEV_MAX_ROWS (`max_rows`).
#:            Switches on: FLUXV_RING_KERNEL=v3 (one core radius for the shared edges).
_LEV_CHOICE = os.environ.get("FLUXV_LEV_MODEL", "hirato").strip().lower()
if _LEV_CHOICE not in {"legacy", "hirato", "hirato_v2"}:
    raise ValueError(
        f"FLUXV_LEV_MODEL must be 'legacy', 'hirato' or 'hirato_v2', got {_LEV_CHOICE!r}"
    )
_LEV_HIRATO = _LEV_CHOICE in {"hirato", "hirato_v2"}
#: The repaired sheet (note A24 section 5).  Everything the old `hirato` path does stays
#: bit-identical when this is False.
_LEV_HIRATO_V2 = _LEV_CHOICE == "hirato_v2"
#: B5 also changes the loads of the repaired path only.
_LEV_HIRATO_V2_LOADS = _LEV_HIRATO_V2

#: How the repaired sheet decides how much circulation a strip sheds (note A24, section 5).
#:
#:   pin   Hirato et al.'s own rule: the nascent strength is solved so that the strip's suction
#:         parameter equals the critical value.  Measured on their own pitch ramp this is a loop
#:         with gain: the sheet's induction raises the attached suction parameter and the pin
#:         answers with a larger strength; the sheet reached 53 m^2/s (177 U c) and the run ended.
#:   flux  the sharp-edge vorticity-flux law instead (Kaden/Didden, `dGamma = 1/2 U_s^2 dt`;
#:         `lev_hirato.flux_shedding_strength`): bounded by construction, and the suction
#:         parameter is then free to fall below its critical value, as Martinez-Carmena et al.
#:         show it must (their Eqs. 8-12).
_LEV_SHED_LAW = os.environ.get("FLUXV_LEV_SHED_LAW", "pin").strip().lower()
if _LEV_SHED_LAW not in {"pin", "flux"}:
    raise ValueError(
        f"FLUXV_LEV_SHED_LAW must be 'pin' or 'flux', got {_LEV_SHED_LAW!r}"
    )
_LEV_FLUX = _LEV_HIRATO_V2 and _LEV_SHED_LAW == "flux"

#: The sheet detaches from the leading edge (2026-11-26, default OFF; needs FLUXV_LEV_MODEL=hirato_v2).
#:
#: Hirato et al. stop the shedding when the suction parameter falls below its critical value.
#: Measured on their own pitch ramp, that never happens in this ring discretisation: the sheet's
#: induction at its own leading-edge row starts out against the freestream (at onset, t* = 1.71,
#: -0.09 of the freestream's normal component, 82 % of the strips) and turns WITH it as the sheet
#: grows (+0.42, 87 % of the strips at t* = 2.04, with the sheet carrying 108 U c inside a quarter
#: chord).  A negative contribution relieves the leading edge and lowers the suction parameter; a
#: positive one raises it, so the strip sheds more and the loop closes -- which is why both the
#: pinned closure and the flux law ended the same way (note A25, section 4).
#:
#: With this switch a strip stops being fed the moment its own sheet's induction at its
#: leading-edge row acts WITH the freestream instead of against it -- the measured mechanism,
#: threshold zero, no new constant.  The strip's existing rings keep convecting; the strip itself
#: returns to the attached solve, which is the state Hirato et al. leave a strip in once its
#: shedding stops.
#: Circulation a strip's sheet may carry before it stops being fed, in U_inf * chord.  The value is
#: the detachment cap already cited in these notes (note A23, L2-4: "a literature detachment cap of
#: 4.2"); a leading-edge vortex detaches at 4-5 U c, and a sheet that keeps growing past it has
#: nothing to do with the flow it is meant to represent (measured here: 108 U c, note A25).
_LEV_CAP = float(os.environ.get("FLUXV_LEV_CAP", "4.2"))

#: How close to the leading edge (in chords) a detached strip's sheet has to come before the strip
#: is allowed to shed again.  This is the latch of the criterion below: without it the sign test
#: toggles the strip on and off every few steps.
_LEV_DETACH_CHORDS = float(os.environ.get("FLUXV_LEV_DETACH_CHORDS", "1.0"))
#: The third state of a strip (2026-11-26, default OFF; needs FLUXV_LEV_DETACH): separated but no
#: longer feeding a sheet.  A detached strip's leading-edge row stops being a no-penetration row --
#: a separated leading edge does not satisfy no-penetration -- and carries the strength of the sheet
#: that left it instead (continuity of the vortex sheet across the separation point).  Without this
#: third state the strip returns to the fully attached solve, which at 45 deg incidence, with a
#: strong sheet still overhead, has no steady answer: measured on the H case, CL swung between
#: -4.6 and +8.4 from step to step while the sheet was frozen (note A25, section 4).
_LEV_SEPARATED_STATE = os.environ.get("FLUXV_LEV_SEPARATED_STATE", "off").strip().lower() in {
    "1", "true", "on", "yes",
}
_LEV_DETACH = os.environ.get("FLUXV_LEV_DETACH", "off").strip().lower() in {
    "1", "true", "on", "yes",
} and _LEV_HIRATO_V2

#: Hirato's leading-edge suction, pi rho c U^2 A0^2 per unit span along the chord towards the
#: leading edge (the paper's Eq. 20), as a force on the front filament of the first panel of each
#: strip, which is where the lattice keeps its leading-edge vortex (the geometric leading edge has
#: no bound filament).  FLUXV_LE_SUCTION=on switches it on.
#:
#: Off by default, on the evidence of 2026-11-25 (Izraelevitz IZRA-15-015, experiment 0.123 and
#: 0.145): with the ring chains of Hirato and this term the thrust coefficient is 0.385, without
#: it 0.137; the old deposit gave 0.200.  The loads here are Kutta-Joukowski vector forces on the
#: bound filaments evaluated in a velocity field that contains the leading-edge rings, so on a
#: shedding strip the suction is already in the vector force and the explicit term counts it
#: twice.  On an attached strip the picture is different: the rigid plate at 5 deg, 150 steps,
#: has a streamwise coefficient of +0.0085 without the term and -0.0013 with it, which is the
#: inviscid flat plate with its leading-edge suction.  Whether the term belongs on attached
#: strips is not settled; it is left to the flag.
_LE_SUCTION = os.environ.get("FLUXV_LE_SUCTION", "off").strip().lower() in {"1", "true", "on", "yes"}


def _require_cuda64(name: str, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device != device or value.device.type != "cuda":
        raise ValueError(f"{name} must remain on {device}")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use float64")
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def _clone_value(value: Any) -> Any:
    if type(value) is torch.Tensor:
        return value.clone()
    if type(value) is dict:
        return {key: _clone_value(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_value(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone_value(item) for item in value)
    return value


def _clone_source_bank(source: CudaLDVMSourceBank) -> CudaLDVMSourceBank:
    result = CudaLDVMSourceBank.__new__(CudaLDVMSourceBank)
    result.__dict__.update({key: _clone_value(value) for key, value in source.__dict__.items()})
    result.cuda_stream = torch.cuda.current_stream(result.device)
    return result


def _clone_particle_field(source: CudaParticleField) -> CudaParticleField:
    result = CudaParticleField.__new__(CudaParticleField)
    for slot in CudaParticleField.__slots__:
        setattr(result, slot, _clone_value(getattr(source, slot)))
    return result


def _axis_owner(value: float, element_count: int) -> tuple[int, float]:
    scaled = value * element_count
    owner = min(int(math.floor(scaled)), element_count - 1)
    local = 2.0 * (scaled - owner) - 1.0
    tolerance = 128.0 * np.finfo(np.float64).eps
    if local < -1.0 - tolerance or local > 1.0 + tolerance:
        raise ValueError("surface coordinate left its Q16 owner")
    return owner, min(max(local, -1.0), 1.0)


def _surface_map(
    mesh: Q16Mesh,
    chordwise_element_count: int,
    spanwise_element_count: int,
    fractions: tuple[tuple[float, float], ...],
) -> Q16SurfaceTransferMap:
    element_indices: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    for chord_fraction, span_fraction in fractions:
        chord_owner, xi = _axis_owner(chord_fraction, chordwise_element_count)
        span_owner, eta = _axis_owner(span_fraction, spanwise_element_count)
        element_indices.append(span_owner * chordwise_element_count + chord_owner)
        coordinates.append((xi, eta, 0.0))
    return Q16SurfaceTransferMap(
        mesh=mesh,
        element_indices=np.ascontiguousarray(element_indices, dtype=np.int64),
        parametric_coordinates=np.ascontiguousarray(coordinates, dtype=np.float64),
    )


def compute_lesp_crit(
    thickness_ratio: float,
    reynolds: float,
    *,
    lesp_flat: float = 0.11,
    thickness_reference: float = 0.013,
    c_thickness: float = 2.0,
    c_reynolds: float = 0.35,
    re_reference: float = 30_000.0,
) -> float:
    """Physics-based critical LESP from airfoil thickness and Reynolds number.

    Correlation calibrated to published data points:
      - Yamano flat plate t/c=1.3%, Re=20k → LESPcrit = 0.11 (Ramesh 2013)
      - NACA 0012 t/c=12%, Re=10k          → LESPcrit = 0.29 (Martinez-Carmena)
      - NACA 0012 t/c=12%, Re=30k          → LESPcrit ≈ 0.32 (Narsipur 2022)
      - NACA 0012 t/c=12%, Re=3M           → LESPcrit ≈ 0.50 (Narsipur 2022)

    Form:
      LESPcrit = LESP_flat + C_t · max(t/c − t₀, 0) · (1 + C_Re · log10(Re/Re₀))

    where t₀ = thickness_reference ≈ flat-plate threshold below which the
    thickness correction vanishes.
    """
    if not (math.isfinite(thickness_ratio) and 0.0 <= thickness_ratio <= 0.5):
        raise ValueError("thickness_ratio must lie in [0, 0.5]")
    if not (math.isfinite(reynolds) and reynolds > 0.0):
        raise ValueError("reynolds must be finite and positive")
    excess_thickness = max(thickness_ratio - thickness_reference, 0.0)
    reynolds_factor = 1.0 + c_reynolds * math.log10(
        max(reynolds, 1.0) / re_reference)
    return lesp_flat + c_thickness * excess_thickness * max(
        reynolds_factor, 0.5)


@dataclass(frozen=True, slots=True)
class NativeV5MConfig:
    chordwise_panels: int = 15
    spanwise_panels: int = 10
    density: float = 1.225
    freestream: float = 10.0
    # Geometric angle of attack of the freestream in the scientific frame
    # (+x chordwise, +z displacement direction): v_inf = U(cos a, 0, sin a).
    # The historical Yamano path leaves this at 0.0.
    freestream_angle_deg: float = 0.0
    aerodynamic_dt: float = 0.0068
    lesp_crit: float = 0.11
    lesp_thickness_ratio: float | None = None
    lesp_reynolds: float | None = None
    # Documented per-case release-threshold hypothesis (e.g. the Izraelevitz
    # Fig.14 static-polar Lcrit = sin(CLmax/CLa)).  This is a direct,
    # caller-owned override with an explicitly stated provenance — it is NOT
    # the thickness/Re correlation (which for the real NACA 63A015 t/c=0.15
    # at Re=310k would give ~0.481) and NOT an escape hatch for the frozen
    # Yamano default: ``lesp_crit`` itself stays pinned at 0.11.
    lesp_crit_override: float | None = None
    dvm_ndiv: int = 20
    dvm_naterm: int = 8
    dvm_max_wake: int = 64
    wake_max_rows: int = 96
    particle_capacity: int = 8192
    # Age cap (aerodynamic steps) for LEV particles, mirroring the ring-wake
    # truncation: 0 disables culling (the historical Yamano behavior, which
    # never sheds particles).  Sustained-release cases must set a positive
    # cap or the particle field grows without bound and exceeds capacity.
    particle_max_age_steps: int = 0
    # Wake-history pressure term: "material" is the author's strong-scheme
    # Mf2_vec1 (A^-1 applied to the wake-motion material derivative) kept by
    # the oracle-verified Yamano path; "bound_rate" is the author's
    # weak-scheme dp_add = (Gamma - Gamma_old)/dt, which stays bounded in
    # sustained-LEV-release regimes where Mf2_vec1 diverges.  These are
    # distinct PHYSICS models with registered identities (aero/v5m/loads.py,
    # plan §8.7) — not interchangeable runtime stability switches.
    wake_history_mode: str = "material"
    # Number of newest wake rows (shed events) that keep full induced
    # convection; older rows convect with the freestream alone (the author's
    # far-wake freeze in generate_wake.m).  0 disables the freeze (the
    # historical fully-free wake, which is O(rows^2) per step).
    wake_free_rows: int = 0
    dvm_core_radius_chord: float = 0.02
    dvm_smoothing_radius_chord: float = 0.04
    dvm_target_spacing_chord: float = 0.04 / 2.125
    gate_rtol: float = 1.0e-8
    # B1 (U4): linearize the LOAD operators (Mf1, Mf2_1) on the SAME bound
    # constraint set that produced gamma, instead of the all-Neumann AIC.
    # Exact for the legacy pinned/separated system (the pin target is
    # velocity- and acceleration-independent); on the joint path it uses the
    # pinned operator and omits the newborn-coupling Schur term, which is
    # recorded in the diagnostics as 'pinned_ignores_newborn_coupling'.
    # Default OFF: the production path keeps the historical operator.
    load_derivative_active_set: bool = False
    # B2 diagnostic (U2): record, on the same state, the TE-leg force implied
    # by the bound-rate history (Gamma_n - Gamma_{n-1}) versus the strength of
    # the wake row actually inserted (gamma_tev = Gamma_TE + gamma_lev).
    # Measurement only -- no production path reads the result.  Default OFF.
    te_leg_candidate_diagnostic: bool = False
    # E1-FLUX (experimental, default OFF): prescribe the shed leading-edge
    # circulation from the sharp-edge vorticity-flux law
    # dGamma = 0.5*U_s^2*dt instead of solving it from the LE
    # no-penetration closure.  Makes Gamma_shed proportional to dt (bounded);
    # see the L2-1 oracle finding (ratio 22.9 -> 48.5 when dt halves).
    flux_consistent_shedding: bool = False
    # E1-WINDOW (0 = OFF, production default): restrict the particles that may
    # enter the sharp-LE closure RHS to the most recent
    # `le_constraint_window_steps` steps.  Measured rationale: the pre-blow-up
    # LE induction is dominated by the ACCUMULATED wake (older 26%, rows 6-10
    # 25%, rows 4-5 22%, newest 3%, mean distance 1.3c), i.e. far-field wake
    # drives a LOCAL separation judgement and ratchets newborn.  The far wake
    # stays fully active in the non-LE rows and in the loads.  For the frozen
    # dt*=0.01, one chord of convection is 1/dt* = 100 steps, so 100 is a
    # DERIVED window, not a searched one.
    le_constraint_window_steps: int = 0
    # P1 joint separation solve (experimental, default OFF). Uses the real
    # deposited ribbon basis and an independent deposited-flow gate. A
    # small instantaneous residual does not certify sustained separation;
    # material-sheet topology, history and wake feedback remain unresolved.
    joint_separation_solve: bool = False
    # Route A experiment (default OFF): deposit the newborn LE sheet OPEN
    # (spanwise filament only, net vorticity -gamma_lev) instead of closing it
    # into a zero-circulation loop.  Measured (S2.0, diagnose_roj_le_kernel_sign)
    # the closed deposit induces only 3.9% (A15) / 4.5% (A19) of a monopole's
    # normal velocity at the first collocation row, because the closure sheet sits
    # ds=0.0049c away and cancels the LE sheet to 96%; the joint closure then
    # inflates gamma_lev ~25x (independent cross-check: L2-1 measured 22.9), which
    # is the gamma_lev ~ 1/dt runaway.  Default OFF keeps the production path
    # bit-identical.
    le_open_deposit: bool = False
    # Route A sub-variant (default OFF): the closed ribbon contributed zero net
    # vorticity, so the shed amount gamma_lev was booked into the TE wake row as
    # ``gamma_tev = Gamma_TE + gamma_lev``.  With an open LE deposit that same
    # circulation now sits at the leading edge, where the physics puts it; this
    # flag removes it from the TE row so the total is not counted twice.  Which
    # variant is right is decided by measurement, not by assertion.
    le_open_deposit_drop_tev_compensation: bool = False
    # Stage 2 experiment (default OFF): 3-D vortex stretching of the particle
    # strengths, d(gamma)/dt = S (- 3 Z gamma in the rVPM form), where the
    # SCHEME for S is ``particle_stretching_transposed`` below, plus the rVPM
    # core-radius evolution
    # d(sigma)/dt = -sigma Z when ``particle_core_spreading`` is also set.
    # Nothing else in the pipeline updates a particle's strength after birth, so
    # the LEV's circulation accumulates without bound (L2-4: 82 x cU by step 300
    # against a literature detachment cap of 4.2) and the zero-vorticity dipole
    # deposit merely hides that from the loads (S2.0/A1-A3).  Stretching is the
    # physical saturation mechanism, so it is measured, not asserted.
    particle_stretching: bool = False
    particle_core_spreading: bool = False
    # Which stretching form the flag above uses.  False = bare Helmholtz
    # dgamma/dt = S (the version measured first, kept for A/B attribution);
    # True = the frozen rVPM form of research_rvpm_arch.md V4, which adds the
    # redistribution term -3 Z gamma that bounds a stretched element instead of
    # letting |gamma| accumulate without limit.
    particle_stretching_rvpm: bool = False
    # Stretching scheme.  True (default) = FLOWVPM ``transposed=true`` /
    # research_rvpm_arch.md V4: S_i = J[m,i] gamma_m.  False = the classic
    # pointwise Helmholtz term S_i = J[i,m] gamma_m.  The two differ by the sign
    # of the local omega x gamma reorientation and only the transposed one
    # cancels pairwise for equal-core particles; exposing both keeps the classic
    # implementation reachable for A/B attribution instead of silently flipping.
    particle_stretching_transposed: bool = True
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if type(self.chordwise_panels) is not int or self.chordwise_panels < 2:
            raise ValueError("chordwise_panels must be an exact int >= 2")
        if type(self.spanwise_panels) is not int or self.spanwise_panels < 1:
            raise ValueError("spanwise_panels must be a positive exact int")
        if type(self.particle_max_age_steps) is not int or self.particle_max_age_steps < 0:
            raise ValueError("particle_max_age_steps must be a non-negative exact int")
        if type(self.joint_separation_solve) is not bool:
            raise TypeError("joint_separation_solve must be an exact bool")
        if type(self.le_open_deposit) is not bool:
            raise TypeError("le_open_deposit must be an exact bool")
        if type(self.le_open_deposit_drop_tev_compensation) is not bool:
            raise TypeError(
                "le_open_deposit_drop_tev_compensation must be an exact bool"
            )
        if self.le_open_deposit_drop_tev_compensation and not self.le_open_deposit:
            raise ValueError(
                "le_open_deposit_drop_tev_compensation requires le_open_deposit"
            )
        if type(self.particle_stretching) is not bool:
            raise TypeError("particle_stretching must be an exact bool")
        if type(self.particle_core_spreading) is not bool:
            raise TypeError("particle_core_spreading must be an exact bool")
        if self.particle_core_spreading and not self.particle_stretching:
            raise ValueError(
                "particle_core_spreading requires particle_stretching"
            )
        if type(self.particle_stretching_rvpm) is not bool:
            raise TypeError("particle_stretching_rvpm must be an exact bool")
        if self.particle_stretching_rvpm and not self.particle_stretching:
            raise ValueError(
                "particle_stretching_rvpm requires particle_stretching"
            )
        if type(self.particle_stretching_transposed) is not bool:
            raise TypeError("particle_stretching_transposed must be an exact bool")
        if self.le_open_deposit and not self.joint_separation_solve:
            raise ValueError(
                "le_open_deposit requires joint_separation_solve "
                "(the closure basis and the deposit must agree)"
            )
        if _TE_WAKE_V3 and self.joint_separation_solve:
            raise ValueError(
                "FLUXV_TE_WAKE=v3 is implemented on the legacy separation path only; "
                "joint_separation_solve writes the wake row with a different rule"
            )
        if _LEV_HIRATO and self.joint_separation_solve:
            raise ValueError(
                "FLUXV_LEV_MODEL=hirato solves the leading-edge vortex in the legacy "
                "separation path; joint_separation_solve is not available with it"
            )
        if _LEV_HIRATO_V2 and not _KERNEL_V3:
            raise ValueError(
                "FLUXV_LEV_MODEL=hirato_v2 needs FLUXV_RING_KERNEL=v3: the shared-edge "
                "cancellations of the sheet (defect B2b of note A24) only hold when the free "
                "rings and their unit influences are evaluated under one core radius"
            )
        if _LEV_HIRATO and not _TE_WAKE_V3:
            raise ValueError(
                "FLUXV_LEV_MODEL=hirato needs the implicit trailing-edge row: its "
                "trailing-edge wake row carries the bound circulation plus the shed one"
            )
        if type(self.load_derivative_active_set) is not bool:
            raise TypeError("load_derivative_active_set must be an exact bool")
        if type(self.te_leg_candidate_diagnostic) is not bool:
            raise TypeError("te_leg_candidate_diagnostic must be an exact bool")
        if type(self.flux_consistent_shedding) is not bool:
            raise TypeError("flux_consistent_shedding must be an exact bool")
        if type(self.le_constraint_window_steps) is not int:
            raise TypeError("le_constraint_window_steps must be an exact int")
        if self.le_constraint_window_steps < 0:
            raise ValueError("le_constraint_window_steps must be >= 0")
        if self.wake_history_mode not in LOAD_HISTORY_MODELS:
            raise ValueError("wake_history_mode must be 'material' or 'bound_rate'")
        if type(self.wake_free_rows) is not int or self.wake_free_rows < 0:
            raise ValueError("wake_free_rows must be a non-negative exact int")
        if self.lesp_crit != 0.11:
            raise ValueError("the Yamano native V5M path freezes Lcrit=0.11")
        if self.lesp_crit_override is not None:
            if not (
                math.isfinite(float(self.lesp_crit_override))
                and float(self.lesp_crit_override) > 0.0
            ):
                raise ValueError(
                    "lesp_crit_override must be finite and positive"
                )
            if (
                self.lesp_thickness_ratio is not None
                or self.lesp_reynolds is not None
            ):
                raise ValueError(
                    "lesp_crit_override is mutually exclusive with the "
                    "thickness/Re correlation inputs (lesp_thickness_ratio, "
                    "lesp_reynolds): one release-threshold source per config"
                )
        for name in ("density", "freestream", "aerodynamic_dt", "gate_rtol"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        angle_deg = float(self.freestream_angle_deg)
        if not math.isfinite(angle_deg) or abs(angle_deg) >= 90.0:
            raise ValueError("freestream_angle_deg must be finite with |angle| < 90")
        if torch.device(self.device).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("native FLUX-V5M requires CUDA")

    @property
    def effective_lesp_crit(self) -> float:
        """Return the release threshold actually used by the separation owner.

        Precedence: (1) an explicit documented per-case hypothesis
        (``lesp_crit_override`` — e.g. a static-polar Lcrit such as
        sin(CLmax/CLa)); (2) the thickness/Re correlation when both
        ``lesp_thickness_ratio`` and ``lesp_reynolds`` are provided; (3) the
        frozen flat-plate default 0.11.
        """
        if self.lesp_crit_override is not None:
            return float(self.lesp_crit_override)
        if self.lesp_thickness_ratio is not None and self.lesp_reynolds is not None:
            return compute_lesp_crit(
                self.lesp_thickness_ratio, self.lesp_reynolds)
        return self.lesp_crit


@dataclass(frozen=True, slots=True)
class NativeV5MGeometry:
    rings: torch.Tensor
    ring_velocity: torch.Tensor
    collocation: torch.Tensor
    collocation_velocity: torch.Tensor
    normals: torch.Tensor
    areas: torch.Tensor
    leading_edge: torch.Tensor
    trailing_edge: torch.Tensor
    leading_velocity: torch.Tensor
    trailing_velocity: torch.Tensor
    quarter_points: torch.Tensor


class Q16NativeV5MSurface:
    """Direct Q16 evaluation of the native V5M vortex lattice."""

    def __init__(
        self,
        mesh: Q16Mesh,
        *,
        q16_chordwise_elements: int,
        q16_spanwise_elements: int,
        aerodynamic_chordwise_panels: int,
        aerodynamic_spanwise_panels: int,
        device: str,
        dense_transfers: bool = False,
    ) -> None:
        if mesh.element_count != q16_chordwise_elements * q16_spanwise_elements:
            raise ValueError("Q16 element topology mismatch")
        # Dense-GEMV fast path for the fixed transfer maps (bit-consistent
        # with the kernel scatter to fp64 round-off; unblocks CUDA graphs).
        self.dense_transfers = bool(dense_transfers)
        self.mesh = mesh
        self.q16_chordwise_elements = q16_chordwise_elements
        self.q16_spanwise_elements = q16_spanwise_elements
        self.nc = aerodynamic_chordwise_panels
        self.ns = aerodynamic_spanwise_panels
        self.device = torch.device(device)
        quarter = tuple(
            ((i + 0.25) / self.nc, j / self.ns)
            for i in range(self.nc)
            for j in range(self.ns + 1)
        )
        leading = tuple((0.0, j / self.ns) for j in range(self.ns + 1))
        trailing = tuple((1.0, j / self.ns) for j in range(self.ns + 1))
        self.quarter_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, quarter),
            device=device,
        )
        self.leading_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, leading),
            device=device,
        )
        self.trailing_transfer = Q16CudaSurfaceTransfer(
            _surface_map(mesh, q16_chordwise_elements, q16_spanwise_elements, trailing),
            device=device,
        )

    def evaluate(self, state: wp.array, velocity: wp.array) -> NativeV5MGeometry:
        if self.dense_transfers:
            q_flat = wp.to_torch(state)[0]
            v_flat = wp.to_torch(velocity)[0]
            quarter = self.quarter_transfer.interpolate_dense(q_flat).reshape(
                self.nc, self.ns + 1, 3
            )
            quarter_velocity = self.quarter_transfer.interpolate_dense(
                v_flat
            ).reshape(self.nc, self.ns + 1, 3)
            leading = self.leading_transfer.interpolate_dense(q_flat)
            trailing = self.trailing_transfer.interpolate_dense(q_flat)
            leading_velocity = self.leading_transfer.interpolate_dense(v_flat)
            trailing_velocity = self.trailing_transfer.interpolate_dense(v_flat)
        else:
            quarter = wp.to_torch(self.quarter_transfer.interpolate(state))[0].reshape(
                self.nc, self.ns + 1, 3
            )
            quarter_velocity = wp.to_torch(
                self.quarter_transfer.interpolate(velocity)
            )[0].reshape(self.nc, self.ns + 1, 3)
            leading = wp.to_torch(self.leading_transfer.interpolate(state))[0]
            trailing = wp.to_torch(self.trailing_transfer.interpolate(state))[0]
            leading_velocity = wp.to_torch(self.leading_transfer.interpolate(velocity))[0]
            trailing_velocity = wp.to_torch(self.trailing_transfer.interpolate(velocity))[0]
        (
            rings,
            ring_velocity,
            collocation,
            collocation_velocity,
            normals,
            areas,
        ) = assemble_native_ring_geometry(
            quarter,
            quarter_velocity,
            trailing,
            trailing_velocity,
            panel_count=self.nc * self.ns,
        )
        for name, value in (
            ("rings", rings),
            ("collocation", collocation),
            ("collocation_velocity", collocation_velocity),
            ("normals", normals),
            ("areas", areas),
            ("leading_edge", leading),
            ("trailing_edge", trailing),
        ):
            _require_cuda64(name, value, self.device)
        return NativeV5MGeometry(
            rings=rings,
            ring_velocity=ring_velocity,
            collocation=collocation,
            collocation_velocity=collocation_velocity,
            normals=normals,
            areas=areas,
            leading_edge=leading,
            trailing_edge=trailing,
            leading_velocity=leading_velocity,
            trailing_velocity=trailing_velocity,
            quarter_points=quarter,
        )


def assemble_native_ring_geometry(
    quarter: torch.Tensor,
    quarter_velocity: torch.Tensor,
    trailing: torch.Tensor,
    trailing_velocity: torch.Tensor,
    *,
    panel_count: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Assemble native V5M rings/collocation/normals/areas from quarter lines.

    Single source of truth for the native V5M ring layout (HANDOFF
    IZRAELEVITZ2017 §8 H2): ``Q16NativeV5MSurface.evaluate`` and the rigid
    prescribed-motion reference grid both build panels through this helper,
    so the Q16-interpolated path and the SurfaceFrame path cannot drift apart.

    Inputs are ``(nc, ns+1, 3)`` quarter-chord lines with their velocities and
    ``(ns+1, 3)`` trailing-edge nodes.  The rear row sits one chordwise panel
    behind the last quarter line (the author's ``rear = quarter + 4/3 (TE -
    quarter)`` offset), so the bound sheet spans ``[0.25 dx, c + 0.25 dx]``
    in physical chord coordinates.  Returns ``(rings, ring_velocity,
    collocation, collocation_velocity, normals, areas)`` with rings ordered
    ``reshape(nc, ns, 4, 3)`` and panel index ``i * ns + j``.
    """
    rear = quarter[-1] + (4.0 / 3.0) * (trailing - quarter[-1])
    rear_velocity = quarter_velocity[-1] + (4.0 / 3.0) * (
        trailing_velocity - quarter_velocity[-1]
    )
    back = torch.cat((quarter[1:], rear.unsqueeze(0)), dim=0)
    back_velocity = torch.cat(
        (quarter_velocity[1:], rear_velocity.unsqueeze(0)), dim=0
    )
    rings = torch.stack(
        (quarter[:, :-1], quarter[:, 1:], back[:, 1:], back[:, :-1]), dim=2
    ).reshape(panel_count, 4, 3)
    ring_velocity = torch.stack(
        (
            quarter_velocity[:, :-1],
            quarter_velocity[:, 1:],
            back_velocity[:, 1:],
            back_velocity[:, :-1],
        ),
        dim=2,
    ).reshape(panel_count, 4, 3)
    collocation = torch.mean(rings, dim=1)
    collocation_velocity = torch.mean(ring_velocity, dim=1)
    diagonal_31 = rings[:, 3] - rings[:, 1]
    diagonal_24 = rings[:, 2] - rings[:, 0]
    cross = torch.linalg.cross(diagonal_31, diagonal_24, dim=1)
    cross_norm = torch.linalg.vector_norm(cross, dim=1)
    if bool(torch.any(cross_norm <= 0.0).item()):
        raise FloatingPointError("native V5M surface contains a collapsed panel")
    normals = cross / cross_norm[:, None]
    areas = 0.5 * cross_norm
    return rings, ring_velocity, collocation, collocation_velocity, normals, areas




def _read_device_scalars(
    floats: tuple[torch.Tensor, ...], ints: tuple[torch.Tensor, ...]
) -> tuple[list[float], list[int]]:
    """Read a batch of 0-dim device tensors with ONE host sync per dtype.

    The step diagnostics used to call ``.item()`` ~14 times per proposal; each call
    is a full stream synchronisation (measured 1,524 items and 1,270 syncs per
    aerodynamic step in the A16 profile).  Stacking the reductions and copying once
    transfers the SAME values, so every diagnostics number stays bit-identical --
    that is what makes this a lossless change rather than a numerics change.
    Ints are kept in their own stack so no dtype cast is involved either.
    """

    float_values = (
        torch.stack([value.reshape(()) for value in floats]).cpu().tolist()
        if floats
        else []
    )
    int_values = (
        [int(item) for item in torch.stack([value.reshape(()) for value in ints]).cpu().tolist()]
        if ints
        else []
    )
    return float_values, int_values


def native_aic(geometry: NativeV5MGeometry, *, chordwise_panels: int) -> torch.Tensor:
    if _KERNEL_V3:
        # Plan V3 (P0-A): the exact law.  The closest approach in this matrix is a
        # collocation point half a panel from its own ring's sides, so no core is needed;
        # the kernel's collinear guard is all the protection there is.  The legacy
        # reference length below, the wing's own chordwise panel length, is then not used.
        expanded = native_ring_velocity_expanded(
            geometry.collocation,
            geometry.rings,
            core_fraction=0.0,
            reference_length=0.0,
        )
    else:
        # Plan V3, Bug-4: the reference length of the legacy core is the wing's panel length,
        # not 1/chordwise_panels metres (the author's one-metre chord)
        expanded = native_ring_velocity_expanded(
            geometry.collocation,
            geometry.rings,
            core_fraction=1.0e-6,
            reference_length=mean_chordwise_panel(geometry.rings),
        )
    result = torch.sum(expanded * geometry.normals[:, None, :], dim=2)
    if bool(torch.any(torch.diagonal(result) >= 0.0).item()):
        raise RuntimeError("native AIC orientation drift")
    return result


@dataclass(slots=True)
class NativeV5MState:
    step: int
    gamma_bound: torch.Tensor
    gamma_previous: torch.Tensor
    wake_rings: torch.Tensor
    wake_gamma: torch.Tensor
    source_bank: CudaLDVMSourceBank
    particle_field: CudaParticleField
    reference_cell_chord: torch.Tensor
    reference_node_chord: torch.Tensor
    alpha_previous: torch.Tensor | None
    # Separation mask of this state's LAST step (n_strips,) bool — lets the
    # next propose() distinguish newly-separated strips from continuing ones
    # and separate real continuing LEV release from strips that never shed
    # (audit R3).  Like alpha_previous, it is history metadata, not part of
    # the numerical digest.
    separated_previous: torch.Tensor | None
    frontier_nodes: torch.Tensor
    frontier_active: torch.Tensor
    diagnostics: tuple[dict[str, Any], ...]
    # Leading-edge shedding ledger (history metadata, like separated_previous
    # -- NOT hashed by digest()):
    #   lev_strip_last_release -- per-strip strength (m^2/s, same measure as
    #     gamma_bound) deposited by the MOST RECENT release; this is the
    #     "lambda_first" of the discrete sheet and the only unambiguous
    #     per-strip LEV number the solver can offer (the per-particle
    #     `circul` array is an EDGE-level quantity repeated per particle and
    #     must never be summed per strip; see the 20261120 note).
    #   lev_strip_released_total -- running sum of every deposit, diagnostic.
    lev_strip_last_release: torch.Tensor | None = None
    lev_strip_released_total: torch.Tensor | None = None
    #: The leading-edge vortex sheet of every strip under FLUXV_LEV_MODEL=hirato
    #: (lev_hirato.LevChain: ring rows, strengths, rows in use, last step's strength);
    #: None on the legacy model.  Unlike the two ledgers above it IS numerical state.
    lev_chain: Any = None
    #: History metadata for synchronized t^{n-1} convection (Plan V4 Track A-1 / hirato_v2):
    #: bound vortex rings and Hirato pseudovortex committed at step n-1 so step n convects
    #: wake/LEV using consistent t^{n-1} geometry and strengths (NOT hashed by digest()).
    bound_rings_previous: torch.Tensor | None = None
    pseudovortex_previous: tuple[torch.Tensor, torch.Tensor] | None = None

    def clone(self) -> "NativeV5MState":
        return NativeV5MState(
            step=self.step,
            gamma_bound=self.gamma_bound.clone(),
            gamma_previous=self.gamma_previous.clone(),
            wake_rings=self.wake_rings.clone(),
            wake_gamma=self.wake_gamma.clone(),
            source_bank=_clone_source_bank(self.source_bank),
            particle_field=_clone_particle_field(self.particle_field),
            reference_cell_chord=self.reference_cell_chord.clone(),
            reference_node_chord=self.reference_node_chord.clone(),
            alpha_previous=None if self.alpha_previous is None else self.alpha_previous.clone(),
            separated_previous=(
                None
                if self.separated_previous is None
                else self.separated_previous.clone()
            ),
            frontier_nodes=self.frontier_nodes.clone(),
            frontier_active=self.frontier_active.clone(),
            lev_strip_last_release=(
                None
                if self.lev_strip_last_release is None
                else self.lev_strip_last_release.clone()
            ),
            lev_strip_released_total=(
                None
                if self.lev_strip_released_total is None
                else self.lev_strip_released_total.clone()
            ),
            lev_chain=None if self.lev_chain is None else self.lev_chain.clone(),
            bound_rings_previous=(
                None
                if self.bound_rings_previous is None
                else self.bound_rings_previous.clone()
            ),
            pseudovortex_previous=(
                None
                if self.pseudovortex_previous is None
                else (
                    self.pseudovortex_previous[0].clone(),
                    self.pseudovortex_previous[1].clone(),
                )
            ),
            diagnostics=tuple(dict(item) for item in self.diagnostics),
        )

    def digest(self) -> str:
        digest = hashlib.sha256(NATIVE_V5M_CONTRACT.encode("ascii"))
        digest.update(str(self.step).encode("ascii"))
        tensors = [
            self.gamma_bound,
            self.gamma_previous,
            self.wake_rings,
            self.wake_gamma,
            self.particle_field.positions_cuda,
            self.particle_field.gammas_cuda,
            self.frontier_nodes,
            self.frontier_active,
        ]
        if self.lev_chain is not None:
            tensors.extend(self.lev_chain.tensors())
        for value in tensors:
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        digest.update(str(self.source_bank.it).encode("ascii"))
        digest.update(str(self.source_bank.nt).encode("ascii"))
        return digest.hexdigest()

    def mutation_fingerprint(self) -> tuple:
        """Cheap "was anything written in place?" token; NO device sync.

        The three transaction guards below only need to know whether a state was
        mutated, which is exactly what a full-state ``digest()`` was being used
        for -- at the cost of 8 blocking D2H copies and a CPU SHA-256 over ~2.25 MB
        per call, several times per aerodynamic step.  Every in-place torch op
        bumps ``Tensor._version``, and a rebind or a changed live count changes
        ``data_ptr``/``shape``, so this catches the same events on the host.

        ``digest()`` is unchanged and remains the value recorded in the ledger.
        """

        tensors = [
            self.gamma_bound,
            self.gamma_previous,
            self.wake_rings,
            self.wake_gamma,
            self.particle_field.positions_cuda,
            self.particle_field.gammas_cuda,
            self.frontier_nodes,
            self.frontier_active,
        ]
        if self.lev_chain is not None:
            # the leading-edge ring chains are replaced, never written in place, so a
            # changed pointer is what a shed step leaves on the trial; the parent keeps its own
            tensors.extend(self.lev_chain.tensors())
        return (
            self.step,
            int(self.source_bank.it),
            int(self.source_bank.nt),
            tuple(
                (value.data_ptr(), value._version, tuple(value.shape))
                for value in tensors
            ),
        )


@dataclass(frozen=True, slots=True)
class NativeV5MLoad:
    panel_positions: torch.Tensor
    panel_forces: torch.Tensor
    pressure: torch.Tensor
    total_force: torch.Tensor
    total_moment: torch.Tensor
    # M1 unified owner: resolved 5-point-per-panel load every consumer
    # projects.  None on legacy/compat paths only; the formal rigid and Q16
    # proposals always publish it.
    surface_load_packet: Any = None


@dataclass(frozen=True, slots=True)
class NativeV5MProposal:
    parent_digest: str
    parent_fingerprint: tuple
    trial_state: NativeV5MState
    load: NativeV5MLoad
    generalized_force: wp.array
    author_load: Any
    #: S4a/S4b: the state right AFTER the material convection and BEFORE the deposit, so
    #: a later propose() can reuse the convected free wake/particles without inheriting
    #: this proposal's deposit (re-depositing would double the particle count).
    preconvected_state: "NativeV5MState | None" = None
    #: The cull accounting of the convection this state came from.  A reusing proposal
    #: reports THESE numbers instead of zeros: the particles really were removed, and a
    #: ledger that reports zero would break the in-repo rule that a cap destroying
    #: circulation must never be silent.  Only the committed proposal's record reaches
    #: the payload, so inheriting cannot double count.
    preconvected_retention: dict | None = None


class Q16NativePanelLoadTransfer:
    """Exact transpose of the author collocation construction into Q16."""

    def __init__(self, surface: Q16NativeV5MSurface) -> None:
        self.surface = surface

    def map(self, panel_forces: torch.Tensor) -> wp.array:
        if panel_forces.ndim != 2:
            raise ValueError("native panel force must have shape (panel,3)")
        return self.map_batch(panel_forces.unsqueeze(0))

    def map_batch(self, panel_forces: torch.Tensor) -> wp.array:
        nc, ns = self.surface.nc, self.surface.ns
        if panel_forces.ndim != 3 or panel_forces.shape[1:] != (nc * ns, 3):
            raise ValueError(
                "panel force batch shape differs from native V5M topology"
            )
        batch = panel_forces.shape[0]
        forces = panel_forces.reshape(batch, nc, ns, 3)
        quarter_force = torch.zeros(
            (batch, nc, ns + 1, 3),
            device=panel_forces.device,
            dtype=torch.float64,
        )
        trailing_force = torch.zeros(
            (batch, ns + 1, 3),
            device=panel_forces.device,
            dtype=torch.float64,
        )
        quarter_force[:, :, :-1] += 0.25 * forces
        quarter_force[:, :, 1:] += 0.25 * forces
        if nc > 1:
            quarter_force[:, 1:, :-1] += 0.25 * forces[:, :-1]
            quarter_force[:, 1:, 1:] += 0.25 * forces[:, :-1]
        quarter_force[:, -1, :-1] -= (1.0 / 12.0) * forces[:, -1]
        quarter_force[:, -1, 1:] -= (1.0 / 12.0) * forces[:, -1]
        trailing_force[:, :-1] += (1.0 / 3.0) * forces[:, -1]
        trailing_force[:, 1:] += (1.0 / 3.0) * forces[:, -1]
        quarter_wp = wp.from_torch(
            quarter_force.reshape(batch, nc * (ns + 1), 3),
            dtype=warp_config.VEC3,
        )
        trailing_wp = wp.from_torch(trailing_force, dtype=warp_config.VEC3)
        q_generalized = self.surface.quarter_transfer.transpose(quarter_wp)
        t_generalized = self.surface.trailing_transfer.transpose(trailing_wp)
        wp.to_torch(q_generalized).add_(wp.to_torch(t_generalized))
        return q_generalized


# Set True only around an active wp.ScopedCapture region: host-side
# validation gates must defer during CUDA-graph capture; the capture-mode
# caller validates the replayed static outputs after the capture ends.
CAPTURING = False


def freestream_vector(
    freestream: float, angle_deg: float, device: str | torch.device
) -> torch.Tensor:
    """v_inf = U (cos a, 0, sin a) in the scientific frame (+x chord, +z up)."""

    angle_rad = math.radians(angle_deg)
    return float(freestream) * torch.tensor(
        [math.cos(angle_rad), 0.0, math.sin(angle_rad)],
        device=torch.device(device),
        dtype=torch.float64,
    )


class Q16NativeV5MSolver:
    """State-complete native V5M stepper with transactional trial ownership."""

    def __init__(self, surface: Q16NativeV5MSurface, settings: NativeV5MConfig) -> None:
        if surface.nc != settings.chordwise_panels or surface.ns != settings.spanwise_panels:
            raise ValueError("surface and aerodynamic topology differ")
        self.surface = surface
        self.settings = settings
        self.device = torch.device(settings.device)
        self.load_transfer = Q16NativePanelLoadTransfer(surface)
        surface.panel_load_transfer = self.load_transfer
        from .q16_flux_v5m_author_loads import Q16NativeAuthorLoadAssembler

        self._resolved_load_assembler = None
        self._last_resolved_vs_quadrature_relative = float("nan")
        # Reference length of the legacy core for every `_ring_velocity` call (plan V3, Bug-4):
        # the chordwise panel length of the current wing, refreshed at the top of propose().
        # The value here only covers calls before the first propose().
        self._legacy_reference_length = 1.0 / settings.chordwise_panels
        # per-step fields the load evaluations read (plan V3 stage 2): the leading-edge ring
        # chains with the pseudovortex rings as (rings, strengths), and the leading-edge
        # suction as a force per panel
        self._lev_field = None
        self._lev_pseudovortex = None
        self._lev_bound_extra = None
        self._step_suction_panel = None
        self._step_suction_replace_mask = None
        # the pseudovortex rings of the step that has been committed: the repaired sheet puts
        # them in the right-hand side and in the convection (note A24, B1/B2)
        self._lev_pseudovortex_previous = None
        # which strips' sheets are detached (note A25): the next convection freezes their rings
        self._lev_detached = None
        self.author_load_assembler = Q16NativeAuthorLoadAssembler(
            surface, density=settings.density
        )
        self.v_inf = freestream_vector(
            settings.freestream, settings.freestream_angle_deg, self.device
        )

    def initialize(self, state: wp.array, velocity: wp.array) -> NativeV5MState:
        geometry = self.surface.evaluate(state, velocity)
        ns, count = self.settings.spanwise_panels, self.settings.chordwise_panels * self.settings.spanwise_panels
        node_chord = torch.linalg.vector_norm(
            geometry.trailing_edge - geometry.leading_edge, dim=1
        )
        cell_chord = 0.5 * (node_chord[:-1] + node_chord[1:])
        lane_chord = torch.cat((cell_chord, node_chord))
        convective_dt = self.settings.aerodynamic_dt * self.settings.freestream / lane_chord
        source = CudaLDVMSourceBank(
            batch_size=2 * ns + 1,
            ndiv=self.settings.dvm_ndiv,
            naterm=self.settings.dvm_naterm,
            delta_time_convective=convective_dt,
            lesp_crit=self.settings.effective_lesp_crit,
            core_radius_chord=self.settings.dvm_core_radius_chord,
            max_wake=self.settings.dvm_max_wake,
            source_parity=True,
            device=self.settings.device,
        )
        return NativeV5MState(
            step=0,
            gamma_bound=torch.zeros(count, device=self.device, dtype=torch.float64),
            gamma_previous=torch.zeros(count, device=self.device, dtype=torch.float64),
            wake_rings=torch.zeros((0, 4, 3), device=self.device, dtype=torch.float64),
            wake_gamma=torch.zeros(0, device=self.device, dtype=torch.float64),
            source_bank=source,
            particle_field=CudaParticleField(
                self.settings.particle_capacity, device=self.settings.device
            ),
            reference_cell_chord=cell_chord,
            reference_node_chord=node_chord,
            alpha_previous=None,
            separated_previous=torch.zeros(
                ns, device=self.device, dtype=torch.bool
            ),
            lev_strip_last_release=torch.zeros(
                ns, device=self.device, dtype=torch.float64
            ),
            lev_strip_released_total=torch.zeros(
                ns, device=self.device, dtype=torch.float64
            ),
            lev_chain=(
                lev_hirato.LevChain.empty(ns, self.device, rows=lev_hirato.max_rows())
                if _LEV_HIRATO
                else None
            ),
            frontier_nodes=geometry.leading_edge.clone(),
            frontier_active=torch.zeros(ns + 1, device=self.device, dtype=torch.bool),
            diagnostics=(),
        )

    def author_anchor_load(
        self, state: wp.array, velocity: wp.array
    ) -> Any:
        geometry = self.surface.evaluate(state, velocity)
        aic = native_aic(
            geometry, chordwise_panels=self.settings.chordwise_panels
        )
        panel_count = self.settings.chordwise_panels * self.settings.spanwise_panels
        zero = torch.zeros(panel_count, device=self.device, dtype=torch.float64)
        zero_gradient = torch.zeros(
            (panel_count, 3), device=self.device, dtype=torch.float64
        )
        return self.author_load_assembler.assemble(
            structural_state=state,
            geometry=geometry,
            aic=aic,
            gamma=zero,
            external_flow=self.v_inf[None, :].expand(panel_count, 3),
            gamma_gradient=zero_gradient,
            mf2_history=zero,
        )

    # Tile budget for the ring-influence sum: (points x rings x 3) float64.
    # Evaluations below the budget take the historical single-shot path
    # (bit-identical for the oracle-verified small cases); larger ones tile
    # over the points so long free wakes cannot materialize multi-GB
    # intermediate tensors.
    _RING_TILE_BYTES = 64 * 1024 * 1024

    def _ring_velocity(
        self,
        points: torch.Tensor,
        rings: torch.Tensor,
        gamma: torch.Tensor,
        *,
        rough: bool,
        core_radius: float | None = None,
        free: bool = False,
        mixed_precision: bool = False,
    ) -> torch.Tensor:
        """Velocity of `rings` at `points`.  `free=True` marks a free-wake evaluation.

        Under the legacy kernel a free-wake evaluation uses the physical core
        (`FLUXV_FREE_WAKE_CORE_FRACTION`, see the note next to it); under the v3 kernel the
        caller passes `core_radius` instead and `free` does nothing.
        """

        if rings.shape[0] == 0:
            return torch.zeros_like(points)
        if _KERNEL_V3:
            # Plan V3 (P0-A): the exact law, or the fourth-order core of one fixed radius
            # where the caller asks for it (the free wake close to other filaments).
            # `rough` and the legacy reference length do not apply on this path.
            keywords = {
                "core_fraction": 0.0,
                "reference_length": 0.0,
                "core_radius": core_radius,
            }
            if (
                mixed_precision
                and points.device.type == "cuda"
                and points.shape[0] * rings.shape[0] >= 4096
            ):
                return native_ring_velocity_fused(
                    points, rings, gamma, **keywords, mixed_precision=True
                )
            full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
            if full_bytes <= self._RING_TILE_BYTES:
                expanded = native_ring_velocity_expanded(points, rings, **keywords)
                return torch.sum(expanded * gamma[None, :, None], dim=1)
            return native_ring_velocity_fused(points, rings, gamma, **keywords)
        if rough:
            core_fraction = 0.1
        elif free and _FREE_WAKE_CORE_STEPS > 0.0:
            # the step-scaled core of the papers, expressed in the legacy kernel's own terms:
            # its radius is core_fraction * max(longest edge of the ring, reference_length), so
            # this gives that radius for a ring no longer than one panel (see the flag's note)
            core_fraction = (
                _FREE_WAKE_CORE_STEPS
                * float(self.settings.freestream * self.settings.aerodynamic_dt)
                / self._legacy_reference_length
            )
        elif free and _FREE_WAKE_CORE_FRACTION > 0.0:
            core_fraction = _FREE_WAKE_CORE_FRACTION
        else:
            core_fraction = 1.0e-6
        # the wing's own panel length, set at the top of propose() (plan V3, Bug-4)
        reference_length = self._legacy_reference_length
        full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
        if full_bytes <= self._RING_TILE_BYTES:
            expanded = native_ring_velocity_expanded(
                points,
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            return torch.sum(expanded * gamma[None, :, None], dim=1)
        # Large systems (long free wakes): the tiled torch path materializes
        # ~15 (tile, rings, 3) intermediates per call and runs ~30x off the
        # FP64 roofline.  The fused warp kernel accumulates per source block
        # in registers (same frozen core model; fp64 summation-order
        # difference only, mirroring the fused particle kernel).
        return native_ring_velocity_fused(
            points,
            rings,
            gamma,
            core_fraction=core_fraction,
            reference_length=reference_length,
        )

    def _core_radius(self, geometry: NativeV5MGeometry) -> float | None:
        """Core radius of the v3 kernel, or None where no core applies.

        FLUXV_WAKE_CORE_PANELS chordwise panel lengths, and only under the v3 kernel.  It is
        tied to the panel size so that it scales with the wing; the legacy kernel had no
        physical core at all (core_fraction 1e-6 times a reference length in metres) and
        leaned on the 1e-9 constant instead.

        It is applied wherever FREE material points are convected (wake nodes, particles),
        to every filament they see, and to the wake filaments at the load evaluation; the
        influence matrix stays exact.  The bound sheet needs it at the trailing-edge
        junction: the upstream corners of the newest wake row lie ON the lines of the
        bound trailing-edge and streamwise legs.  The collinear guard covers a point that
        is exactly on such a line; one Runge-Kutta stage later the point is a distance h
        off it, and the exact law then returns (Gamma_j - Gamma_j+1) / (4 pi h) from the
        two anti-parallel streamwise legs that meet there.  Measured on the rigid plate at
        alpha = 5 deg with the core on the wake only: a normal velocity of about 1.4 m/s
        at a freestream of 0.022 m/s (third step of the startup ramp).
        """

        if not _KERNEL_V3:
            return None
        if _FREE_WAKE_CORE_STEPS > 0.0:
            # the core of the papers that state one: a fraction of one convection step
            # (Li et al.: r_core = U dt; Hirato et al.: below 0.5 of the smallest ring dimension)
            return _FREE_WAKE_CORE_STEPS * float(
                self.settings.freestream * self.settings.aerodynamic_dt
            )
        if _WAKE_CORE_PANELS <= 0.0:
            return None
        panel = 0.5 * torch.linalg.vector_norm(
            geometry.rings[:, 2] + geometry.rings[:, 3]
            - geometry.rings[:, 0] - geometry.rings[:, 1],
            dim=1,
        )
        return _WAKE_CORE_PANELS * float(torch.mean(panel).item())

    def _lev_rhs_core_radius(self, geometry: NativeV5MGeometry) -> float | None:
        """Uniform core radius for LEV chain + pseudovortex on RHS, LHS basis, and loads."""

        if not _KERNEL_V3 or _LEV_CORE_RHS_FRACTION <= 0.0:
            return None
        core = self._core_radius(geometry)
        return None if core is None else core * _LEV_CORE_RHS_FRACTION

    def _lev_live_core_radii(
        self,
        chain,
        base_core: float | None,
        *,
        row_age_offset: int = 0,
        continuing_mask: torch.Tensor | None = None,
    ) -> float | torch.Tensor | None:
        """Per-ring Squire turbulent core radii for live LEV rings (or scalar `base_core`).

        r_c(m, j) = sqrt(r_0^2 + _SQUIRE_CORE * |Gamma(m, j)| * age_steps * dt).
        Using `row_age_offset=1` (on continuing strips) before commit matches `row_age_offset=0`
        after `push_nascent` bit-for-bit, preserving exact ~1e-14 post-commit Neumann parity.
        """

        if base_core is None or _SQUIRE_CORE <= 0.0 or not _KERNEL_V3:
            return base_core
        rows, ns = chain.rings.shape[0], chain.rings.shape[1]
        row_idx = torch.arange(rows, device=self.device, dtype=torch.float64)[:, None]
        if row_age_offset and continuing_mask is not None:
            offset = continuing_mask.to(torch.float64)[None, :] * float(row_age_offset)
        else:
            offset = float(row_age_offset)
        age_s = (row_idx + offset) * float(self.settings.aerodynamic_dt)
        r0_sq = float(base_core) * float(base_core)
        rc_grid = torch.sqrt(r0_sq + _SQUIRE_CORE * torch.abs(chain.gamma) * age_s)
        mask = lev_hirato.active_rows(chain)
        return rc_grid.reshape(-1)[mask]

    def _wake_core_radii(
        self,
        wake_gamma: torch.Tensor,
        base_core: float | None,
        *,
        row_age_offset: int = 1,
    ) -> float | torch.Tensor | None:
        """Per-ring Squire turbulent core radii for free TE wake rings (or scalar `base_core`)."""

        if base_core is None or _SQUIRE_CORE <= 0.0 or not _KERNEL_V3 or wake_gamma.numel() == 0:
            return base_core
        ns = self.settings.spanwise_panels
        total_rows = wake_gamma.shape[0] // ns
        row_idx = (
            torch.arange(total_rows, device=self.device, dtype=torch.float64) + float(row_age_offset)
        )[:, None].expand(total_rows, ns).reshape(-1)
        age_s = row_idx * float(self.settings.aerodynamic_dt)
        r0_sq = float(base_core) * float(base_core)
        return torch.sqrt(r0_sq + _SQUIRE_CORE * torch.abs(wake_gamma) * age_s)

    def _wake_collocation_core_radius(self, geometry: NativeV5MGeometry) -> float | None:
        """Optional uniform core radius for free TE wake evaluated at wing collocation points."""

        if not _KERNEL_V3 or _WAKE_CORE_AT_COLLOCATION <= 0.0:
            return None
        core = self._core_radius(geometry)
        return None if core is None else core * _WAKE_CORE_AT_COLLOCATION

    def _ring_unit_normal(
        self,
        geometry: NativeV5MGeometry,
        rings: torch.Tensor,
        *,
        core_radius: float | None = None,
        free: bool = False,
    ) -> torch.Tensor:
        """(P, M) normal velocity induced per unit strength by `rings` at `geometry.collocation`."""

        if _KERNEL_V3:
            expanded = native_ring_velocity_expanded(
                geometry.collocation,
                rings,
                core_fraction=0.0,
                reference_length=0.0,
                core_radius=core_radius,
            )
        else:
            if free and _FREE_WAKE_CORE_STEPS > 0.0:
                core_fraction = (
                    _FREE_WAKE_CORE_STEPS
                    * float(self.settings.freestream * self.settings.aerodynamic_dt)
                    / self._legacy_reference_length
                )
            elif free and _FREE_WAKE_CORE_FRACTION > 0.0:
                core_fraction = _FREE_WAKE_CORE_FRACTION
            else:
                core_fraction = 1.0e-6
            expanded = native_ring_velocity_expanded(
                geometry.collocation,
                rings,
                core_fraction=core_fraction,
                reference_length=self._legacy_reference_length,
            )
        return torch.sum(expanded * geometry.normals[:, None, :], dim=2)

    def _external_velocity(
        self,
        points: torch.Tensor,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
        core_radius: float | None = None,
    ) -> torch.Tensor:
        # free material points: under the legacy kernel every filament they see carries the
        # free-wake core (FLUXV_FREE_WAKE_CORE_FRACTION), bound sheet included
        bound_velocity = self._ring_velocity(
            points, geometry.rings, trial.gamma_bound, rough=False, core_radius=core_radius,
            free=True,
        )
        wake_velocity = self._ring_velocity(
            points, trial.wake_rings, trial.wake_gamma, rough=False, core_radius=core_radius,
            free=True,
        )
        return self.v_inf[None, :] + bound_velocity + wake_velocity

    def _lev_chain_velocity(
        self, points: torch.Tensor, chain, core_radius: float | None = None
    ) -> torch.Tensor:
        """Induction of the leading-edge ring chains at `points` (skips zero rows)."""

        rings, gamma = lev_hirato.flat_live(chain)
        return self._ring_velocity(
            points, rings, gamma, rough=False, core_radius=core_radius, free=True
        )

    def _advance_material_state_v4(
        self,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
        dt: float,
        core: float | None,
        retention: dict[str, float | int],
    ) -> dict[str, float | int]:
        """Synchronized t^{n-1} Heun (RK2) material convection (Plan V4 Track A-1 + hirato_v2).

        Fixes four defects of the legacy sequential convection simultaneously:
          1. Uses `bound_rings_previous` paired with `gamma_bound` (consistent t^{n-1} time level).
          2. Evaluates ONE unified velocity field (v_inf + bound + wake + live LEV + previous
             pseudovortex + particles) with uniform `core_radius=core`, so TE wake and LEV see
             each other with exact leading/trailing-edge cancellation and zero singularity.
          3. Deduplicates connected structured wake corners into (R_free+1)*(ns+1) unique vertices
             (3.9x fewer targets) and compacts LEV chains to live rows only (up to 99% fewer).
          4. Updates stage-2 wake/LEV/particle source positions (non-frozen RK2 predictor-corrector).
        """

        ns = self.settings.spanwise_panels
        nc = self.settings.chordwise_panels
        has_wake = trial.wake_rings.shape[0] > 0
        chain = trial.lev_chain
        has_lev = chain is not None and bool((chain.count > 0).any().item())
        n_part = int(trial.particle_field.n)
        if not (has_wake or has_lev or n_part):
            return retention

        bound_src = (
            trial.bound_rings_previous
            if trial.bound_rings_previous is not None
            else geometry.rings
        )
        bound_gamma = trial.gamma_bound
        bound_last = bound_src.reshape(nc, ns, 4, 3)[-1]

        pv_src = (
            trial.pseudovortex_previous
            if trial.pseudovortex_previous is not None
            else getattr(self, "_lev_pseudovortex_previous", None)
        )
        has_pv = (
            chain is not None
            and pv_src is not None
            and pv_src[0].shape[0] > 0
            and bool((chain.prev != 0.0).any().item())
        )
        pv_rings = pv_src[0] if has_pv else None
        pv_gamma = chain.prev if has_pv else None

        free_rows = 0
        use_unique_wake = False
        wake_frozen_0 = None
        wake_pts_0 = None
        if has_wake:
            total_rows = trial.wake_rings.shape[0] // ns
            freeze_rows = self.settings.wake_free_rows
            free_rows = (
                min(total_rows, freeze_rows)
                if (freeze_rows and total_rows > freeze_rows)
                else total_rows
            )
            wake_all_0 = trial.wake_rings.clone()
            wake_active_0 = wake_all_0[: free_rows * ns]
            wake_frozen_0 = wake_all_0[free_rows * ns :]
            if _SHED_DISTANCE_FRACTION == 0.0:
                w_grid = wake_active_0.reshape(free_rows, ns, 4, 3)
                verts_0 = torch.empty(
                    (free_rows + 1, ns + 1, 3), device=self.device, dtype=torch.float64
                )
                verts_0[0, :-1] = w_grid[0, :, 0]
                verts_0[0, -1] = w_grid[0, -1, 1]
                verts_0[1:, :-1] = w_grid[:, :, 3]
                verts_0[1:, -1] = w_grid[:, -1, 2]
                recon_0 = torch.stack(
                    (verts_0[:-1, :-1], verts_0[:-1, 1:], verts_0[1:, 1:], verts_0[1:, :-1]),
                    dim=2,
                ).reshape(free_rows * ns, 4, 3)
                if bool((torch.max(torch.abs(recon_0 - wake_active_0)) <= 1.0e-12).item()):
                    use_unique_wake = True
                    wake_pts_0 = verts_0.reshape(-1, 3)
            if not use_unique_wake:
                wake_pts_0 = wake_active_0.reshape(-1, 3)

        live_mask = None
        lev_gamma = None
        lev_pts_0 = None
        cont_mask = None
        if has_lev:
            live_mask = lev_hirato.active_rows(chain)
            lev_rings_0 = chain.rings.reshape(-1, 4, 3)[live_mask].clone()
            lev_gamma = chain.gamma.reshape(-1)[live_mask]
            lev_pts_0 = lev_rings_0.reshape(-1, 3)
            cont_mask = lev_hirato.continuing(chain)

        part_pts_0 = trial.particle_field.pos[:n_part].clone() if n_part else None

        def _reconstruct_active_wake(w_pts: torch.Tensor) -> torch.Tensor:
            if use_unique_wake:
                v = w_pts.reshape(free_rows + 1, ns + 1, 3)
                return torch.stack(
                    (v[:-1, :-1], v[:-1, 1:], v[1:, 1:], v[1:, :-1]), dim=2
                ).reshape(free_rows * ns, 4, 3)
            return w_pts.reshape(free_rows * ns, 4, 3)

        def _eval_stage(
            w_pts: torch.Tensor | None,
            l_pts: torch.Tensor | None,
            p_pts: torch.Tensor | None,
            stage_dt: float,
        ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
            src_rings_list = [bound_src]
            src_gamma_list = [bound_gamma]
            use_squire = _SQUIRE_CORE > 0.0 and _KERNEL_V3 and core is not None
            src_core_list = (
                [torch.full((bound_src.shape[0],), float(core), device=self.device, dtype=torch.float64)]
                if use_squire
                else None
            )
            if has_wake and w_pts is not None:
                w_act = _reconstruct_active_wake(w_pts)
                if wake_frozen_0 is not None and wake_frozen_0.shape[0] > 0:
                    w_src = torch.cat((w_act, wake_frozen_0 + stage_dt * self.v_inf), dim=0)
                else:
                    w_src = w_act
                src_rings_list.append(w_src)
                src_gamma_list.append(trial.wake_gamma)
                if use_squire:
                    src_core_list.append(self._wake_core_radii(trial.wake_gamma, core, row_age_offset=1))
                if stage_dt > 0.0 and _CONVECTION_V4_STAGE2_BRIDGE:
                    w_front = w_act[:ns]
                    te_bridge = torch.stack(
                        (bound_last[:, 3], bound_last[:, 2], w_front[:, 1], w_front[:, 0]),
                        dim=1,
                    )
                    src_rings_list.append(te_bridge)
                    src_gamma_list.append(trial.wake_gamma[:ns])
                    if use_squire:
                        src_core_list.append(
                            torch.full((ns,), float(core), device=self.device, dtype=torch.float64)
                        )
            if has_lev and l_pts is not None:
                l_live = l_pts.reshape(-1, 4, 3)
                src_rings_list.append(l_live)
                src_gamma_list.append(lev_gamma)
                if use_squire:
                    src_core_list.append(self._lev_live_core_radii(chain, core, row_age_offset=1))
                if (
                    stage_dt > 0.0
                    and _CONVECTION_V4_STAGE2_BRIDGE
                    and has_pv
                    and cont_mask is not None
                    and bool(cont_mask.any().item())
                ):
                    rows = chain.rings.shape[0]
                    l_full = torch.zeros_like(chain.rings).reshape(-1, 4, 3)
                    l_full[live_mask] = l_live
                    l0 = l_full.reshape(rows, ns, 4, 3)[0, cont_mask]
                    pv_c = pv_rings[cont_mask]
                    le_bridge = torch.stack(
                        (l0[:, 3], l0[:, 2], pv_c[:, 1], pv_c[:, 0]),
                        dim=1,
                    )
                    src_rings_list.append(le_bridge)
                    src_gamma_list.append(pv_gamma[cont_mask])
                    if use_squire:
                        src_core_list.append(
                            torch.full(
                                (le_bridge.shape[0],),
                                float(core),
                                device=self.device,
                                dtype=torch.float64,
                            )
                        )
            if has_pv:
                src_rings_list.append(pv_rings)
                src_gamma_list.append(pv_gamma)
                if use_squire:
                    src_core_list.append(
                        torch.full((pv_rings.shape[0],), float(core), device=self.device, dtype=torch.float64)
                    )
            src_rings = torch.cat(src_rings_list, dim=0)
            src_gamma = torch.cat(src_gamma_list, dim=0)
            src_cores = torch.cat(src_core_list, dim=0) if use_squire else core

            target_parts = []
            n_w = w_pts.shape[0] if w_pts is not None else 0
            n_l = l_pts.shape[0] if l_pts is not None else 0
            n_p = p_pts.shape[0] if p_pts is not None else 0
            if n_w:
                target_parts.append(w_pts)
            if n_l:
                target_parts.append(l_pts)
            if n_p:
                target_parts.append(p_pts)
            targets = (
                target_parts[0]
                if len(target_parts) == 1
                else torch.cat(target_parts, dim=0)
            )
            vel = self.v_inf[None, :] + self._ring_velocity(
                targets,
                src_rings,
                src_gamma,
                rough=False,
                core_radius=src_cores,
                free=True,
                mixed_precision=_RING_MIXED_PRECISION,
            )
            if n_part:
                if stage_dt > 0.0 and p_pts is not None:
                    trial.particle_field.pos[:n_part].copy_(p_pts)
                vel = vel + trial.particle_field.velocity_at_cuda(targets)
            offset = 0
            u_w = vel[offset : offset + n_w] if n_w else None
            offset += n_w
            u_l = vel[offset : offset + n_l] if n_l else None
            offset += n_l
            u_p = vel[offset : offset + n_p] if n_p else None
            return u_w, u_l, u_p

        u1_w, u1_l, u1_p = _eval_stage(wake_pts_0, lev_pts_0, part_pts_0, 0.0)
        w_pred = None if wake_pts_0 is None else wake_pts_0 + dt * u1_w
        l_pred = None if lev_pts_0 is None else lev_pts_0 + dt * u1_l
        p_pred = None if part_pts_0 is None else part_pts_0 + dt * u1_p
        if _WING_SLIP_WALL > 0.0:
            if w_pred is not None:
                w_pred = self._apply_wing_slip_wall(geometry, bound_src, wake_pts_0, w_pred)
            if l_pred is not None:
                l_pred = self._apply_wing_slip_wall(geometry, bound_src, lev_pts_0, l_pred)
            if p_pred is not None:
                p_pred = self._apply_wing_slip_wall(geometry, bound_src, part_pts_0, p_pred)
        u2_w, u2_l, u2_p = _eval_stage(w_pred, l_pred, p_pred, dt)

        if has_wake and wake_pts_0 is not None:
            w_new_pts = wake_pts_0 + 0.5 * dt * (u1_w + u2_w)
            if not bool(torch.isfinite(w_new_pts).all().item()):
                raise FloatingPointError("wake rings convected to a non-finite position")
            if _WING_SLIP_WALL > 0.0:
                w_new_pts = self._apply_wing_slip_wall(geometry, bound_src, wake_pts_0, w_new_pts)
            updated_active = _reconstruct_active_wake(w_new_pts)
            if wake_frozen_0 is not None and wake_frozen_0.shape[0] > 0:
                trial.wake_rings = torch.cat(
                    (updated_active, wake_frozen_0 + dt * self.v_inf), dim=0
                )
            else:
                trial.wake_rings = updated_active

        if has_lev and lev_pts_0 is not None:
            l_new_pts = lev_pts_0 + 0.5 * dt * (u1_l + u2_l)
            if not bool(torch.isfinite(l_new_pts).all().item()):
                raise FloatingPointError(
                    "leading-edge vortex rings convected to a non-finite position"
                )
            if _WING_SLIP_WALL > 0.0:
                l_new_pts = self._apply_wing_slip_wall(geometry, bound_src, lev_pts_0, l_new_pts)
            rows = chain.rings.shape[0]
            rings_full = torch.zeros_like(chain.rings).reshape(-1, 4, 3)
            rings_full[live_mask] = l_new_pts.reshape(-1, 4, 3)
            moved_grid = rings_full.reshape(rows, ns, 4, 3)
            if _LEV_DETACH and getattr(self, "_lev_detached", None) is not None:
                mask = self._lev_detached.reshape(1, ns, 1, 1)
                moved_grid = torch.where(
                    mask, chain.rings + dt * self.v_inf[None, None, None, :], moved_grid
                )
            trial.lev_chain = lev_hirato._mask_invalid(
                lev_hirato.reconnect(chain.with_rings(moved_grid)), chain.count
            )

        if n_part and part_pts_0 is not None:
            p_new = part_pts_0 + 0.5 * dt * (u1_p + u2_p)
            if not bool(torch.isfinite(p_new).all().item()):
                raise FloatingPointError("vortex particles convected to a non-finite position")
            if _WING_SLIP_WALL > 0.0:
                p_new = self._apply_wing_slip_wall(geometry, bound_src, part_pts_0, p_new)
            trial.particle_field.pos[:n_part].copy_(p_new)
            if self.settings.particle_max_age_steps > 0:
                horizon = trial.step - self.settings.particle_max_age_steps
                if horizon > 0:
                    alive = trial.particle_field.n
                    stale = trial.particle_field.birth_step[:alive] < horizon
                    removed = int(torch.count_nonzero(stale).item())
                    if removed:
                        retention["particle_cull_count"] = removed
                        retention["particle_cull_circulation"] = float(
                            trial.particle_field.circul[:alive][stale].sum().item()
                        )
                        impulse = 0.5 * torch.cross(
                            trial.particle_field.pos[:alive][stale],
                            trial.particle_field.gamma[:alive][stale],
                            dim=1,
                        ).sum(dim=0)
                        retention["particle_cull_linear_impulse"] = [
                            float(value) for value in impulse.tolist()
                        ]
                        trial.particle_field.remove_mask(
                            trial.particle_field.birth_step[: trial.particle_field.n]
                            < horizon
                        )
        return retention

    def _apply_wing_slip_wall(
        self,
        geometry: NativeV5MGeometry,
        bound_rings_prev: torch.Tensor,
        pts_old: torch.Tensor,
        pts_new: torch.Tensor,
    ) -> torch.Tensor:
        """Wing-impenetrability slip-wall projection (Hirato §4.8.3 / Ramesh 2014 / Bird 2021).

        Prevents free vortex vertices (LEV rings, TE wake, or rVPM particles) from tunneling
        through the solid wing surface between discrete collocation points, sliding them
        tangentially at clearance `d_wall = _WING_SLIP_WALL * mean_dx` on their incoming side.
        General for flat, cambered, hinged, flapping, and flexible FSI wings.
        """

        if pts_new.shape[0] == 0:
            return pts_new
        nc = self.settings.chordwise_panels
        ns = self.settings.spanwise_panels
        r_now = geometry.rings
        c_now = 0.5 * (r_now[:, 2] + r_now[:, 3] - r_now[:, 0] - r_now[:, 1])
        dx_now = torch.linalg.vector_norm(c_now, dim=1).clamp(min=1.0e-12)
        tx_now = c_now / dx_now[:, None]
        s_now = 0.5 * (r_now[:, 1] + r_now[:, 2] - r_now[:, 0] - r_now[:, 3])
        dy_now = torch.linalg.vector_norm(s_now, dim=1).clamp(min=1.0e-12)
        ty_now = s_now / dy_now[:, None]
        n_now = geometry.normals
        # Shift bound-ring centre forward by 0.25*dx to recover the physical solid panel centre
        phys_now = 0.25 * (r_now[:, 0] + r_now[:, 1] + r_now[:, 2] + r_now[:, 3]) - 0.25 * c_now

        r_prev = bound_rings_prev
        c_prev = 0.5 * (r_prev[:, 2] + r_prev[:, 3] - r_prev[:, 0] - r_prev[:, 1])
        phys_prev = 0.25 * (r_prev[:, 0] + r_prev[:, 1] + r_prev[:, 2] + r_prev[:, 3]) - 0.25 * c_prev
        n_prev_raw = torch.linalg.cross(r_prev[:, 2] - r_prev[:, 0], r_prev[:, 3] - r_prev[:, 1], dim=1)
        n_prev_norm = torch.linalg.vector_norm(n_prev_raw, dim=1, keepdim=True).clamp(min=1.0e-12)
        n_prev = n_prev_raw / n_prev_norm
        flip = torch.sum(n_prev * n_now, dim=1) < 0.0
        n_prev = torch.where(flip[:, None], -n_prev, n_prev)

        d_wall = _WING_SLIP_WALL * float(torch.mean(dx_now).item())
        out = pts_new.clone()
        tile = 4096
        n_pts = pts_new.shape[0]
        for start in range(0, n_pts, tile):
            stop = min(start + tile, n_pts)
            p_new = pts_new[start:stop]
            p_old = pts_old[start:stop]
            diff = p_new[:, None, :] - phys_now[None, :, :]
            idx = torch.argmin(torch.sum(diff * diff, dim=2), dim=1)
            rel_new = p_new - phys_now[idx]
            xi = torch.sum(rel_new * tx_now[idx], dim=1)
            eta = torch.sum(rel_new * ty_now[idx], dim=1)
            h_new = torch.sum(rel_new * n_now[idx], dim=1)
            dx_i = dx_now[idx]
            dy_i = dy_now[idx]
            row_i = idx // ns
            # Leave a 0.05*dx margin at sharp LE and TE so nascent edge attachment is undisturbed
            in_chord = (
                ((row_i > 0) | (xi > -0.45 * dx_i))
                & ((row_i < nc - 1) | (xi < 0.45 * dx_i))
                & (torch.abs(xi) <= 0.55 * dx_i)
            )
            in_span = torch.abs(eta) <= 0.55 * dy_i
            inside = in_chord & in_span
            h_old = torch.sum((p_old - phys_prev[idx]) * n_prev[idx], dim=1)
            ones = torch.ones_like(h_new)
            side = torch.where(
                torch.abs(h_old) > 1.0e-12,
                torch.where(h_old >= 0.0, ones, -ones),
                torch.where(h_new >= 0.0, ones, -ones),
            )
            violated = inside & ((side * h_new) < d_wall)
            if bool(violated.any().item()):
                shift = torch.where(violated, side * d_wall - h_new, torch.zeros_like(h_new))
                out[start:stop] = p_new + shift[:, None] * n_now[idx]
        return out

    def _convect_lev_chain(self, geometry, trial, dt: float, core: float | None):
        """One Runge-Kutta step of the leading-edge ring chains; a new chain, the old one intact.

        Every corner is a material point.  A corner lies on the lines of its own legs, which
        give nothing there, and sees everything else: the sheet, the wake, the other rows and
        the particles.  Afterwards the aft corners of a row are pinned again to the free
        corners of the row in front of it, so the chain keeps sharing its edges exactly.
        """

        chain = trial.lev_chain
        if not bool((chain.count > 0).any().item()):
            return chain
        rows, span_panels = chain.rings.shape[0], chain.rings.shape[1]
        corners = chain.rings.reshape(-1, 3)
        rings, gamma = lev_hirato.flat_live(chain)

        # B2 (note A24): the convection must see the pseudovortex rings as well as the chain
        # rows, or the leading and trailing edges keep an uncancelled +-Gamma_L filament that the
        # first RK3 stage then winds into the wake.
        if _LEV_HIRATO_V2:
            previous = getattr(self, "_lev_pseudovortex_previous", None)
            sources = lev_hirato.free_ring_sources(
                chain,
                previous[0] if previous is not None
                else torch.zeros((0, 4, 3), device=self.device, dtype=torch.float64),
                chain.prev,
            )
            lev_core = core
        else:
            sources = (rings, gamma)
            lev_core = None

        def velocity(at: torch.Tensor) -> torch.Tensor:
            value = self._external_velocity(at, geometry, trial, core)
            value = value + self._ring_velocity(
                at, *sources, rough=False, core_radius=lev_core, free=True
            )
            if trial.particle_field.n:
                value = value + trial.particle_field.velocity_at_cuda(at)
            return value

        u1 = velocity(corners)
        u2 = velocity(corners + 0.5 * dt * u1)
        u3 = velocity(corners + dt * (-u1 + 2.0 * u2))
        moved = corners + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
        if _LEV_DETACH and getattr(self, "_lev_detached", None) is not None:
            # A detached strip's sheet is no longer attached to the edge: it drifts with the
            # freestream instead of being held there by its own induction, exactly as the frozen
            # far wake does beyond `wake_free_rows`.  Without this the sheet stays within a
            # quarter chord of the edge, keeps its adverse induction at the leading-edge row, and
            # the strip's attached solve (which resumes on detachment) has to fight it: measured
            # on the H case, CL = -2.1 at step 187 and 150 at step 195.
            mask = (
                self._lev_detached.reshape(1, span_panels, 1, 1)
                .expand(rows, span_panels, 4, 1)
                .reshape(-1, 1)
            )
            frozen = corners + dt * self.v_inf[None, :]
            moved = torch.where(mask, frozen, moved)
        if not bool(torch.isfinite(moved).all().item()):
            raise FloatingPointError("leading-edge vortex rings convected to a non-finite position")
        return lev_hirato._mask_invalid(
            lev_hirato.reconnect(chain.with_rings(moved.reshape(rows, span_panels, 4, 3))),
            chain.count,
        )

    def _lev_split_max_length(self, geometry: NativeV5MGeometry) -> torch.Tensor | None:
        """Optional per-strip nascent LEV length cap ``_LEV_SPLIT_CAP_DX * dx_0(j)``."""

        if _LEV_SPLIT_CAP_DX <= 0.0:
            return None
        ns = self.settings.spanwise_panels
        dx_left = torch.linalg.vector_norm(geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1)
        dx_right = torch.linalg.vector_norm(geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1)
        dx_local = 0.5 * (dx_left + dx_right)
        return (_LEV_SPLIT_CAP_DX * dx_local).clamp(min=1.0e-6)

    def _hirato_solve(
        self, geometry, trial, system_aic, nascent_normal, rhs, scale, lesp_pre,
        split_from_rhs=None, chain_normal_at_le=None,
        *,
        chain_split=None,
        pv_norm_cached=None,
        pseudovortex_cached=None,
    ) -> dict:
        """The circulation solve with the nascent leading-edge vortex of every shedding strip
        as an unknown (lev_hirato.solve_coupled).

        The first ring of a strip goes where Ramesh's rule puts it; a strip that already has a
        chain puts its new ring one third of the way to the chain's end (the paper's Eq. 7).
        After a solve a strip that did not shed may have crossed the critical value through
        its neighbours' induction; it joins and the solve is repeated, which is the second
        decision of the paper's flowchart.
        """

        ns = self.settings.spanwise_panels
        crit = self.settings.effective_lesp_crit
        chain = trial.lev_chain
        was_shedding = lev_hirato.continuing(chain)
        edge = lev_hirato.strip_pairs(geometry.leading_edge)
        rear = lev_hirato.bound_rear(geometry.rings, self.settings.chordwise_panels)
        back = rear                                   # (N, 2, 3): the strip's two aft corners
        pseudovortex = (
            pseudovortex_cached
            if pseudovortex_cached is not None
            else lev_hirato.pseudovortex_ring(geometry.leading_edge, rear)
        )
        # Ramesh's first-vortex displacement in the strip's own frame: chord towards the
        # trailing edge, suction-side normal, the geometric angle of the flow.  The
        # paper's A0 shares its sign with the leading-edge circulation, the solver's suction
        # parameter is minus that circulation over `scale`.
        chord_tangent = back.mean(dim=1) - edge.mean(dim=1)
        chord_tangent = chord_tangent / torch.linalg.vector_norm(
            chord_tangent, dim=1, keepdim=True
        )
        normal = geometry.normals[:ns]
        if _LEV_HIRATO_V2:
            # Relative LE velocity v_inf - v_LE handles plunging, pitching, and flapping wings
            # (reduces identically to v_inf on a stationary wing).
            le_vel = 0.5 * (geometry.leading_velocity[:-1] + geometry.leading_velocity[1:])
            v_rel = self.v_inf[None, :] - le_vel
            speed_rel = torch.linalg.vector_norm(v_rel, dim=1)
            v_inf_speed = torch.linalg.vector_norm(self.v_inf).clamp(min=1.0e-12)
            valid_rel = speed_rel > 1.0e-12
            safe_v_rel = torch.where(
                valid_rel[:, None], v_rel, self.v_inf[None, :].expand_as(v_rel)
            )
            safe_speed = torch.where(valid_rel, speed_rel, v_inf_speed)
            sin_alpha = torch.sum(safe_v_rel * normal, dim=1) / safe_speed
            cos_alpha = torch.sum(safe_v_rel * chord_tangent, dim=1) / safe_speed
            first = lev_hirato.first_free_points(
                edge, chord_tangent, normal, safe_speed,
                self.settings.aerodynamic_dt, -lesp_pre, sin_alpha, cos_alpha,
            )
            split_cap = self._lev_split_max_length(geometry)
            if split_cap is not None:
                disp0 = first - edge
                len0 = torch.linalg.vector_norm(disp0.mean(dim=1), dim=1).clamp(min=1.0e-12)
                first = edge + disp0 * torch.clamp(split_cap / len0, max=1.0)[:, None, None]
            if chain_split is None:
                if bool(was_shedding.any().item()):
                    chain_split, split = lev_hirato.split_row0(
                        chain, geometry.leading_edge, was_shedding, max_length=split_cap
                    )
                else:
                    chain_split = chain
                    split = first
            else:
                split = split_from_rhs if split_from_rhs is not None else first
            free = torch.where(was_shedding[:, None, None], split, first)
            nascent = lev_hirato.nascent_row(free, edge)
            lev_core_rhs = self._lev_rhs_core_radius(geometry)
            nascent_norm = self._ring_unit_normal(
                geometry, nascent, core_radius=lev_core_rhs, free=True
            )
            pv_norm = (
                pv_norm_cached
                if pv_norm_cached is not None
                else self._ring_unit_normal(
                    geometry, pseudovortex, core_radius=lev_core_rhs, free=False
                )
            )
            basis = nascent_norm + pv_norm                              # (P, ns)
            # Zero-shedding test gamma_pre already carried chain.prev on continuing strips,
            # so only strips whose |lesp_pre| exceeds crit shed new vorticity (dGamma_L != 0).
            # Continuing strips below crit stay in `held` (lam = chain.prev, dGamma_L = 0).
            active = torch.abs(lesp_pre) > crit
            detached = None
            if _LEV_DETACH and chain_normal_at_le is not None:
                freestream_normal = torch.sum(self.v_inf[None, :] * geometry.normals[:ns], dim=1)
                own = self._own_sheet_normal_at_leading_edge(
                    geometry, chain_split, pseudovortex=pseudovortex, core_radius=lev_core_rhs
                )
                adverse = (own * freestream_normal) > 0.0
                chain_rows = chain_split.rings.reshape(-1, ns, 4, 3)
                live = (
                    torch.arange(chain_rows.shape[0], device=self.device)[:, None]
                    < chain_split.count[None, :]
                )
                centres = chain_rows.mean(dim=2)
                strip_chord = 0.5 * (
                    geometry.trailing_edge[:-1] + geometry.trailing_edge[1:]
                    - geometry.leading_edge[:-1] - geometry.leading_edge[1:]
                )
                strip_length = torch.linalg.vector_norm(strip_chord, dim=1)
                if bool(live.any().item()):
                    chord_hat = strip_chord / strip_length[:, None]
                    strip_leading = 0.5 * (
                        geometry.leading_edge[:-1] + geometry.leading_edge[1:]
                    )
                    along = torch.sum(
                        (centres - strip_leading[None, :, :]) * chord_hat[None, :, :], dim=2
                    )
                    along = torch.where(live, along, torch.full_like(along, 1.0e9))
                    nearest = along.min(dim=0).values
                else:
                    nearest = torch.full((ns,), 1.0e9, device=self.device, dtype=torch.float64)
                near = (chain_split.count > 0) & (nearest < _LEV_DETACH_CHORDS * strip_length)
                prev_det = getattr(self, "_lev_detached", None)
                near_latch = (
                    (prev_det & near)
                    if prev_det is not None
                    else torch.zeros(ns, device=self.device, dtype=torch.bool)
                )
                carried = torch.sum(
                    torch.where(live, chain_split.gamma, torch.zeros_like(chain_split.gamma)),
                    dim=0,
                )
                over_cap = carried.abs() >= _LEV_CAP * self.settings.freestream * strip_length
                detached = adverse | near_latch | over_cap
                active = active & ~detached
            imposed = None
            if _LEV_SEPARATED_STATE and detached is not None and bool(detached.any().item()):
                front = torch.where(
                    chain_split.count > 0, chain_split.gamma[0], torch.zeros_like(chain_split.gamma[0])
                )
                imposed = (detached, front)
            pin_sign = torch.where(
                lesp_pre >= 0.0, torch.ones_like(lesp_pre), -torch.ones_like(lesp_pre)
            )
            lam = torch.zeros(ns, device=self.device, dtype=torch.float64)
            if _LEV_FLUX:
                leading_edge_panel = 0.5 * (
                    torch.linalg.vector_norm(
                        geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
                    )
                    + torch.linalg.vector_norm(
                        geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
                    )
                )
                dgamma = lev_hirato.flux_shedding_strength(
                    lesp_pre, scale, leading_edge_panel, self.settings.aerodynamic_dt
                )
                prescribed = torch.where(was_shedding, chain.prev + dgamma, dgamma)
                held_base = torch.where(was_shedding, chain.prev, torch.zeros_like(chain.prev))
                iterations = 0
                for iterations in range(1, 9):
                    lam = torch.where(active, prescribed, held_base)
                    push_mask = active | was_shedding
                    gamma, residual = lev_hirato.solve_prescribed(
                        system_aic, nascent_normal, basis, rhs, push_mask, lam
                    )
                    solved = -gamma[:ns] / scale
                    crossed = (torch.abs(solved) > crit) & ~active
                    if detached is not None:
                        crossed = crossed & ~detached
                    if not bool(crossed.any().item()):
                        break
                    active = active | crossed
                return {
                    "gamma": gamma, "lam": lam, "active": active,
                    "push_mask": active | was_shedding, "chain_split": chain_split,
                    "pin_sign": pin_sign, "residual": residual,
                    "nascent": nascent, "iterations": iterations, "shed_law": "flux",
                    "detached_strips": int(detached.sum().item()) if detached is not None else 0,
                    "imposed_strips": imposed[0] if imposed is not None else None,
                }
            iterations = 0
            while True:
                iterations += 1
                target = -pin_sign * crit * scale
                held = was_shedding & ~active
                held_idx = torch.nonzero(held, as_tuple=False).reshape(-1)
                if held_idx.numel():
                    rhs_iter = rhs - (
                        nascent_normal[:, held_idx] + basis[:, held_idx]
                    ) @ chain.prev[held_idx]
                else:
                    rhs_iter = rhs
                index = torch.nonzero(active, as_tuple=False).reshape(-1)
                gamma, strength, residual = lev_hirato.solve_coupled(
                    system_aic, nascent_normal, basis[:, index], rhs_iter, active, target,
                    imposed=imposed,
                )
                lam.zero_()
                if held_idx.numel():
                    lam[held_idx] = chain.prev[held_idx]
                if index.numel():
                    lam[index] = strength
                solved = -gamma[:ns] / scale
                crossed = (torch.abs(solved) > crit) & ~active
                if detached is not None:
                    crossed = crossed & ~detached
                dlam = lam - chain.prev
                wrong_dir = active & ((dlam * torch.sign(target)) < 0.0)
                if not (bool(crossed.any().item()) or bool(wrong_dir.any().item())) or iterations >= 8:
                    break
                pin_sign = torch.where(
                    crossed,
                    torch.where(solved >= 0.0, torch.ones_like(solved), -torch.ones_like(solved)),
                    pin_sign,
                )
                active = (active | crossed) & ~wrong_dir
            self._lev_detached = detached
            return {
                "gamma": gamma, "lam": lam, "active": active,
                "push_mask": active | was_shedding, "chain_split": chain_split,
                "pin_sign": pin_sign, "residual": residual,
                "nascent": nascent, "iterations": iterations,
                "detached_strips": int(detached.sum().item()) if detached is not None else 0,
                "imposed_strips": imposed[0] if imposed is not None else None,
            }

        speed = torch.linalg.vector_norm(self.v_inf)
        sin_alpha = torch.sum(self.v_inf[None, :] * normal, dim=1) / speed
        cos_alpha = torch.sum(self.v_inf[None, :] * chord_tangent, dim=1) / speed
        first = lev_hirato.first_free_points(
            edge, chord_tangent, normal, float(speed.item()),
            self.settings.aerodynamic_dt, -lesp_pre, sin_alpha, cos_alpha,
        )
        # the paper's split applies to a chain that is still shedding; a chain that stopped
        # has left the edge, and the strip's next ring is a first ring on the edge again
        # B3 (note A24): Eq. (7) measures one third of the way to the sheet's FREE end; the old
        # call measured to its leading-edge-side end, which is one convection step from the edge
        # whatever the sheet's age, so every new ring came out a third of a step long.  Under
        # hirato_v2 the split was already taken when the right-hand side was built and the same
        # one is used here, so the solve and the commit agree (B1).
        split = (
            split_from_rhs
            if split_from_rhs is not None
            else lev_hirato.split_points(edge, lev_hirato.row_aft_pairs(chain.rings[0]))
        )
        free = torch.where(lev_hirato.continuing(chain)[:, None, None], split, first)
        nascent = lev_hirato.nascent_row(free, edge)

        def unit_normal(rings: torch.Tensor) -> torch.Tensor:
            # the matrix of the solve: the exact law of the selected kernel, no core
            if _KERNEL_V3:
                expanded = native_ring_velocity_expanded(
                    geometry.collocation, rings, core_fraction=0.0, reference_length=0.0
                )
            else:
                expanded = native_ring_velocity_expanded(
                    geometry.collocation,
                    rings,
                    core_fraction=1.0e-6,
                    reference_length=self._legacy_reference_length,
                )
            return torch.sum(expanded * geometry.normals[:, None, :], dim=2)

        basis = unit_normal(nascent) + unit_normal(pseudovortex)      # (P, ns)
        # a strip starts shedding when its attached suction parameter is past the critical
        # value, and one that was shedding stays in until the rule below takes it out
        was_shedding = lev_hirato.continuing(chain)
        active = (torch.abs(lesp_pre) > crit) | was_shedding
        detached = None
        if _LEV_DETACH and chain_normal_at_le is not None:
            # The sheet's induction at its own leading-edge row against the freestream's: when the
            # two share a sign the sheet no longer relieves the edge, so the strip stops being fed.
            # A strip that has detached stays detached while its sheet is still in the near field
            # (`FLUXV_LEV_DETACH_CHORDS` chords of the leading edge): without that latch the sign
            # flips back as soon as the sheet moves a little, the strip sheds again, and the result
            # is the on/off switching this criterion exists to remove -- measured on the H case:
            # 14, 8, 4, 26, 9 strips in five samples, then the runaway at step 214.
            freestream_normal = torch.sum(self.v_inf[None, :] * geometry.normals[:ns], dim=1)
            # Only a strip's OWN sheet can detach it.  The total chain induction at a strip's
            # leading-edge row is dominated by its neighbours' sheets across the span, so using it
            # detached all 48 strips the moment 14 began to shed (measured: step 162 -> 163, note
            # A25).  Each strip's own rows are evaluated at its own leading-edge point instead.
            own = self._own_sheet_normal_at_leading_edge(geometry, trial.lev_chain)
            adverse = (own * freestream_normal) > 0.0
            chain_rows = trial.lev_chain.rings.reshape(-1, ns, 4, 3)
            live = (
                torch.arange(chain_rows.shape[0], device=self.device)[:, None]
                < trial.lev_chain.count[None, :]
            )
            # how far the nearest ring of every strip is downstream of its leading edge
            centres = chain_rows.mean(dim=2)
            if bool(live.any().item()):
                # the strip's own chord direction, from the leading edge to the trailing edge
                strip_chord = 0.5 * (
                    geometry.trailing_edge[:-1] + geometry.trailing_edge[1:]
                    - geometry.leading_edge[:-1] - geometry.leading_edge[1:]
                )
                chord_hat = strip_chord / torch.linalg.vector_norm(
                    strip_chord, dim=1, keepdim=True
                )
                strip_leading = 0.5 * (
                    geometry.leading_edge[:-1] + geometry.leading_edge[1:]
                )
                along = torch.sum(
                    (centres - strip_leading[None, :, :]) * chord_hat[None, :, :], dim=2
                )
                along = torch.where(live, along, torch.full_like(along, 1.0e9))
                nearest = along.min(dim=0).values
            else:
                nearest = torch.full((ns,), 1.0e9, device=self.device, dtype=torch.float64)
            strip_length = torch.linalg.vector_norm(strip_chord, dim=1)
            near = (trial.lev_chain.count > 0) & (
                nearest < _LEV_DETACH_CHORDS * strip_length
            )
            # what the strip's sheet already carries, against the detachment cap
            carried = torch.sum(
                torch.where(live, trial.lev_chain.gamma, torch.zeros_like(trial.lev_chain.gamma)),
                dim=0,
            )
            chord_length = torch.linalg.vector_norm(geometry.trailing_edge[:-1]
                                                    - geometry.leading_edge[:-1], dim=1)
            over_cap = carried.abs() >= _LEV_CAP * self.settings.freestream * chord_length
            detached = adverse | near | over_cap
            active = active & ~detached
        target = -torch.sign(lesp_pre) * crit * scale
        imposed = None
        if _LEV_SEPARATED_STATE and detached is not None and bool(detached.any().item()):
            # the sheet that left the strip: its newest ring carries the strength the bound row
            # takes over, so the loading a separated strip holds is the one it separated at
            front = torch.where(
                chain.count > 0, chain.gamma[0], torch.zeros_like(chain.gamma[0])
            )
            imposed = (detached, front)
        lam = torch.zeros(ns, device=self.device, dtype=torch.float64)
        if _LEV_FLUX:
            # the flux law prescribes the strength; nothing is pinned, every row keeps its
            # no-penetration equation
            leading_edge_panel = 0.5 * (
                torch.linalg.vector_norm(
                    geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
                )
                + torch.linalg.vector_norm(
                    geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
                )
            )
            prescribed = lev_hirato.flux_shedding_strength(
                lesp_pre, scale, leading_edge_panel, self.settings.aerodynamic_dt
            )
            for _ in range(8):
                lam = torch.where(active, prescribed, torch.zeros_like(prescribed))
                gamma, residual = lev_hirato.solve_prescribed(
                    system_aic, nascent_normal, basis, rhs, active, lam
                )
                solved = -gamma[:ns] / scale
                crossed = (torch.abs(solved) > crit) & ~active
                if detached is not None:
                    # a detached strip's row carries the sheet's strength, not the pin; putting it
                    # back in the active set would give that row two equations at once
                    crossed = crossed & ~detached
                if not bool(crossed.any().item()):
                    break
                active = active | crossed
            return {
                "gamma": gamma, "lam": lam, "active": active, "residual": residual,
                "nascent": nascent, "iterations": _, "shed_law": "flux",
            }
        iterations = 0
        while True:
            iterations += 1
            index = torch.nonzero(active, as_tuple=False).reshape(-1)
            gamma, strength, residual = lev_hirato.solve_coupled(
                system_aic, nascent_normal, basis[:, index], rhs, active, target,
                imposed=imposed,
            )
            lam.zero_()
            lam[index] = strength
            solved = -gamma[:ns] / scale
            crossed = (torch.abs(solved) > crit) & ~active
            if detached is not None:
                # a detached strip's row carries the sheet's strength, not the pin; putting it back
                # in the active set would give that row two equations at once
                crossed = crossed & ~detached
            # A strip that was shedding goes on shedding until continuing would need a
            # leading-edge vortex against the bound circulation: then it stops.  Without this a
            # strip whose suction sits at the critical value switches on and off every step, its
            # pseudovortex ring appears and disappears at full strength, and the rate term turns
            # that into a force swinging by the whole load (rigid plate at 16 deg, note A23, 6c).
            if _LEV_HIRATO_V2:
                # B4 (note A24): a strip stops only when the pin asks for a leading-edge vortex of
                # the opposite sign to the one it carries; that is the paper's rule and it does
                # not fire on a mere decrease of magnitude, which switched the edge strips on and
                # off every step (H case: 14/20/18/24/20/30 strips in six steps).
                _, keeps_shedding = lev_hirato.cessation_delta(active, lam, chain.prev)
                reversed_sign = was_shedding & active & ~keeps_shedding
            else:
                reversed_sign = was_shedding & active & (lam * torch.sign(target) < 0.0)
            if not (bool(crossed.any().item()) or bool(reversed_sign.any().item())) or iterations >= 8:
                break
            active = (active | crossed) & ~reversed_sign
        # A detached strip never entered the active set, so it never received a column in the
        # solve and its strength is already zero: zeroing it again here would break the pairing
        # between gamma and lam (measured: with the zeroing the run went to CL = -78 at step 175
        # with no strip shedding at all).
        # the next step's convection reads this: a detached strip's rings stop deforming and drift
        # with the freestream, like the frozen far wake (`wake_free_rows`)
        self._lev_detached = detached
        return {
            "gamma": gamma, "lam": lam, "active": active, "residual": residual,
            "nascent": nascent, "iterations": iterations,
            "detached_strips": int(detached.sum().item()) if detached is not None else 0,
            "imposed_strips": imposed[0] if imposed is not None else None,
        }

    def _own_sheet_normal_at_leading_edge(
        self,
        geometry,
        chain,
        *,
        pseudovortex: torch.Tensor | None = None,
        core_radius: float | None = None,
    ) -> torch.Tensor:
        """(ns,) the leading-edge normal velocity each strip's OWN sheet (+ pseudovortex) induces
        on its own edge.

        The criterion for detachment is about the sheet that a strip feeds, so the neighbours'
        sheets must not enter it (note A25, section 4).
        """

        from .kernels_ring_velocity import native_ring_velocity_expanded

        strips = self.settings.spanwise_panels
        leading = 0.5 * (geometry.leading_edge[:-1] + geometry.leading_edge[1:])
        normals = geometry.normals[:strips]
        result = torch.zeros(strips, device=self.device, dtype=torch.float64)
        keywords = (
            {"core_fraction": 0.0, "reference_length": 0.0, "core_radius": core_radius}
            if _KERNEL_V3
            else {"core_fraction": 1.0e-6, "reference_length": self._legacy_reference_length}
        )
        for strip in range(strips):
            count = int(chain.count[strip].item())
            if count == 0:
                continue
            rings = chain.rings[:count, strip].contiguous()
            gammas = chain.gamma[:count, strip]
            if pseudovortex is not None and float(chain.prev[strip].item()) != 0.0:
                rings = torch.cat((rings, pseudovortex[strip : strip + 1]), dim=0)
                gammas = torch.cat((gammas, chain.prev[strip : strip + 1]), dim=0)
            unit = native_ring_velocity_expanded(
                leading[strip][None, :], rings, **keywords
            )                                                        # (1, count(+1), 3)
            value = torch.sum(unit[0] * gammas[:, None], dim=0)
            result[strip] = torch.sum(value * normals[strip])
        return result

    def _hirato_commit(self, geometry, trial, solution: dict) -> dict:
        """Store the nascent rows, audit the rows that leave, set the load fields.

        Every row stays an exact ring up to `LEV_MAX_ROWS` per strip; a row that leaves is
        dropped and what it carried goes into the retention ledger (the trailing-edge wake is
        truncated the same way).  A first version handed the leaving rows to the particle field;
        they were smoothed away there (note A23, 6b, lev_hirato.py).
        """

        chain = trial.lev_chain
        active, strength = solution["active"], solution["lam"]
        pseudovortex = lev_hirato.pseudovortex_ring(
            geometry.leading_edge,
            lev_hirato.bound_rear(geometry.rings, self.settings.chordwise_panels),
        )
        if _LEV_HIRATO_V2:
            base_chain = solution.get("chain_split", chain)
            push_mask = solution.get("push_mask", active)
            pushed, leaving_rings, leaving_gamma, released = lev_hirato.push_nascent(
                base_chain, push_mask, solution["nascent"], strength
            )
            audit = lev_hirato.truncation_audit(leaving_rings, leaving_gamma, released)
            delta = torch.where(active, strength - chain.prev, torch.zeros_like(strength))
            new_prev = torch.where(push_mask, strength, chain.prev)
            trial.lev_chain = lev_hirato.LevChain(
                rings=pushed.rings, gamma=pushed.gamma, count=pushed.count,
                prev=new_prev, joined=pushed.joined,
            )
            trial.lev_strip_last_release = new_prev.clone()
            trial.lev_strip_released_total = trial.lev_strip_released_total + delta
            rings, gamma = lev_hirato.flat_live(trial.lev_chain)
            self._lev_field = (rings, gamma)
            self._lev_field_cores = self._lev_live_core_radii(
                trial.lev_chain, self._lev_rhs_core_radius(geometry), row_age_offset=0
            )
            self._lev_pseudovortex = (pseudovortex, new_prev)
            trial.pseudovortex_previous = (pseudovortex.clone(), new_prev.clone())
            self._lev_pseudovortex_previous = (pseudovortex.clone(), new_prev.clone())
            return audit

        pushed, leaving_rings, leaving_gamma, released = lev_hirato.push_nascent(
            chain, active, solution["nascent"], strength
        )
        audit = lev_hirato.truncation_audit(leaving_rings, leaving_gamma, released)
        trial.lev_chain = lev_hirato.LevChain(
            rings=pushed.rings, gamma=pushed.gamma, count=pushed.count,
            prev=torch.where(active, strength, 0.0), joined=pushed.joined,
        )
        release = torch.where(active, strength, 0.0)
        trial.lev_strip_last_release = release.clone()
        trial.lev_strip_released_total = trial.lev_strip_released_total + release
        # what every load evaluation sees: the chains as they stand, the nascent rows with
        # their solved strength included, and the pseudovortex rings at that strength
        rings, gamma = lev_hirato.flat_live(trial.lev_chain)
        # two fields, not one: the chain rows are free vorticity and carry the free-wake core
        # wherever they are evaluated, in the loads as in the solve; the pseudovortex rings are
        # bound and are evaluated with the exact law
        self._lev_field = (rings, gamma)
        self._lev_pseudovortex = (pseudovortex, strength)
        return audit

    def _hirato_suction(
        self, geometry, gamma, scale, chord, *, speed=None, force_enable: bool = False
    ):
        """The leading-edge suction of every strip as a force per panel, or None."""

        if not (_LE_SUCTION or force_enable):
            return None
        ns = self.settings.spanwise_panels
        leading, trailing = geometry.leading_edge, geometry.trailing_edge
        width = torch.linalg.vector_norm(leading[1:] - leading[:-1], dim=1)
        forward = 0.5 * (leading[:-1] + leading[1:] - trailing[:-1] - trailing[1:])
        forward = forward / torch.linalg.vector_norm(forward, dim=1, keepdim=True)
        lesp = -gamma[:ns] / scale
        u_eff = self.settings.freestream if speed is None else speed
        force = lev_hirato.suction_force(
            self.settings.density, u_eff, chord, width, forward, lesp
        )
        panel = torch.zeros((gamma.shape[0], 3), device=gamma.device, dtype=torch.float64)
        panel[:ns] = force                # the first chordwise row, panel i * ns + j
        return panel

    def _advance_material_state(
        self, geometry: NativeV5MGeometry, trial: NativeV5MState
    ) -> dict[str, float | int]:
        """Convect the free wake/particles; return the retention record.

        The particle age-cull that mirrors the ring-wake truncation happens
        here; the removed circulation/impulse are audited BEFORE removal so
        the caller's circulation ledger can carry the accounting (plan §5.9,
        §8.5 — a silent cap that destroys circulation is forbidden).
        """
        retention: dict[str, float | int] = {
            "particle_cull_count": 0,
            "particle_cull_circulation": 0.0,
            "particle_cull_linear_impulse": [0.0, 0.0, 0.0],
        }
        dt = self.settings.aerodynamic_dt
        synchronous_material = self.settings.joint_separation_solve
        core = self._core_radius(geometry)
        if (_CONVECTION_V4 or _LEV_HIRATO_V2) and not synchronous_material:
            return self._advance_material_state_v4(geometry, trial, dt, core, retention)
        if synchronous_material:
            from .native_material_convection import advance_material_positions_rk3

            advance_material_positions_rk3(self, geometry, trial)
        if trial.wake_rings.shape[0] and not synchronous_material:
            ns = self.settings.spanwise_panels
            total_rows = trial.wake_rings.shape[0] // ns
            freeze_rows = self.settings.wake_free_rows
            if freeze_rows and total_rows > freeze_rows:
                # The author's far-wake freeze (generate_wake.m): rows older
                # than `wake_free_rows` shed events convect with the
                # freestream alone and stop deforming, bounding the O(N^2)
                # self-convection cost of long free wakes.  Rows stay full
                # vortex sources for everything they influence.
                original = trial.wake_rings.clone()
                active = original[: freeze_rows * ns]

                def active_velocity(at: torch.Tensor) -> torch.Tensor:
                    return self._external_velocity(
                        at.reshape(-1, 3), geometry, trial, core
                    ).reshape_as(at)

                u1 = active_velocity(active)
                if trial.particle_field.n:
                    u1 = u1 + trial.particle_field.velocity_at_cuda(
                        active.reshape(-1, 3)
                    ).reshape_as(active)
                u2 = active_velocity(active + 0.5 * dt * u1)
                if trial.particle_field.n:
                    u2 = u2 + trial.particle_field.velocity_at_cuda(
                        (active + 0.5 * dt * u1).reshape(-1, 3)
                    ).reshape_as(active)
                u3 = active_velocity(active + dt * (-u1 + 2.0 * u2))
                if trial.particle_field.n:
                    u3 = u3 + trial.particle_field.velocity_at_cuda(
                        (active + dt * (-u1 + 2.0 * u2)).reshape(-1, 3)
                    ).reshape_as(active)
                updated_active = active + (dt / 6.0) * (u1 + 4.0 * u2 + u3)
                frozen_tail = original[freeze_rows * ns :] + self.v_inf * dt
                trial.wake_rings = torch.cat((updated_active, frozen_tail), dim=0)
            else:
                original = trial.wake_rings.clone()

                def wake_velocity(at: torch.Tensor) -> torch.Tensor:
                    flat = at.reshape(-1, 3)
                    velocity = self._external_velocity(flat, geometry, trial, core)
                    if trial.particle_field.n:
                        velocity = velocity + trial.particle_field.velocity_at_cuda(flat)
                    return velocity.reshape_as(at)

                u1 = wake_velocity(original)
                u2 = wake_velocity(original + 0.5 * dt * u1)
                u3 = wake_velocity(original + dt * (-u1 + 2.0 * u2))
                trial.wake_rings.copy_(original + (dt / 6.0) * (u1 + 4.0 * u2 + u3))

        # Plan V3 stage 2: the leading-edge ring chains convect like the wake, with the
        # same Runge-Kutta step; the trailing-edge wake itself does not see them (they sit at
        # the other end of the chord).
        if trial.lev_chain is not None:
            trial.lev_chain = self._convect_lev_chain(geometry, trial, dt, core)

        def particle_external(points: torch.Tensor) -> torch.Tensor:
            velocity = self._external_velocity(points, geometry, trial, core)
            if trial.lev_chain is not None:
                velocity = velocity + self._lev_chain_velocity(points, trial.lev_chain)
            return velocity

        if trial.particle_field.n:
            if not synchronous_material:
                trial.particle_field.advance_wrk3(dt, particle_external)
            if self.settings.particle_max_age_steps > 0:
                horizon = trial.step - self.settings.particle_max_age_steps
                if horizon > 0:
                    alive = trial.particle_field.n
                    stale = (
                        trial.particle_field.birth_step[:alive] < horizon
                    )
                    removed = int(torch.count_nonzero(stale).item())
                    if removed:
                        # Audit before removal (retention ledger, plan §8.5).
                        retention["particle_cull_count"] = removed
                        retention["particle_cull_circulation"] = float(
                            trial.particle_field.circul[:alive][stale]
                            .sum()
                            .item()
                        )
                        impulse = 0.5 * torch.cross(
                            trial.particle_field.pos[:alive][stale],
                            trial.particle_field.gamma[:alive][stale],
                            dim=1,
                        ).sum(dim=0)
                        retention["particle_cull_linear_impulse"] = [
                            float(value) for value in impulse.tolist()
                        ]
                        trial.particle_field.remove_mask(
                            trial.particle_field.birth_step[: trial.particle_field.n]
                            < horizon
                        )
        return retention

    def _dvm_kinematics(
        self, geometry: NativeV5MGeometry, trial: NativeV5MState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ns = self.settings.spanwise_panels
        chord_cell = 0.5 * (
            geometry.trailing_edge[:-1]
            + geometry.trailing_edge[1:]
            - geometry.leading_edge[:-1]
            - geometry.leading_edge[1:]
        )
        cell_x = chord_cell / torch.linalg.vector_norm(chord_cell, dim=1, keepdim=True)
        panel_normals = geometry.normals.reshape(self.settings.chordwise_panels, ns, 3)
        cell_normal = torch.mean(panel_normals, dim=0)
        cell_normal = cell_normal / torch.linalg.vector_norm(cell_normal, dim=1, keepdim=True)
        node_x_vector = geometry.trailing_edge - geometry.leading_edge
        node_x = node_x_vector / torch.linalg.vector_norm(node_x_vector, dim=1, keepdim=True)
        node_normal = torch.empty((ns + 1, 3), device=self.device, dtype=torch.float64)
        node_normal[0], node_normal[-1] = cell_normal[0], cell_normal[-1]
        if ns > 1:
            node_normal[1:-1] = cell_normal[:-1] + cell_normal[1:]
        node_normal = node_normal / torch.linalg.vector_norm(node_normal, dim=1, keepdim=True)
        cell_alpha = torch.atan2(
            torch.sum(cell_normal * self.v_inf, dim=1),
            torch.sum(cell_x * self.v_inf, dim=1),
        )
        node_alpha = torch.atan2(
            torch.sum(node_normal * self.v_inf, dim=1),
            torch.sum(node_x * self.v_inf, dim=1),
        )
        alpha = torch.cat((cell_alpha, node_alpha))
        if trial.alpha_previous is None:
            alpha_rate = torch.zeros_like(alpha)
        else:
            delta = torch.remainder(alpha - trial.alpha_previous + math.pi, 2.0 * math.pi) - math.pi
            alpha_rate = delta / trial.source_bank.dt
        node_heave = torch.sum(geometry.leading_velocity * node_normal, dim=1) / self.settings.freestream
        heave = torch.cat((0.5 * (node_heave[:-1] + node_heave[1:]), node_heave))
        return alpha, alpha_rate, heave, node_x, node_normal

    def _lev_sheet_frontier(
        self,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
        source_result: dict[str, torch.Tensor | bool],
        node_x: torch.Tensor,
        node_normal: torch.Tensor,
        node_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aft edge of the newborn leading-edge sheet, per span node.

        ONE definition shared by the production deposit and the joint solve's ribbon
        basis.  The two sites used to compute this inline and they had drifted apart a
        sign once before (the leading-edge pin), so the placement lives here.
        """

        ns = self.settings.spanwise_panels
        birth_2d = source_result["lev_birth_position"][ns:]
        anchor_2d = source_result["lev_edge_position"][ns:]
        displacement = birth_2d - anchor_2d
        frontier = geometry.leading_edge + trial.reference_node_chord[:, None] * (
            displacement[:, :1] * node_x + displacement[:, 1:] * node_normal
        )
        if _LE_SHEET_EXTENT != "off":
            # A controlled sweep of the deposit's chordwise reach -- NOT a literature
            # placement; the flag comment says exactly what it is and is not.  The 2-D
            # birth displacement keeps its NORMAL component (that is what lifts the
            # sheet off the surface); only the chordwise extent is replaced.
            #
            # UNITS: `reference_node_chord` is |trailing_edge - leading_edge|, i.e. the
            # FULL chord (initialize()), while the reach is given in PANELS.  Writing
            # `_LE_SHEET_EXTENT_PANELS * node_x` directly would put the sheet two chords
            # aft, so the panel count is divided by the chordwise panel count here.
            reach_fraction = _LE_SHEET_EXTENT_PANELS / self.settings.chordwise_panels
            frontier = geometry.leading_edge + trial.reference_node_chord[:, None] * (
                reach_fraction * node_x + displacement[:, 1:] * node_normal
            )
        if node_active is not None:
            frontier = torch.where(
                node_active[:, None], frontier, geometry.leading_edge
            )
        return frontier

    def _deposit_dvm_ribbon(
        self,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
        result: dict[str, torch.Tensor | bool],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ns = self.settings.spanwise_panels
        cell_active = result["shed_lev"][:ns]
        node_active = result["shed_lev"][ns:]
        gamma_lev = result["gamma_lev_new"][:ns] * (
            self.settings.freestream * trial.reference_cell_chord
        )
        # Leading-edge shedding ledger (m^2/s, the SAME measure as the bound
        # circulation): the newest LE sheet row is exactly this strength, so
        # it is the only unambiguous per-strip LEV number the solver can
        # offer.  Never reconstruct it from particle `circul` (edge-level,
        # repeated per particle).
        trial.lev_strip_last_release = gamma_lev.detach().clone()
        if trial.lev_strip_released_total is None:
            trial.lev_strip_released_total = torch.zeros_like(gamma_lev)
        trial.lev_strip_released_total = (
            trial.lev_strip_released_total + gamma_lev.detach()
        )
        alpha, _, _, node_x, node_normal = self._dvm_kinematics(geometry, trial)
        del alpha
        frontier = self._lev_sheet_frontier(
            geometry, trial, result, node_x, node_normal, node_active
        )
        if self.settings.le_open_deposit:
            # Route A: a single-signed spanwise filament ON the physical leading
            # edge.  Net vorticity is -gamma_lev (not zero), so the deposited field
            # is a monopole sheet and its induction at the first collocation row is
            # the full monopole value instead of 4% of it.
            trial.particle_field.add_open_filament_particles(
                geometry.leading_edge,
                -gamma_lev,
                smoothing_radius=self.settings.dvm_smoothing_radius_chord
                * float(torch.mean(trial.reference_cell_chord).item()),
                target_spacing=self.settings.dvm_target_spacing_chord
                * float(torch.mean(trial.reference_cell_chord).item()),
                birth_step=trial.step,
            )
        else:
            trial.particle_field.add_connected_ribbon_particles(
                geometry.leading_edge,
                frontier,
                -gamma_lev,
                smoothing_radius=self.settings.dvm_smoothing_radius_chord
                * float(torch.mean(trial.reference_cell_chord).item()),
                target_spacing=self.settings.dvm_target_spacing_chord
                * float(torch.mean(trial.reference_cell_chord).item()),
                birth_step=trial.step,
                connector_source_strips=torch.clamp(
                    torch.arange(ns + 1, device=self.device, dtype=torch.int64) - 1,
                    min=0,
                    max=ns - 1,
                ),
            )
        trial.frontier_nodes = torch.where(
            node_active[:, None], frontier, trial.frontier_nodes
        )
        trial.frontier_active = node_active.clone()
        return gamma_lev, cell_active

    def propose(
        self,
        committed: NativeV5MState,
        structural_state: wp.array,
        structural_velocity: wp.array,
        preconvected: "NativeV5MState | None" = None,
        capture_preconvected: bool = False,
        inherited_retention: dict | None = None,
    ) -> NativeV5MProposal:
        parent_digest = committed.digest()
        parent_fingerprint = committed.mutation_fingerprint()
        if preconvected is None:
            trial = committed.clone()
        else:
            # S4a (2026-11-22, flag-gated diagnostic branch): start from a state whose
            # free wake and particles were already convected this outer step instead of
            # convecting ``committed`` again.  The caller passes the trial state of a
            # proposal made at a structural state that differs only by the coupling
            # residual, so this is a LAGGED convection, not a repeated identical
            # computation -- the acceptance is therefore a tolerance on the loads, never
            # bitwise equality.
            #
            # The preconvected state is CLONED: `propose` writes into `trial`
            # (lev_strip_last_release, the deposit, gamma_bound, ...) and the caller
            # still owns the proposal this state came from.
            trial = preconvected.clone()
            # The source state was itself produced by a propose(), so it already carries
            # step = committed.step + 1 and `propose` increments the step at the end.  Reset
            # it to the committed step so that (a) the commit guard still sees exactly one
            # advance and (b) every in-propose use of `trial.step` (particle birth labels,
            # the LE window cutoff, the age horizon) means the same thing it does on the
            # default path.
            trial.step = committed.step
        geometry = self.surface.evaluate(structural_state, structural_velocity)
        # the legacy core's reference length follows the wing (plan V3, Bug-4)
        self._legacy_reference_length = mean_chordwise_panel(geometry.rings)
        # This step's release is not known yet: clear the per-step record so a
        # non-releasing step cannot inherit the previous step's strength.
        trial.lev_strip_last_release = torch.zeros(
            self.settings.spanwise_panels,
            device=self.device,
            dtype=torch.float64,
        )
        if preconvected is None:
            retention = self._advance_material_state(geometry, trial)
        else:
            # The age cull ran inside the convection this state came from, so its
            # accounting is INHERITED rather than reported as zero -- otherwise the
            # committed record would silently claim nothing was removed.  The marker
            # `convected_reuse` keeps the branch visible in the payload.
            retention = {
                "particle_cull_count": int(
                    (inherited_retention or {}).get("particle_cull_count", 0)
                ),
                "particle_cull_circulation": float(
                    (inherited_retention or {}).get("particle_cull_circulation", 0.0)
                ),
                "particle_cull_linear_impulse": list(
                    (inherited_retention or {}).get(
                        "particle_cull_linear_impulse", [0.0, 0.0, 0.0]
                    )
                ),
                "convected_reuse": 1,
            }
        # Stage 2 (experimental, default OFF): 3-D vortex stretching on the live
        # particles.  Applied right after the material convection so the strength
        # is updated from the same positions the next stage will see; first order
        # in time for the strength (positions keep their RK3 update).
        # Rationale: A1/A2/A3 showed Gamma_LEV accumulates without bound and
        # nothing saturates it -- the dipole representation merely hid it.  The
        # Helmholtz stretching term is the physical mechanism for both, so it is
        # measured rather than asserted.
        if getattr(self.settings, "particle_stretching", False):
            stretching = trial.particle_field.apply_vortex_stretching(
                self.settings.aerodynamic_dt,
                core_spreading=bool(
                    getattr(self.settings, "particle_core_spreading", False)
                ),
                rvpm=bool(getattr(self.settings, "particle_stretching_rvpm", False)),
                transposed=bool(
                    getattr(self.settings, "particle_stretching_transposed", True)
                ),
            )
            retention["particle_stretching"] = stretching
        # S4a: snapshot here -- after the free wake/particles are convected and stretched,
        # but BEFORE any deposit or wake truncation.  Handing this to a later propose()
        # lets it skip the O(N^2) convection while still doing its own deposit and its own
        # truncation, which is what keeps the particle count and the ledger identical.
        preconvected_state = trial.clone() if capture_preconvected else None
        preconvected_retention = (
            {
                name: retention[name]
                for name in (
                    "particle_cull_count",
                    "particle_cull_circulation",
                    "particle_cull_linear_impulse",
                )
            }
            if capture_preconvected
            else None
        )
        aic = native_aic(geometry, chordwise_panels=self.settings.chordwise_panels)
        # Plan V4 Track A-1: under FLUXV_TE_WAKE=v4, split row 0 of the convected wake at
        # split_te = te_edge + beta * (conv_front - te_edge) BEFORE evaluating wake_velocity
        # so the RHS, nascent_rings on LHS, loads, and committed wake all see one continuous,
        # gap-free, non-overlapping vortex sheet with exact cancellation at split_te.
        if _TE_WAKE_V4 and _SHED_DISTANCE_FRACTION == 0.0 and trial.wake_rings.shape[0]:
            span_panels = self.settings.spanwise_panels
            chord_panels = self.settings.chordwise_panels
            last_row_v4 = geometry.rings.reshape(chord_panels, span_panels, 4, 3)[-1]
            te_edge_v4 = torch.stack((last_row_v4[:, 3], last_row_v4[:, 2]), dim=1)
            conv_front_v4 = trial.wake_rings[:span_panels, 0:2]
            split_te_v4 = te_edge_v4 + _TE_SPLIT_BETA * (conv_front_v4 - te_edge_v4)
            trial.wake_rings[:span_panels, 0] = split_te_v4[:, 0]
            trial.wake_rings[:span_panels, 1] = split_te_v4[:, 1]
        wake_col_core = self._wake_collocation_core_radius(geometry)
        if wake_col_core is not None and _SQUIRE_CORE > 0.0 and trial.wake_rings.shape[0]:
            wake_col_core = self._wake_core_radii(
                trial.wake_gamma, wake_col_core, row_age_offset=1
            )
        wake_velocity = self._ring_velocity(
            geometry.collocation,
            trial.wake_rings,
            trial.wake_gamma,
            rough=False,
            core_radius=wake_col_core,
            free=True,
        )
        # Plan V3 (P0-B): under FLUXV_TE_WAKE=v3/v4 the row that leaves the trailing
        # edge this step is part of the solve.  Its back edge is the trailing
        # edge and its front edge is where the newest convected row's front
        # corners are (one U*dt downstream on the first step, or beta*U_rel*dt under v4),
        # so the chain stays connected and no row overlaps its neighbour.  Its strength is
        # the trailing-edge unknown of its strip, so its unit influence joins
        # the trailing-edge columns; the leading-edge part of the row is
        # deposited later at a known strength and goes to the right-hand side.
        # On the legacy path none of this exists and `system_aic` is `aic`.
        nascent_rings = None
        nascent_unit = None
        nascent_normal = None
        system_aic = aic
        if _TE_WAKE_V3:
            span_panels, chord_panels = (
                self.settings.spanwise_panels,
                self.settings.chordwise_panels,
            )
            last_row = geometry.rings.reshape(chord_panels, span_panels, 4, 3)[-1]
            if trial.wake_rings.shape[0]:
                newest_front = trial.wake_rings[:span_panels, 0:2]       # (ns, 2, 3)
            elif _TE_WAKE_V4:
                te_vel = torch.stack(
                    (geometry.trailing_velocity[:-1], geometry.trailing_velocity[1:]), dim=1
                )
                newest_front = (
                    torch.stack((last_row[:, 3], last_row[:, 2]), dim=1)
                    + _TE_SPLIT_BETA
                    * (self.v_inf[None, None, :] - te_vel)
                    * self.settings.aerodynamic_dt
                )
            else:
                newest_front = (
                    torch.stack((last_row[:, 3], last_row[:, 2]), dim=1)
                    + self.v_inf[None, :] * self.settings.aerodynamic_dt
                )
            if _SHED_DISTANCE_FRACTION > 0.0:
                # the shed-vortex distance of Kumar et al. (both papers, after Katz and Plotkin):
                # the newborn row is its own ring, from the trailing edge downstream by this
                # fraction of one convection step, and it is not tied to the rows behind it
                newest_front = torch.stack((last_row[:, 3], last_row[:, 2]), dim=1) + (
                    _SHED_DISTANCE_FRACTION
                    * self.settings.freestream
                    * self.settings.aerodynamic_dt
                    * self.v_inf[None, None, :]
                    / torch.linalg.vector_norm(self.v_inf)
                )
            # corners front-left, front-right, back-right, back-left, as everywhere else
            nascent_rings = torch.stack(
                (last_row[:, 3], last_row[:, 2], newest_front[:, 1], newest_front[:, 0]),
                dim=1,
            )
            if _KERNEL_V3:
                nascent_kwargs = {"core_fraction": 0.0, "reference_length": 0.0}
            else:
                # No core: this row is part of the influence matrix and its back leg must
                # cancel the bound trailing-edge leg exactly there.  It never leaves the edge,
                # so it cannot come close to a collocation point.
                nascent_kwargs = {
                    "core_fraction": 1.0e-6,
                    "reference_length": self._legacy_reference_length,
                }
            # (points, strips, 3): the kernel has already summed the four legs
            nascent_unit = native_ring_velocity_expanded(
                geometry.collocation, nascent_rings, **nascent_kwargs
            )
            nascent_normal = torch.sum(
                nascent_unit * geometry.normals[:, None, :], dim=2
            )                                                      # (P, ns)
            system_aic = aic.clone()
            system_aic[:, (chord_panels - 1) * span_panels :] += nascent_normal
        particle_velocity = (
            trial.particle_field.velocity_at_cuda(geometry.collocation)
            if trial.particle_field.n
            else torch.zeros_like(geometry.collocation)
        )
        if _ZERO_INDUCED in {"particles", "both"}:
            particle_velocity = torch.zeros_like(particle_velocity)
        if _ZERO_INDUCED in {"wake", "both"}:
            wake_velocity = torch.zeros_like(wake_velocity)
            if nascent_normal is not None:
                # the attribution diagnostic counts the nascent row as wake
                nascent_normal = torch.zeros_like(nascent_normal)
                system_aic = aic
        rhs = torch.sum(
            (geometry.collocation_velocity - self.v_inf - wake_velocity - particle_velocity)
            * geometry.normals,
            dim=1,
        )
        # Plan V3 stage 2: the leading-edge ring chains as convected by the previous step are
        # known sources here.  Two per-step fields the load evaluations read further down.
        hirato = trial.lev_chain is not None
        self._lev_field = None
        self._lev_pseudovortex = None
        self._lev_bound_extra = None
        self._step_suction_panel = None
        self._step_suction_replace_mask = None
        chain_normal_at_le = None
        split_from_rhs = None
        hirato_chain_split = None
        hirato_pv_norm = None
        hirato_pseudovortex = None
        ns, nc = self.settings.spanwise_panels, self.settings.chordwise_panels
        if hirato:
            if _LEV_HIRATO_V2:
                # Clean hirato_v2 (Stage 1): split ONLY continuing strips, put ONLY the split
                # free LEV chain (NO pseudovortex!) into `rhs` so the coupled solve carries the
                # pseudovortex strictly ONCE on the LHS.  Evaluate the zero-shedding pre-solve
                # (`gamma_pre`) at dGamma_L = 0 (continuing strips hold `chain.prev`).
                lev_core_rhs = self._lev_rhs_core_radius(geometry)
                was_shedding_pre = lev_hirato.continuing(trial.lev_chain)
                hirato_pseudovortex = lev_hirato.pseudovortex_ring(
                    geometry.leading_edge, lev_hirato.bound_rear(geometry.rings, nc)
                )
                if bool((trial.lev_chain.count > 0).any().item()):
                    if bool(was_shedding_pre.any().item()):
                        hirato_chain_split, split_from_rhs = lev_hirato.split_row0(
                            trial.lev_chain,
                            geometry.leading_edge,
                            was_shedding_pre,
                            max_length=self._lev_split_max_length(geometry),
                        )
                    else:
                        hirato_chain_split = trial.lev_chain
                    live_rings, live_gamma = lev_hirato.flat_live(hirato_chain_split)
                    live_cores = self._lev_live_core_radii(
                        hirato_chain_split,
                        lev_core_rhs,
                        row_age_offset=1,
                        continuing_mask=was_shedding_pre,
                    )
                    chain_velocity = self._ring_velocity(
                        geometry.collocation,
                        live_rings,
                        live_gamma,
                        rough=False,
                        core_radius=live_cores,
                        free=True,
                    )
                    rhs = rhs - torch.sum(chain_velocity * geometry.normals, dim=1)
                    chain_normal_at_le = torch.sum(
                        chain_velocity[:ns] * geometry.normals[:ns], dim=1
                    )
                else:
                    hirato_chain_split = trial.lev_chain
                if bool(was_shedding_pre.any().item()):
                    edge_pre = lev_hirato.strip_pairs(geometry.leading_edge)
                    nascent_cont = lev_hirato.nascent_row(split_from_rhs, edge_pre)
                    nascent_cont_norm = self._ring_unit_normal(
                        geometry, nascent_cont, core_radius=lev_core_rhs, free=True
                    )
                    hirato_pv_norm = self._ring_unit_normal(
                        geometry, hirato_pseudovortex, core_radius=lev_core_rhs, free=False
                    )
                    held_prev = torch.where(
                        was_shedding_pre,
                        trial.lev_chain.prev,
                        torch.zeros_like(trial.lev_chain.prev),
                    )
                    rhs_pre = rhs - (
                        nascent_normal + nascent_cont_norm + hirato_pv_norm
                    ) @ held_prev
                else:
                    rhs_pre = rhs
                gamma_pre = torch.linalg.solve(system_aic, rhs_pre)
            else:
                rhs = rhs - torch.sum(
                    self._lev_chain_velocity(geometry.collocation, trial.lev_chain) * geometry.normals,
                    dim=1,
                )
                gamma_pre = torch.linalg.solve(system_aic, rhs)
        else:
            gamma_pre = torch.linalg.solve(system_aic, rhs)
        chord = 0.5 * (
            torch.linalg.vector_norm(
                geometry.trailing_edge[:-1] - geometry.leading_edge[:-1], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.trailing_edge[1:] - geometry.leading_edge[1:], dim=1
            )
        )
        first_dx = 0.5 * (
            torch.linalg.vector_norm(
                geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
            )
            + torch.linalg.vector_norm(
                geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
            )
        )
        theta = torch.acos(torch.clamp(1.0 - 2.0 * first_dx / chord, -1.0, 1.0))
        if _LESP_UREL:
            le_vel_strip = 0.5 * (
                geometry.leading_velocity[:-1] + geometry.leading_velocity[1:]
            )
            lesp_speed = torch.linalg.vector_norm(
                self.v_inf[None, :] - le_vel_strip, dim=1
            ).clamp(min=1.0e-12)
        else:
            lesp_speed = self.settings.freestream
        scale = chord * lesp_speed * (theta + torch.sin(theta)) / LESP_FACTOR
        lesp_pre_3d = -gamma_pre[:ns] / scale
        # Single separation owner (plan §8.3): the 3D actual-surface LESP is
        # the ONLY truth for whether a strip is separated.
        surface_separated = unified_separation_mask(
            lesp_pre_3d, self.settings.effective_lesp_crit
        )
        if _LESP_MASK_OFF:
            surface_separated = torch.zeros_like(surface_separated)
        alpha, alpha_rate, heave, _, _ = self._dvm_kinematics(geometry, trial)
        # Single separation owner (plan §8.3): the 3D mask computed above is
        # passed INTO the source bank as a gate on its own 2D LESP trigger,
        # so the bank can never release a strip the 3D solve says is
        # attached — its 2D strip-theory LESP (computed from the bank's own
        # 2D wake, not the 3D ring solve) no longer deposits particles or
        # feeds gamma_lev/TEV on attached strips.  A strip the 3D solve
        # separates but whose 2D LESP is still subcritical stays unshed; it
        # is a REAL continuing release only if it already carries LE
        # circulation, otherwise it is merely pinned at its free A0 with no
        # LEV ever shed (see the release-flow diagnostics below, audit R3).
        # Forcing the coupled release of a subcritical 2D section was
        # rejected: it pins the section's LESP at sign*crit from below and
        # injects opposite-signed circulation (CT diverges). The
        # bank's un-gated 2D opinion survives only as the returned
        # ``raw_shed_lev`` diagnostic (counted in ``release_gate_overrides``
        # below).
        trial.alpha_previous = alpha.clone()
        particle_start = trial.particle_field.n
        if hirato:
            # Plan V3 stage 2: no 2-D source bank and no ribbon deposit.  The strength of the
            # leading-edge vortex is solved with the bound circulation further down and the
            # sheet is a chain of rings, not particles.
            source_result = None
            gamma_lev = torch.zeros(ns, device=self.device, dtype=torch.float64)
            released = surface_separated.clone()
        else:
            source_result = trial.source_bank.step(
                alpha,
                alpha_rate,
                heave,
                node_topology_from_cell_count=ns,
                cell_release_mask=surface_separated,
            )
        # P1: on the joint path the ribbon deposit moves AFTER the solve
        # (newborn strength from the 3D system); the legacy path keeps the
        # original order so the newborn-particle induction below matches
        # the deposited particles exactly.
        joint_solve_early = bool(
            getattr(self.settings, "joint_separation_solve", False)
        )
        if not joint_solve_early and not hirato:
            gamma_lev, released = self._deposit_dvm_ribbon(
                geometry, trial, source_result
            )
        if trial.particle_field.n > particle_start:
            newborn_velocity = trial.particle_field.velocity_at_cuda(
                geometry.collocation,
                source_start=particle_start,
                source_stop=trial.particle_field.n,
            )
            rhs = rhs - torch.sum(newborn_velocity * geometry.normals, dim=1)
        # Plan V3 (P0-B): the nascent row's leading-edge part is a known strength.
        system_rhs = rhs
        if nascent_normal is not None:
            system_rhs = rhs - nascent_normal @ gamma_lev
        # Reconcile the source bank's release mask with the 3D truth (plan
        # §8.3): pin_active IS the 3D mask — never the union of two owners.
        # Because the 3D mask gated the bank's release above, ``released``
        # can only fire on separated strips, so the conflict count (releases
        # without the owner's sanction) is a wiring guard that must stay
        # zero: any nonzero value means a second owner is deciding
        # separation again.  Separated strips the bank has not shed are NOT
        # conflicts — they stay pinned; that is continuing release only when
        # LEV circulation exists on the strip, otherwise the strip was never
        # shed at all (see release-flow diagnostics below, audit R3).
        # P1 single ownership: every 3D-separated strip carries a
        # newborn-LEV unknown in the joint solve (pin_active = the 3D
        # actual-surface mask itself).  The bank cannot veto a release any
        # more -- its role below is placement; the wiring guard stays and
        # is zero by construction.
        pin_active = surface_separated
        release_owner_conflicts = torch.zeros_like(surface_separated)
        # Observability (not a gate): how often the single-owner gate
        # suppressed a 2D strip-theory LESP release vote this step (the
        # bank wanted to shed a strip the 3D solve says is attached).
        if hirato:
            # there is no second owner: the 3-D mask is the whole release decision
            raw_2d_release = surface_separated.clone()
        else:
            raw_2d_release = source_result["raw_shed_lev"][:ns]
        release_gate_overrides = int(
            torch.count_nonzero(raw_2d_release & ~surface_separated).item()
        )
        # Release-flow observability (audit R3): decompose the two owners'
        # opinions and the strip separation transitions, and record whether
        # the separated strips actually carry LEV circulation.  A
        # 3D-separated strip whose 2D LESP is subcritical is only a REAL
        # "continuing release" if LE circulation exists on it; with none
        # (e.g. IZRA-15-090: separated_strip_count>0, lev_release=0,
        # particles=0 for the whole run) the strip is merely held pinned at
        # its free A0 and no LEV ever existed.
        previous_separated = (
            committed.separated_previous
            if committed.separated_previous is not None
            else torch.zeros_like(surface_separated)
        )
        if hirato:
            has_existing_lev_circulation = committed.lev_chain.count > 0
        else:
            has_existing_lev_circulation = (
                torch.count_nonzero(committed.source_bank.lg[:ns], dim=1) > 0
            )
        release_3d_only_count = int(
            torch.count_nonzero(surface_separated & ~raw_2d_release).item()
        )
        release_2d_only_count = release_gate_overrides
        newly_separated_count = int(
            torch.count_nonzero(surface_separated & ~previous_separated).item()
        )
        continuing_separated_count = int(
            torch.count_nonzero(surface_separated & previous_separated).item()
        )
        strips_with_existing_lev_circulation = int(
            torch.count_nonzero(has_existing_lev_circulation).item()
        )
        # P1 joint solve (P0_REVIEW_AND_RIGID_DIAGNOSTIC_6527A6F item 2 /
        # LOAD_REPRODUCTION_DIAGNOSIS §3): the separated strips' leading
        # rows KEEP the full solid-surface no-penetration equation; a
        # newborn-LEV unknown per active strip uses the real Gaussian ribbon
        # influence. The pin below applies to BOUND circulation. The physical
        # post-deposition gate must also pass before returning a trial.
        le_indices = torch.arange(ns, device=self.device, dtype=torch.int64)
        active_indices = le_indices[pin_active]
        lesp_pre = -gamma_pre[:ns] / scale
        gamma_lev_newborn = torch.zeros(ns, device=self.device, dtype=torch.float64)
        # E1-FLUX diagnostics must exist on every path (attached / legacy /
        # joint), otherwise the diagnostics dict raises UnboundLocalError.
        flux_shed_max = 0.0
        le_flux_residual = torch.zeros((), device=self.device, dtype=torch.float64)
        le_window_steps = int(
            getattr(self.settings, "le_constraint_window_steps", 0)
        )
        le_window_fraction = 0.0
        joint_solve = bool(getattr(self.settings, "joint_separation_solve", False))
        # B1: the load derivative operators (Mf1 and the substep Mf2_1) must
        # linearize the SAME bound system that produced gamma.  The assembler
        # receives `aic` and stores it for velocity_force(), so this is the
        # single place to redirect them.  Off by default: the production path
        # keeps the historical unconstrained operator bit-for-bit.  Under plan V3
        # (P0-B) the system that produced gamma includes the nascent row, so that
        # is the matrix handed over (`system_aic` is `aic` on the legacy path).
        load_derivative_aic = system_aic
        load_derivative_scope = "unconstrained"
        use_active_set = bool(
            getattr(self.settings, "load_derivative_active_set", False)
        )
        hirato_solution = None
        if hirato:
            # Plan V3 stage 2: the nascent leading-edge vortex of every shedding strip is an
            # unknown of the circulation solve, next to the bound circulation.
            hirato_solution = self._hirato_solve(
                geometry, trial, system_aic, nascent_normal, rhs, scale, lesp_pre_3d,
                split_from_rhs, chain_normal_at_le,
                chain_split=hirato_chain_split,
                pv_norm_cached=hirato_pv_norm,
                pseudovortex_cached=hirato_pseudovortex,
            )
            gamma = hirato_solution["gamma"]
            # the active set may have grown while iterating (the paper's flowchart)
            pin_active = hirato_solution["active"]
            active_indices = le_indices[pin_active]
            residual_rows = torch.abs(hirato_solution["residual"])
            imposed = hirato_solution.get("imposed_strips")
            if _LEV_SEPARATED_STATE and imposed is not None and bool(imposed.any().item()):
                # A separated leading edge does not satisfy no-penetration, so the rows this model
                # imposes a circulation on are not equations: their residual is recorded and not
                # gated, the same downgrade the prescribed-strength flux mode documents.
                residual_rows = residual_rows.clone()
                residual_rows[:ns] = torch.where(
                    imposed, torch.zeros_like(residual_rows[:ns]), residual_rows[:ns]
                )
            residual_max = torch.max(residual_rows)
            residual_scale = torch.clamp(torch.max(torch.abs(rhs)), min=1.0)
            if bool((residual_max > self.settings.gate_rtol * residual_scale).item()):
                raise RuntimeError(
                    "native V5M leading-edge vortex: no-penetration rows failed: "
                    f"{float(residual_max.item()):.9e} m/s"
                )
            load_derivative_scope = "hirato_coupled"
        elif joint_solve and active_indices.numel():
            # P1-2 REAL-operator joint solve: the newborn-LEV basis is the
            # induction of the material vortex-sheet ribbon that will
            # actually be deposited (same frontier geometry, spacing and
            # Gaussian sigma as _deposit_dvm_ribbon), evaluated per active
            # strip at unit strength on a scratch field -- NOT the AIC LE
            # surrogate (review: 105% off, false acceptance).
            k = int(active_indices.numel())
            panel_count = int(aic.shape[0])
            crit = self.settings.effective_lesp_crit
            pin_target_gamma = -torch.sign(lesp_pre[active_indices]) * crit * scale[active_indices]
            # Pinned bound system (LE rows pinned at the critical LESP in
            # Gamma units; every other row keeps no-penetration).
            pinned_aic = aic.clone()
            pinned_aic[active_indices] = 0.0
            pinned_aic[active_indices, active_indices] = 1.0
            pinned_rhs0 = rhs.clone()
            pinned_rhs0[active_indices] = pin_target_gamma
            # Ribbon frontier/geometry inputs (identical to the deposit).
            alpha_rate, _, _, node_x, node_normal = self._dvm_kinematics(
                geometry, trial
            )
            del alpha_rate
            frontier = self._lev_sheet_frontier(
                geometry, trial, source_result, node_x, node_normal
            )
            smoothing = (
                self.settings.dvm_smoothing_radius_chord
                * float(torch.mean(trial.reference_cell_chord).item())
            )
            spacing = (
                self.settings.dvm_target_spacing_chord
                * float(torch.mean(trial.reference_cell_chord).item())
            )

            ribbon_basis = torch.zeros(
                (panel_count, k), device=self.device, dtype=torch.float64
            )
            scratch = CudaParticleField(
                self.settings.particle_capacity, device=self.settings.device
            )
            for j, strip in enumerate(active_indices.tolist()):
                le_line = geometry.leading_edge[strip : strip + 2]
                front_line = frontier[strip : strip + 2]
                scratch.n = 0
                if self.settings.le_open_deposit:
                    scratch.add_open_filament_particles(
                        le_line,
                        -torch.ones(
                            1, device=self.device, dtype=torch.float64
                        ),
                        smoothing_radius=smoothing,
                        target_spacing=spacing,
                        birth_step=trial.step,
                    )
                else:
                    scratch.add_connected_ribbon_particles(
                        le_line,
                        front_line,
                        -torch.ones(
                            1, device=self.device, dtype=torch.float64
                        ),
                        smoothing_radius=smoothing,
                        target_spacing=spacing,
                        birth_step=trial.step,
                    )
                if scratch.n:
                    ribbon_basis[:, j] = torch.sum(
                        scratch.velocity_at_cuda(geometry.collocation)
                        * geometry.normals,
                        dim=1,
                    )
            # Linearity of the pinned solve: Gamma(gamma) = Gamma_0 +
            # sum_j gamma_j Gamma_j with Gamma_j = solve(pin, -b_j).
            gamma_0 = torch.linalg.solve(pinned_aic, pinned_rhs0)
            # ------------------------------------------------------------------
            # E1-FLUX (experimental, default OFF): prescribe the shed
            # circulation from the sharp-edge vorticity-flux law instead of
            # solving it from the LE no-penetration closure.
            #
            #     dGamma_shed = 0.5 * U_s^2 * dt     (Kaden/Didden slug model)
            #
            # Why: the L2-1 oracle showed the solved circulation is O(20-50)x
            # the flux estimate AND dt-independent (ratio doubles when dt
            # halves), so gamma_recon = Gamma/ds ~ 1/dt -- refining the time
            # step amplifies the near-LE sheet instead of resolving it.  The
            # flux law makes Gamma_shed proportional to dt and to U_s^2, hence
            # bounded by construction and the self-induction loop cannot close.
            #
            # U_s is taken from the UNCONSTRAINED pre-solve bound solution
            # (gamma_pre, LE row) divided by the first-cell chord: for a vortex
            # sheet the tangential velocity jump equals its strength, so this is
            # the discrete slip at separation, available without circularity.
            #
            # Consequence, recorded honestly: the LE no-penetration row is NOT
            # satisfied exactly in this mode (the sheet strength is prescribed,
            # not solved).  The residual is recorded as a diagnostic and the
            # strict deposited-flow gate is downgraded to finite checks here.
            # This is a different discrete shedding MODEL, not a relaxed gate
            # on the old model.
            flux_mode = bool(getattr(self.settings, "flux_consistent_shedding", False))
            le_flux_residual = torch.zeros((), device=self.device, dtype=torch.float64)
            if flux_mode:
                dx_le = 0.5 * (
                    torch.linalg.vector_norm(
                        geometry.rings[:ns, 3] - geometry.rings[:ns, 0], dim=1
                    )
                    + torch.linalg.vector_norm(
                        geometry.rings[:ns, 2] - geometry.rings[:ns, 1], dim=1
                    )
                )
                slip = torch.abs(gamma_pre[:ns]) / torch.clamp(dx_le, min=1e-30)
                shed_flux = 0.5 * slip**2 * self.settings.aerodynamic_dt
                newborn = (
                    torch.sign(lesp_pre[active_indices])
                    * shed_flux[active_indices]
                )
                # The prescribed sheet also induces at every collocation point,
                # so the bound circulation must be re-solved with that induction
                # moved to the RHS.  Non-LE rows then close EXACTLY, while the
                # LE rows stay pinned at the critical LESP and their residual
                # (the part this model does not enforce) is recorded.
                pinned_rhs_flux = pinned_rhs0.clone()
                non_le_rows = torch.ones(
                    panel_count, device=self.device, dtype=torch.bool
                )
                non_le_rows[active_indices] = False
                sheet_induction = ribbon_basis @ newborn
                pinned_rhs_flux = pinned_rhs_flux - (
                    non_le_rows.to(torch.float64) * sheet_induction
                )
                gamma = torch.linalg.solve(pinned_aic, pinned_rhs_flux)
                gamma_lev_newborn[active_indices] = newborn
                le_flux_residual = torch.max(
                    torch.abs(
                        aic[active_indices] @ gamma
                        + ribbon_basis[active_indices, :] @ newborn
                        - rhs[active_indices]
                    )
                )
                flux_shed_max = float(torch.max(torch.abs(newborn)).item())
                if not bool(torch.isfinite(newborn).all().item()):
                    raise FloatingPointError(
                        "E1-FLUX prescribed a non-finite shed circulation"
                    )
            else:
                flux_shed_max = 0.0
            basis_rhs = -ribbon_basis  # (P, k)
            # The pin constrains the ABSOLUTE bound value (carried by
            # gamma_0); basis responses must keep the pinned rows at zero.
            basis_rhs[active_indices, :] = 0.0
            if not flux_mode:
                pinned_batch = pinned_aic[None, :, :].expand(k, -1, -1).contiguous()
                gammas_j = torch.linalg.solve(
                    pinned_batch, basis_rhs.T.contiguous()[:, :, None]
                ).squeeze(-1)  # (k, P): row j = Gamma_j
                # LE-row closure with the REAL ribbon operator: for each active
                # strip s: aic[le_s].Gamma(gamma) + sum_j b[le_s, j] gamma_j
                # = rhs[le_s].  (The pinned rows themselves do not enforce the
                # LE no-penetration; the newborn sheet does, exactly.)
                le_rows = aic[active_indices]  # (k, P)
                closure_matrix = (
                    le_rows @ gammas_j.T + ribbon_basis[active_indices, :]
                )  # (k, k)
                closure_rhs = rhs[active_indices] - le_rows @ gamma_0
                # E1-WINDOW: drop the OLD particle induction from the LE rows
                # only.  rhs carries -n.(v_inf + wake + particles), so removing
                # the old particles means ADDING +n.v_old there.
                if le_window_steps > 0 and trial.particle_field.n:
                    field = trial.particle_field
                    live = field.n
                    cutoff = trial.step - le_window_steps
                    old_mask = field.birth_step[:live] < cutoff
                    old_count = int(torch.count_nonzero(old_mask).item())
                    if old_count:
                        le_window_fraction = float(old_count) / float(live)
                        scratch_old = CudaParticleField(
                            max(old_count, 1), device=self.settings.device
                        )
                        scratch_old.pos[:old_count].copy_(
                            field.pos[:live][old_mask]
                        )
                        scratch_old.gamma[:old_count].copy_(
                            field.gamma[:live][old_mask]
                        )
                        scratch_old.sigma[:old_count].copy_(
                            field.sigma[:live][old_mask]
                        )
                        scratch_old.circul[:old_count].copy_(
                            field.circul[:live][old_mask]
                        )
                        scratch_old.n = old_count
                        v_old = scratch_old.velocity_at_cuda(
                            geometry.collocation[active_indices]
                        )
                        closure_rhs = closure_rhs + torch.sum(
                            v_old * geometry.normals[active_indices], dim=1
                        )
                newborn = torch.linalg.solve(closure_matrix, closure_rhs)
                gamma = gamma_0 + newborn @ gammas_j
                gamma_lev_newborn[active_indices] = newborn
            if use_active_set:
                # Pinned operator: exact for the bound block (the pin RHS is
                # velocity-independent), but it omits the newborn-coupling
                # term B (E A^-1 B)^-1 E A^-1 of the augmented Schur
                # complement.  Declared, not hidden.
                load_derivative_aic = pinned_aic
                load_derivative_scope = "pinned_ignores_newborn_coupling"
        elif active_indices.numel():
            # legacy production path: pin row replaces the LE no-penetration
            separated_aic = system_aic.clone()
            separated_rhs = system_rhs.clone()
            source_a0 = source_result["A0"][:ns]
            separated_aic[active_indices] = 0.0
            separated_aic[active_indices, active_indices] = 1.0
            if _LE_PIN == "legacy":
                # Kept as the literal historical expression so the default path stays
                # bit-identical (see the FLUXV_LE_PIN note at the top of this module).
                separated_rhs[active_indices] = (-source_a0 * scale)[pin_active]
            else:
                # `gamma_pre` here is the attached solve BEFORE the pin; its sign is the
                # sign the leading-edge row carries when no strip is separated.
                sign_pre = torch.sign(gamma_pre[:ns])
                if _LE_PIN == "signed_mag":
                    pinned_value = sign_pre * torch.abs(source_a0) * scale
                else:  # "signed"
                    pinned_value = source_a0 * scale
                separated_rhs[active_indices] = pinned_value[pin_active]
            gamma = torch.linalg.solve(separated_aic, separated_rhs)
            if use_active_set:
                # Exact: d(gamma)/d(rhs) = separated_aic^-1 because the pin
                # target sign*crit*scale does not depend on velocity or
                # acceleration.
                load_derivative_aic = separated_aic
                load_derivative_scope = "separated_pinned_exact"
        else:
            gamma = torch.linalg.solve(system_aic, system_rhs)
        # Acceptance: joint path checks the FULL surface (LE included);
        # legacy path checks the retained rows exactly as before.
        if hirato:
            # the residual of all P no-penetration rows with the nascent leading-edge
            # vortex at its solved strength, the pinned leading-edge rows included
            total_residual = hirato_solution["residual"]
        else:
            total_residual = system_aic @ gamma - system_rhs
        if joint_solve and active_indices.numel():
            # REAL ribbon operator (per-unit-strength deposited-sheet
            # induction), consistent with the closure equations above.
            lev_induction = ribbon_basis @ gamma_lev_newborn[active_indices]
            total_residual = total_residual + lev_induction
        if joint_solve:
            residual_scale = torch.clamp(torch.max(torch.abs(rhs)), min=1.0)
            residual_max = torch.max(torch.abs(total_residual))
            if (
                bool(getattr(self.settings, "flux_consistent_shedding", False))
                or le_window_steps > 0
            ) and active_indices.numel():
                # E1-FLUX prescribes the shed strength, so the LE rows are not
                # equations of this model.  Every other row must still close
                # exactly; the LE residual lives in the diagnostics.
                non_le = torch.ones_like(total_residual, dtype=torch.bool)
                non_le[active_indices] = False
                non_le_max = torch.max(torch.abs(total_residual[non_le]))
                if bool(
                    (non_le_max > self.settings.gate_rtol * residual_scale).item()
                ):
                    raise RuntimeError(
                        "native V5M E1-FLUX: non-LE Neumann rows failed: "
                        f"{float(non_le_max.item()):.9e} m/s"
                    )
            elif bool((residual_max > self.settings.gate_rtol * residual_scale).item()):
                raise RuntimeError("native V5M full-surface Neumann rows failed")
        else:
            retained = torch.ones_like(rhs, dtype=torch.bool)
            if not hirato:
                retained[active_indices] = False
            if _LEV_SEPARATED_STATE and hirato_solution is not None:
                # a separated leading edge does not satisfy no-penetration: the imposed rows are
                # not equations, so they are excluded here as they are from the gate in the
                # hirato branch above
                imposed_rows = hirato_solution.get("imposed_strips")
                if imposed_rows is not None and bool(imposed_rows.any().item()):
                    retained[:ns] = retained[:ns] & ~imposed_rows
            retained_max = torch.max(torch.abs(total_residual[retained]))
            retained_scale = torch.clamp(torch.max(torch.abs(rhs[retained])), min=1.0)
            residual_max = retained_max
            if bool(
                (retained_max > self.settings.gate_rtol * retained_scale).item()
            ):
                raise RuntimeError("native V5M retained Neumann rows failed")
        # LESP of the BOUND LE circulation; on the joint path the newborn
        # excess is not part of LESP, on the legacy path it is the full pin.
        solved_lesp = -gamma[:ns] / scale
        pin_error = torch.zeros((), device=self.device, dtype=torch.float64)
        if hirato:
            # every shedding strip sits at the critical suction parameter, with its sign
            pin_error = torch.zeros((), device=self.device, dtype=torch.float64)
            if active_indices.numel():
                pin_sign = hirato_solution.get("pin_sign", torch.sign(lesp_pre_3d))
                pin_error = torch.max(
                    torch.abs(
                        solved_lesp[pin_active]
                        - pin_sign[pin_active] * self.settings.effective_lesp_crit
                    )
                )
            if bool((pin_error > 1.0e-6).item()) and not _LEV_FLUX:
                # the flux law prescribes the strength instead of pinning the suction
                # parameter, so a strip's solved suction parameter is expected to leave the
                # critical value; the departure stays in `lesp_pin_max_abs` and is not a gate
                # (note A24, section 5)
                raise RuntimeError("native V5M LESP pin failed")
        elif active_indices.numel() and not joint_solve:
            source_a0 = source_result["A0"][:ns]
            if _LE_PIN == "legacy":
                # The legacy convention, consistently: LESP is -Gamma/scale and the pin
                # demands that this equal the bank's (positive) A0.
                pin_error = torch.max(torch.abs(solved_lesp[active_indices] - source_a0[active_indices]))
            else:
                # Literature convention (Hirato 2019 eq. 6): A0 and the leading-edge
                # circulation share a sign, so the pin demands +Gamma_LE/scale == A0.
                # The expected value moves WITH the pin form; this is the same gate
                # expressed in the convention the arm uses, not a relaxed one.
                pin_error = torch.max(torch.abs(solved_lesp[active_indices] + source_a0[active_indices]))
            if bool((pin_error > 1.0e-6).item()):
                raise RuntimeError("native V5M LESP pin failed")
        if joint_solve and active_indices.numel():
            target_lesp = torch.sign(lesp_pre[active_indices]) * self.settings.effective_lesp_crit
            pin_error = torch.max(torch.abs(solved_lesp[active_indices] - target_lesp))
            if bool((pin_error > 1.0e-6).item()):
                raise RuntimeError("native V5M LESP pin failed")
        # The newborn LEV carries into the release/dep strength so the
        # source bank's role is PLACEMENT, not strength (joint-solve
        # ownership); strips the bank held subcritical still release the
        # physically required circulation the 3D solve found.
        if joint_solve:
            release_override = pin_active & (gamma_lev_newborn != 0.0)
            joint_source_result = dict(source_result)
            dimensional_unit = self.settings.freestream * trial.reference_cell_chord
            bank_strength = source_result["gamma_lev_new"][:ns]
            joint_strength = torch.where(
                release_override,
                gamma_lev_newborn / dimensional_unit,
                bank_strength,
            )
            joint_source_result["gamma_lev_new"] = torch.cat(
                (joint_strength, source_result["gamma_lev_new"][ns:])
            )
            joint_shed = source_result["shed_lev"].clone()
            joint_shed[:ns] = joint_shed[:ns] | release_override
            # Node mask must follow the strip override: a strip's ribbon
            # spans nodes [s, s+1]; without them the frontier falls back to
            # the leading edge and deposits coincident canceling edges
            # (review finding 2).  Node j is active when either adjacent
            # strip is releasing.
            node_override = torch.zeros(
                ns + 1, device=self.device, dtype=joint_shed.dtype
            )
            node_override[:-1] = node_override[:-1] | release_override.to(joint_shed.dtype)
            node_override[1:] = node_override[1:] | release_override.to(joint_shed.dtype)
            joint_shed[ns:] = joint_shed[ns:] | node_override
            joint_source_result["shed_lev"] = joint_shed
            gamma_lev, released = self._deposit_dvm_ribbon(
                geometry, trial, joint_source_result
            )
        elif hirato:
            # the chains take this step's nascent rows; a row that leaves is audited into the
            # retention ledger; the fields for the loads are set
            retention.update(self._hirato_commit(geometry, trial, hirato_solution))
            gamma_lev = hirato_solution["lam"]
            released = pin_active
            # The pseudovortex ring of a shedding strip is bound vorticity (the paper's
            # Eq. 15-17: the potential jump across the chord is the bound circulation together
            # with the leading-edge one, and the trailing edge sheds their sum).  So the
            # Kutta-Joukowski forces are integrated with the bound circulation plus the strip's
            # leading-edge strength; the unsteady term already carries the rate of that sum.
            self._lev_bound_extra = torch.zeros_like(gamma)
            self._lev_bound_extra.reshape(nc, ns)[:] = gamma_lev[None, :]
        elif not joint_solve:
            pass  # legacy path deposited before the solve (original order)

        algebraic_neumann_max = residual_max
        if joint_solve:
            # Check the actual uncommitted sources at the load time level,
            # before insertion of the next TE wake row. In particular, do
            # not reuse A[:, LE] @ q as evidence of the particle induction.
            newborn_velocity = trial.particle_field.velocity_at_cuda(
                geometry.collocation,
                source_start=particle_start,
                source_stop=trial.particle_field.n,
            )
            bound_velocity = self._ring_velocity(
                geometry.collocation, geometry.rings, gamma, rough=False
            )
            particle_velocity = particle_velocity + newborn_velocity
            deposited_residual = torch.sum(
                (self.v_inf + wake_velocity + particle_velocity + bound_velocity
                 - geometry.collocation_velocity) * geometry.normals,
                dim=1,
            )
            residual_max = torch.max(torch.abs(deposited_residual))
            residual_scale = torch.clamp(torch.max(torch.abs(rhs)), min=1.0)
            if not bool(torch.isfinite(deposited_residual).all().item()):
                raise FloatingPointError(
                    "native V5M joint deposited-flow Neumann residual is non-finite"
                )
            if bool(getattr(self.settings, "flux_consistent_shedding", False)) or le_window_steps > 0:
                # E1-FLUX prescribes the shed strength, and E1-WINDOW keeps the
                # far-field particles out of the LE rows; in both cases those
                # rows are not equations of this model and are not required to close; every
                # other row must.  The LE residual is recorded in the
                # diagnostics (flux_le_no_penetration_residual), and this is a
                # DIFFERENT discrete model, not a relaxed gate on the old one.
                non_le_mask = torch.ones_like(
                    deposited_residual, dtype=torch.bool
                )
                non_le_mask[active_indices] = False
                non_le_deposited = torch.max(
                    torch.abs(deposited_residual[non_le_mask])
                )
                if bool(
                    (non_le_deposited > self.settings.gate_rtol * residual_scale).item()
                ):
                    raise RuntimeError(
                        "native V5M E1-FLUX: non-LE deposited-flow rows failed: "
                        f"{float(non_le_deposited.item()):.9e} m/s"
                    )
            elif bool(
                (residual_max > self.settings.gate_rtol * residual_scale).item()
            ):
                raise RuntimeError(
                    "native V5M joint deposited-flow Neumann rows failed: "
                    f"actual={float(residual_max.item()):.9e} m/s, "
                    f"algebraic={float(algebraic_neumann_max.item()):.9e} m/s; "
                    "joint birth basis or solution differs from deposited flow"
                )
        tev_compensation = (
            torch.zeros_like(gamma_lev)
            if getattr(self.settings, "le_open_deposit_drop_tev_compensation", False)
            else gamma_lev
        )
        gamma_tev = gamma.reshape(nc, ns)[-1] + tev_compensation
        kelvin = torch.max(
            torch.abs(gamma_tev - gamma.reshape(nc, ns)[-1] - tev_compensation)
        )
        if bool((kelvin > self.settings.gate_rtol * torch.clamp(torch.max(torch.abs(gamma)), min=1.0)).item()):
            raise RuntimeError("native V5M joint TEV relation failed")

        if not joint_solve:
            bound_velocity = self._ring_velocity(
                geometry.collocation, geometry.rings, gamma, rough=False
            )
            if nascent_unit is not None:
                # the nascent row at its final strength, the state the step ends in
                wake_velocity = wake_velocity + torch.sum(
                    nascent_unit * gamma_tev[None, :, None], dim=1
                )
            if self._lev_field is not None:
                # the leading-edge ring chains as they stand, and the pseudovortex rings
                lev_core_load = self._lev_rhs_core_radius(geometry) if _LEV_HIRATO_V2 else None
                lev_field_cores = (
                    getattr(self, "_lev_field_cores", lev_core_load)
                    if _LEV_HIRATO_V2
                    else None
                )
                wake_velocity = wake_velocity + self._ring_velocity(
                    geometry.collocation,
                    *self._lev_field,
                    rough=False,
                    core_radius=lev_field_cores,
                    free=True,
                )
                wake_velocity = wake_velocity + self._ring_velocity(
                    geometry.collocation,
                    *self._lev_pseudovortex,
                    rough=False,
                    core_radius=lev_core_load,
                )
        external_flow = self.v_inf + wake_velocity + particle_velocity + bound_velocity
        if hirato and _LEV_HIRATO_V2 and not joint_solve:
            # Physical post-commit Neumann verification on the ACTUAL 1x-pseudovortex committed
            # vortex system: catches any double-counting, core mismatch, or split drift immediately.
            post_neumann = torch.sum(
                (external_flow - geometry.collocation_velocity) * geometry.normals, dim=1
            )
            imposed_rows = hirato_solution.get("imposed_strips")
            if _LEV_SEPARATED_STATE and imposed_rows is not None and bool(imposed_rows.any().item()):
                check_mask = torch.ones_like(post_neumann, dtype=torch.bool)
                check_mask[:ns] = ~imposed_rows
                post_neumann_max = torch.max(torch.abs(post_neumann[check_mask]))
            else:
                post_neumann_max = torch.max(torch.abs(post_neumann))
            residual_max = post_neumann_max
            post_scale = torch.clamp(torch.max(torch.abs(rhs)), min=1.0)
            if (
                int(retention.get("lev_truncate_ring_count", 0)) == 0
                and bool((post_neumann_max > self.settings.gate_rtol * post_scale).item())
            ):
                raise RuntimeError(
                    "native V5M hirato_v2 post-commit physical Neumann check failed: "
                    f"actual={float(post_neumann_max.item()):.9e} m/s, "
                    f"algebraic={float(algebraic_neumann_max.item()):.9e} m/s"
                )
        # Plan V3 stage 2: the circulation the loads are built from.  `gamma` stays the solved
        # bound circulation for the state, for the trailing-edge row and for the rate; the loads
        # see it plus the leading-edge strength of every shedding strip (its pseudovortex ring).
        gamma_load = gamma if self._lev_bound_extra is None else gamma + self._lev_bound_extra
        grid = gamma_load.reshape(nc, ns)
        dx = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 2] + geometry.rings[:, 3] - geometry.rings[:, 0] - geometry.rings[:, 1]), dim=1
        ).reshape(nc, ns)
        dy = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 1] + geometry.rings[:, 2] - geometry.rings[:, 0] - geometry.rings[:, 3]), dim=1
        ).reshape(nc, ns)
        if _LEV_HIRATO_V2_LOADS:
            # B5 (note A24): the leading-edge chordwise difference is one-sided, so a
            # chordwise-constant offset (the strip's Gamma_L, written into every row by
            # `_lev_bound_extra`) would appear there as a slope Gamma_L / dx_0.  The potential
            # jump already holds Gamma_L at the leading edge (Eq. 13, 16, 17), so its chordwise
            # derivative there is zero: the first row's difference carries the bound circulation
            # `gamma` alone.  The Gamma_L part stays in the spanwise differences and unsteady term.
            dx_gamma = lev_hirato.leading_edge_chordwise_difference(gamma.reshape(nc, ns), dx)
        else:
            dx_gamma = torch.empty_like(grid)
            dx_gamma[0] = grid[0] / dx[0]
            dx_gamma[1:] = (grid[1:] - grid[:-1]) / dx[1:]
        padded = torch.nn.functional.pad(grid, (1, 1))
        dy_gamma = (padded[:, 2:] - padded[:, :-2]) / (2.0 * dy)
        dy_gamma[:, 0] = grid[:, 0] / dy[:, 0]
        dy_gamma[:, -1] = -grid[:, -1] / dy[:, -1]
        tau_x = 0.5 * (
            geometry.rings[:, 2] + geometry.rings[:, 3] - geometry.rings[:, 0] - geometry.rings[:, 1]
        )
        tau_y = 0.5 * (
            geometry.rings[:, 1] + geometry.rings[:, 2] - geometry.rings[:, 0] - geometry.rings[:, 3]
        )
        tau_x = tau_x / torch.linalg.vector_norm(tau_x, dim=1, keepdim=True)
        tau_y = tau_y / torch.linalg.vector_norm(tau_y, dim=1, keepdim=True)
        gradient = tau_x * dx_gamma.reshape(-1, 1) + tau_y * dy_gamma.reshape(-1, 1)
        if self.settings.wake_history_mode == "bound_rate":
            # BOUND_RATE_MODEL identity (aero/v5m/loads.py, plan §8.7):
            # the author's weak-scheme dp_add (calc_fluid_force.m), the bound
            # circulation time derivative, bounded by construction.  The
            # strong-scheme Mf2_vec1 (MATERIAL_MODEL) diverges when every
            # wake row carries a large persistent LEV release.
            # One-step difference: at this point ``trial.gamma_bound`` still
            # holds the last COMMITTED bound circulation Gamma_(n-1) (it is
            # only overwritten with the new Gamma_n at the bottom of
            # propose()), while ``trial.gamma_previous`` lags one step further
            # behind (Gamma_(n-2)) because of the state-update ordering
            # (gamma_previous copies the pre-update gamma_bound).  Reading
            # gamma_previous here doubled the derivative's amplitude and
            # shifted its phase; the author's dp_add is (Gamma_n -
            # Gamma_(n-1))/dt exactly.
            mf2_history = (
                gamma - trial.gamma_bound
            ) / self.settings.aerodynamic_dt
            wake_vertex_velocity = None
        elif trial.wake_rings.shape[0]:
            from .q16_flux_v5m_author_loads import (
                material_ring_velocity_derivative_expanded,
            )

            wake_vertex_velocity = self._external_velocity(
                trial.wake_rings.reshape(-1, 3), geometry, trial
            ).reshape_as(trial.wake_rings)
            if trial.particle_field.n:
                wake_vertex_velocity = wake_vertex_velocity + trial.particle_field.velocity_at_cuda(
                    trial.wake_rings.reshape(-1, 3)
                ).reshape_as(trial.wake_rings)
            # The author's wake is a connected chain (generate_wake.m):
            # each row's trailing-edge-side corners ARE the previous row's
            # downstream corners, and the newest row is anchored to the
            # moving trailing edge with the sheet's structural velocity.
            # Free convection of detached front corners slides them under
            # the collocation points and poisons the wake-history term.
            chain_velocity = wake_vertex_velocity.reshape(-1, ns, 4, 3)
            chain_velocity[1:, :, 0] = chain_velocity[:-1, :, 3]
            chain_velocity[1:, :, 1] = chain_velocity[:-1, :, 2]
            chain_velocity[0, :, 0] = geometry.trailing_velocity[:-1]
            chain_velocity[0, :, 1] = geometry.trailing_velocity[1:]
            influence_rate = material_ring_velocity_derivative_expanded(
                geometry.collocation,
                geometry.collocation_velocity,
                trial.wake_rings,
                wake_vertex_velocity,
            )
            wake_velocity_rate = torch.sum(
                influence_rate * trial.wake_gamma[None, :, None], dim=1
            )
            wake_normal_rate = torch.sum(
                wake_velocity_rate * geometry.normals, dim=1
            )
            mf2_history = torch.linalg.solve(system_aic, -wake_normal_rate)
        else:
            wake_vertex_velocity = None
            mf2_history = torch.zeros_like(gamma)
        if hirato:
            # The paper's Eq. (17): the unsteady term carries d/dt of the bound circulation
            # together with that of the strip's leading-edge vortex, which is zero on a strip
            # that is not shedding.
            lev_rate = (
                trial.lev_chain.prev - committed.lev_chain.prev
            ) / self.settings.aerodynamic_dt
            mf2_history = mf2_history + lev_rate[None, :].expand(nc, ns).reshape(-1)
            self._step_suction_panel = self._hirato_suction(
                geometry,
                gamma,
                scale,
                chord,
                speed=lesp_speed,
                force_enable=(_HIRATO_PRESSURE_LOADS or _LEV_SEP_SUCTION),
            )
            if _LEV_SEP_SUCTION and not _HIRATO_PRESSURE_LOADS:
                self._step_suction_replace_mask = (trial.lev_chain.prev != 0.0) | pin_active
        author_load = self.author_load_assembler.assemble(
            structural_state=structural_state,
            geometry=geometry,
            aic=load_derivative_aic,
            gamma=gamma_load,
            external_flow=external_flow,
            gamma_gradient=gradient,
            mf2_history=mf2_history,
        )
        pressure = author_load.constant_pressure
        pressure_forces = pressure[:, None] * geometry.areas[:, None] * geometry.normals
        # Hirato's leading-edge suction (the paper's Eq. 20) acts on the front filament of
        # each strip's first panel; it is a chordwise force and enters the streamwise
        # coefficient, never the normal one.
        suction = (
            None
            if (_LEV_SEP_SUCTION and not (_LE_SUCTION or _HIRATO_PRESSURE_LOADS))
            else self._step_suction_panel
        )
        panel_forces = pressure_forces if suction is None else pressure_forces + suction
        extra_moment = (
            torch.zeros(3, device=self.device, dtype=torch.float64)
            if suction is None
            else torch.sum(
                torch.linalg.cross(
                    0.5 * (geometry.rings[:ns, 0] + geometry.rings[:ns, 1]), suction[:ns], dim=1
                ),
                dim=0,
            )
        )
        generalized = wp.clone(author_load.constant_generalized_force)
        # M1-3: publish the unified resolved packet from the SAME solver
        # state (shared NativeResolvedLoadAssembler; induction via this
        # solver's own kernels, wake state as advanced for this solve).
        # The pressure-quadrature totals above stay in place during
        # migration; the recorded discrepancy becomes the M1-4 hard gate.
        if getattr(self, "_skip_super_resolved_packet", False):
            packet = None
            te_leg_candidate = {}
            self._last_resolved_vs_quadrature_relative = 0.0
        else:
            if self._resolved_load_assembler is None:
                from fluxvortex.aero.v5m.load_packet import (
                    NativeResolvedLoadAssembler,
                )

                self._resolved_load_assembler = NativeResolvedLoadAssembler(
                    density=self.settings.density,
                    device=self.settings.device,
                    chordwise_panels=self.settings.chordwise_panels,
                    spanwise_panels=self.settings.spanwise_panels,
                    aerodynamic_dt=self.settings.aerodynamic_dt,
                    wake_history_mode="bound_rate",
                )
            resolved = self._resolved_load_assembler
            mids_flat = resolved.leg_midpoints_flat(geometry)
            if nascent_rings is None:
                solution_velocity_flat = (
                    self.v_inf[None, :].expand(mids_flat.shape[0], 3)
                    + self._ring_velocity(
                        mids_flat, geometry.rings, gamma, rough=False
                    )
                    + self._ring_velocity(
                        mids_flat, trial.wake_rings, trial.wake_gamma, rough=False
                    )
                )
            else:
                # Plan V3 (P0-A/B): the free wake as it stands at the end of the step,
                # nascent row included, with the wake core.
                core = self._core_radius(geometry)
                wake_pkt_cores = (
                    self._wake_core_radii(trial.wake_gamma, core, row_age_offset=1)
                    if trial.wake_rings.shape[0]
                    else core
                )
                solution_velocity_flat = (
                    self.v_inf[None, :].expand(mids_flat.shape[0], 3)
                    + self._ring_velocity(
                        mids_flat, geometry.rings, gamma, rough=False
                    )
                    + self._ring_velocity(
                        mids_flat,
                        trial.wake_rings,
                        trial.wake_gamma,
                        rough=False,
                        core_radius=wake_pkt_cores,
                        free=True,
                        mixed_precision=_RING_MIXED_PRECISION,
                    )
                    + self._ring_velocity(
                        mids_flat,
                        nascent_rings,
                        gamma_tev,
                        rough=False,
                        core_radius=core,
                    )
                )
            if trial.particle_field.n:
                solution_velocity_flat = (
                    solution_velocity_flat
                    + trial.particle_field.velocity_at_cuda(mids_flat)
                )
            if self._lev_field is not None:
                lev_core_load = self._lev_rhs_core_radius(geometry) if _LEV_HIRATO_V2_LOADS else None
                lev_field_cores = (
                    getattr(self, "_lev_field_cores", lev_core_load)
                    if _LEV_HIRATO_V2_LOADS
                    else None
                )
                solution_velocity_flat = solution_velocity_flat + self._ring_velocity(
                    mids_flat,
                    *self._lev_field,
                    rough=False,
                    core_radius=lev_field_cores,
                    free=True,
                    mixed_precision=_RING_MIXED_PRECISION,
                )
                solution_velocity_flat = solution_velocity_flat + self._ring_velocity(
                    mids_flat,
                    *self._lev_pseudovortex,
                    rough=False,
                    core_radius=lev_core_load,
                )
            packet_gamma = (
                gamma
                if (_LEV_HIRATO_V2_LOADS and self._lev_pseudovortex is not None)
                else gamma_load
            )
            packet_pseudo = (
                self._lev_pseudovortex[1]
                if (_LEV_HIRATO_V2_LOADS and self._lev_pseudovortex is not None)
                else None
            )
            packet = resolved.assemble_packet(
                geometry=geometry,
                gamma=packet_gamma,
                solution_velocity_flat=solution_velocity_flat.contiguous(),
                mf2_history=mf2_history,
                front_extra_force=self._step_suction_panel,
                front_suction_replace_mask=self._step_suction_replace_mask,
                pseudovortex_strip=packet_pseudo,
                component_ledger={
                    "author_constant_pressure": author_load.constant_pressure,
                },
            )
            # B2 diagnostic (U2, off by default): the TE leg of the last chordwise
            # row currently carries (Gamma_n - Gamma_{n-1}) from the bound-rate
            # history, while the wake row actually inserted carries
            # gamma_tev = Gamma_TE + gamma_lev.  Quantify the two candidates on
            # the SAME state instead of guessing which one the load should use.
            te_leg_candidate: dict[str, object] = {}
            if bool(getattr(self.settings, "te_leg_candidate_diagnostic", False)):
                packet_alt = resolved.assemble_packet(
                    geometry=geometry,
                    gamma=gamma,
                    solution_velocity_flat=solution_velocity_flat.contiguous(),
                    mf2_history=mf2_history,
                    te_leg_strength_override=gamma_tev.reshape(-1),
                )
                delta = packet_alt.total_force_I - packet.total_force_I
                force_norm = max(
                    float(torch.linalg.vector_norm(packet.total_force_I).item()), 1e-30
                )
                bound_rate_te = (
                    (gamma - trial.gamma_bound).reshape(
                        self.settings.chordwise_panels, self.settings.spanwise_panels
                    )[-1]
                )
                te_leg_candidate = {
                    "te_leg_baseline_strength_max_abs": float(
                        bound_rate_te.abs().max().item()
                    ),
                    "te_leg_shed_strength_max_abs": float(gamma_tev.abs().max().item()),
                    "te_leg_candidate_force_delta_norm_n": float(
                        torch.linalg.vector_norm(delta).item()
                    ),
                    "te_leg_candidate_force_delta_relative": float(
                        torch.linalg.vector_norm(delta).item() / force_norm
                    ),
                    "te_leg_candidate_delta_n": delta.detach().cpu().tolist(),
                }
            quadrature_force = torch.sum(panel_forces, dim=0)
            force_scale = max(
                float(torch.linalg.vector_norm(quadrature_force).item()), 1e-30
            )
            self._last_resolved_vs_quadrature_relative = float(
                torch.linalg.vector_norm(
                    packet.total_force_I - quadrature_force
                ).item()
                / force_scale
            )
        load = NativeV5MLoad(
            panel_positions=geometry.collocation.clone(),
            panel_forces=panel_forces.clone(),
            pressure=pressure.clone(),
            total_force=torch.sum(panel_forces, dim=0),
            total_moment=(
                torch.sum(
                    torch.linalg.cross(geometry.collocation, pressure_forces, dim=1), dim=0
                )
                + extra_moment
            ),
            surface_load_packet=packet,
        )

        rear = geometry.rings.reshape(nc, ns, 4, 3)[-1, :, 2:4]
        # P1 review (P1_REAL_B_REVIEW item 2, corrected per
        # P1_SHARED_EDGE_TIME_DISCRETE_DERIVATION_AND_REVIEW items 1-2):
        # the newborn TE wake row must CONNECT the current trailing edge to
        # the CONVECTED front edge of the previous newest row, not span a
        # fixed U*dt segment while the previous front is re-anchored back to
        # the TE.  Two independent defects had to be fixed together:
        #   (a) the TE re-anchor of the previous front ran unconditionally,
        #       so the old front never stayed at its convected position;
        #   (b) the convected front was written into the newborn row in the
        #       SAME order, producing a crossed quadrilateral whose shared
        #       edge had the same direction (cosine +1) instead of opposite.
        # Required endpoint identity for row k (newest) and k+1:
        #   r[k, 3] == r[k+1, 0],  r[k, 2] == r[k+1, 1]
        # so the shared edge vectors are exact negatives.
        downstream = rear + self.v_inf * self.settings.aerodynamic_dt
        # Plan V3 (P0-B): the connected chain is unconditional there; the nascent
        # row built for the solve IS the row inserted, corners and all.
        reconnect_wake = bool(
            getattr(self.settings, "joint_separation_solve", False)
        ) or nascent_rings is not None
        separate_rows = _SHED_DISTANCE_FRACTION > 0.0
        if separate_rows:
            # the shed-vortex distance of Kumar et al. (after Katz and Plotkin): the newborn row
            # is its own ring, `FLUXV_SHED_DISTANCE_FRACTION * U dt` long behind the trailing
            # edge, and every stored row keeps the position it was convected to.  No row is tied
            # to its neighbour and none is re-anchored to the wing.  Under TE_WAKE=v3 the row is
            # already built on the solve side (`nascent_rings`); on the legacy path it is built
            # here, with the length the papers state instead of one convection step.
            if nascent_rings is None:
                downstream = rear + (
                    _SHED_DISTANCE_FRACTION
                    * self.settings.freestream
                    * self.settings.aerodynamic_dt
                    * self.v_inf[None, :]
                    / torch.linalg.vector_norm(self.v_inf)
                )
                new_wake = torch.stack(
                    (rear[:, 1], rear[:, 0], downstream[:, 0], downstream[:, 1]), dim=1
                )
            else:
                new_wake = nascent_rings
            combined_rings = torch.cat((new_wake, trial.wake_rings), dim=0)
            combined_gamma = torch.cat((gamma_tev, trial.wake_gamma), dim=0)
        else:
            if trial.wake_rings.shape[0]:
                chain = trial.wake_rings.reshape(-1, ns, 4, 3)
                # Captured BEFORE any mutation: the previous newest row's
                # convected front edge.
                convected_front = torch.stack(
                    (chain[0, :, 0].clone(), chain[0, :, 1].clone()), dim=1
                )
                chain[1:, :, 0] = chain[:-1, :, 3]
                chain[1:, :, 1] = chain[:-1, :, 2]
                if reconnect_wake:
                    # Continuous material sheet: the downstream edge of the
                    # newborn row IS the previous row's convected front edge,
                    # traversed in the OPPOSITE direction:
                    #   new[2] = old[1],  new[3] = old[0]
                    # The previous front is NOT re-anchored to the wing TE.
                    downstream = torch.stack(
                        (convected_front[:, 1], convected_front[:, 0]), dim=1
                    )
                else:
                    # Legacy default, bit-for-bit: re-anchor the newest row's
                    # front to the current trailing edge and keep the U*dt
                    # downstream segment.
                    chain[0, :, 0] = rear[:, 1]
                    chain[0, :, 1] = rear[:, 0]
            if nascent_rings is not None:
                new_wake = nascent_rings
            else:
                new_wake = torch.stack(
                    (
                        rear[:, 1],
                        rear[:, 0],
                        downstream[:, 0],
                        downstream[:, 1],
                    ),
                    dim=1,
                )
            combined_rings = torch.cat((new_wake, trial.wake_rings), dim=0)
            combined_gamma = torch.cat((gamma_tev, trial.wake_gamma), dim=0)
        wake_capacity = self.settings.wake_max_rows * ns
        if combined_rings.shape[0] > wake_capacity:
            # Audit the truncated oldest rows BEFORE the slice (retention
            # ledger, plan §8.5): circulation and the closed-filament impulse
            # 0.5*Gamma*sum(x_i x x_j) of each dropped ring.
            dropped_rings = combined_rings[wake_capacity:]
            dropped_gamma = combined_gamma[wake_capacity:]
            retention["wake_truncate_ring_count"] = int(
                dropped_rings.shape[0]
            )
            retention["wake_truncate_circulation"] = float(
                dropped_gamma.sum().item()
            )
            corners = dropped_rings.reshape(-1, 4, 3)
            cross_sum = torch.cross(
                corners, torch.roll(corners, -1, dims=1), dim=2
            ).sum(dim=1)
            impulse = 0.5 * dropped_gamma[:, None] * cross_sum
            retention["wake_truncate_linear_impulse"] = [
                float(value) for value in impulse.sum(dim=0).tolist()
            ]
        else:
            retention["wake_truncate_ring_count"] = 0
            retention["wake_truncate_circulation"] = 0.0
            retention["wake_truncate_linear_impulse"] = [0.0, 0.0, 0.0]
        trial.wake_rings = combined_rings[:wake_capacity]
        trial.wake_gamma = combined_gamma[:wake_capacity]
        if (
            _LEV_GAUGE_REBASE
            and hirato
            and _LEV_HIRATO_V2
            and trial.lev_chain is not None
            and self.settings.wake_max_rows > 0
        ):
            ch_rb = trial.lev_chain
            cap_rb = min(int(self.settings.wake_max_rows), int(ch_rb.rings.shape[0]) - 1)
            if cap_rb >= 1:
                over_rb = ch_rb.count > cap_rb
                if bool(over_rb.any().item()):
                    g_drop = torch.where(
                        over_rb, ch_rb.gamma[cap_rb - 1], torch.zeros_like(ch_rb.prev)
                    )
                    new_count = torch.clamp(ch_rb.count, max=cap_rb)
                    row_idx = torch.arange(
                        ch_rb.rings.shape[0], device=ch_rb.rings.device
                    )[:, None]
                    valid_rb = row_idx < new_count[None, :]
                    new_gamma = torch.where(
                        valid_rb,
                        ch_rb.gamma - g_drop[None, :],
                        torch.zeros_like(ch_rb.gamma),
                    )
                    new_rings = torch.where(
                        valid_rb[:, :, None, None],
                        ch_rb.rings,
                        torch.zeros_like(ch_rb.rings),
                    )
                    new_prev = ch_rb.prev - g_drop
                    new_joined = ch_rb.joined & valid_rb[:-1] & valid_rb[1:]
                    trial.lev_chain = lev_hirato.LevChain(
                        rings=new_rings,
                        gamma=new_gamma,
                        count=new_count,
                        prev=new_prev,
                        joined=new_joined,
                    )
                    if trial.pseudovortex_previous is not None:
                        pv_r, pv_g = trial.pseudovortex_previous
                        trial.pseudovortex_previous = (pv_r, pv_g - g_drop)
                    if trial.wake_gamma.shape[0] > 0:
                        n_wrows = trial.wake_gamma.shape[0] // ns
                        trial.wake_gamma = (
                            trial.wake_gamma.reshape(n_wrows, ns) - g_drop[None, :]
                        ).reshape(-1)
                    retention["lev_gauge_rebase_max_abs"] = float(g_drop.abs().max().item())
        trial.gamma_previous = trial.gamma_bound.clone()
        trial.gamma_bound = gamma.clone()
        if _CONVECTION_V4 or _LEV_HIRATO_V2:
            trial.bound_rings_previous = geometry.rings.clone()
        trial.separated_previous = surface_separated.clone()
        trial.step += 1
        # One device->host read for the whole diagnostics block: 14 independent
        # .item() calls collapsed to 2 (one float64 stack, one int64 stack).
        released_total = trial.lev_strip_released_total
        batch, batch_ints = _read_device_scalars(
            (
                torch.max(torch.abs(lesp_pre_3d)),
                pin_error,
                kelvin,
                torch.sum(trial.gamma_bound),
                torch.sum(released_total) if released_total is not None else torch.zeros((), device=self.device, dtype=torch.float64),
                torch.max(torch.abs(released_total)) if released_total is not None else torch.zeros((), device=self.device, dtype=torch.float64),
                residual_max,
                algebraic_neumann_max,
                le_flux_residual,
                torch.max(torch.abs(trial.lev_strip_last_release)) if trial.lev_strip_last_release is not None else torch.zeros((), device=self.device, dtype=torch.float64),
                torch.max(torch.abs(gamma_lev_newborn)) if gamma_lev_newborn.numel() else torch.zeros((), device=self.device, dtype=torch.float64),
            ),
            (
                torch.count_nonzero(released),
                torch.count_nonzero(pin_active),
                torch.count_nonzero(release_owner_conflicts),
            ),
        )
        (
            lesp_pre_max_abs_host,
            pin_error_host,
            kelvin_host,
            bound_circulation_host,
            lev_released_host,
            lev_released_max_host,
            residual_max_host,
            algebraic_neumann_host,
            le_flux_residual_host,
            lev_strip_last_release_host,
            newborn_host,
        ) = batch
        lev_release_count_host, separated_strip_host, owner_conflicts_host = batch_ints
        lev_hirato_record = None
        if hirato:
            strength = hirato_solution["lam"]
            lev_hirato_record = {
                "active_strips": int(torch.count_nonzero(hirato_solution["active"]).item()),
                "iterations": int(hirato_solution["iterations"]),
                "strength_max_abs": float(strength.abs().max().item()),
                "strength_sum": float(strength.sum().item()),
                "chain_rows": int(trial.lev_chain.count.sum().item()),
                "detached_strips": int(hirato_solution.get("detached_strips", 0)),
                "suction_max_abs": (
                    float(self._step_suction_panel.abs().max().item())
                    if self._step_suction_panel is not None
                    else 0.0
                ),
            }
        trial.diagnostics = trial.diagnostics + (
            {
                "step": trial.step,
                "lev_model": (
                    ("hirato_v2_" + _LEV_SHED_LAW) if _LEV_HIRATO_V2 else "hirato"
                ) if hirato else "legacy",
                "lev_hirato": lev_hirato_record,
                "lev_release_count": lev_release_count_host,
                "separated_strip_count": separated_strip_host,
                "release_owner_conflicts": owner_conflicts_host,
                "release_gate_overrides": release_gate_overrides,
                # Release-flow decomposition (audit R3)
                "release_3d_only_count": release_3d_only_count,
                "release_2d_only_count": release_2d_only_count,
                "newly_separated_count": newly_separated_count,
                "continuing_separated_count": continuing_separated_count,
                "strips_with_existing_lev_circulation": (
                    strips_with_existing_lev_circulation
                ),
                "lesp_pre_max_abs": lesp_pre_max_abs_host,
                "lesp_pin_max_abs": pin_error_host,
                # L2-5: this is an ALGEBRAIC IDENTITY, not evidence.
                # gamma_tev is assembled as Gamma_TE + gamma_lev and then
                # subtracted from itself, so the residual is zero by
                # construction (measured: exactly 0.0 every step).  It used to
                # be reported as "kelvin_max_abs", which reads like a conservation
                # check and was cited as one.  Renamed to say what it is, and the
                # real cross-time ledger below is what should be quoted instead.
                "kelvin_identity_residual": kelvin_host,
                "bound_circulation_total": bound_circulation_host,
                "lev_released_total": (
                    lev_released_host if released_total is not None else 0.0
                ),
                "full_surface_neumann_max_abs": residual_max_host,
                "algebraic_neumann_max_abs": algebraic_neumann_host,
                "neumann_acceptance_scope": (
                    "deposited_flow_all_rows" if joint_solve else "legacy_retained_rows"
                ),
                "material_convection_scheme": (
                    "synchronous_rk3" if joint_solve else "legacy_split_frozen_ring_sources"
                ),
                "load_derivative_scope": load_derivative_scope,
                "flux_consistent_shedding": bool(
                    getattr(self.settings, "flux_consistent_shedding", False)
                ),
                "flux_shed_max": float(flux_shed_max),
                "le_constraint_window_steps": int(le_window_steps),
                "le_window_old_particle_fraction": float(le_window_fraction),
                "flux_le_no_penetration_residual": le_flux_residual_host,
                "te_leg_candidate": te_leg_candidate,
                "lev_strip_last_release_max_abs": (
                    lev_strip_last_release_host
                    if trial.lev_strip_last_release is not None
                    else 0.0
                ),
                "lev_strip_released_total_sum": (
                    lev_released_host if released_total is not None else 0.0
                ),
                "lev_strip_released_total_abs_max": lev_released_max_host,
                "newborn_lev_circulation_max_abs": newborn_host,
                "wake_ring_count": int(trial.wake_gamma.numel()),
                "particle_count": int(trial.particle_field.n),
                "cuda_float64": True,
                **retention,
            },
        )
        if committed.mutation_fingerprint() != parent_fingerprint:
            raise RuntimeError("native V5M trial mutated its committed parent")
        return NativeV5MProposal(
            parent_digest=parent_digest,
            parent_fingerprint=parent_fingerprint,
            trial_state=trial,
            load=load,
            generalized_force=generalized,
            author_load=author_load,
            preconvected_state=preconvected_state,
            preconvected_retention=preconvected_retention,
        )


#: A6 diagnostics (2026-11-23, default OFF): zero one induced-velocity term in the
#: collocation RHS so the load can be attributed to its sources by ablation --
#:   "particles" -> drop the deposited LEV/free particles,
#:   "wake"      -> drop the bound+wake ring influence,
#:   "both"      -> drop both (nothing but the kinematics and the freestream remains).
#: Ablation rather than a linear split because the LESP pinning downstream is nonlinear,
#: so a load-level superposition would not close.  Zeroing here means the reported load is
#: the load the wing WOULD carry without that source, with the same nonlinear path.
_ZERO_INDUCED = os.environ.get("FLUXV_ZERO_INDUCED", "").strip().lower()
if _ZERO_INDUCED not in {"", "particles", "wake", "both"}:
    raise ValueError(
        f"FLUXV_ZERO_INDUCED must be '', 'particles', 'wake' or 'both', got {_ZERO_INDUCED!r}"
    )

#: A18 diagnostic (2026-11-24, default OFF): force the 3-D separation mask to "attached"
#: on every strip, i.e. solve the SAME ring lattice with no leading-edge pin and no
#: leading-edge shedding.  It exists to price the pin at an incidence where no
#: leading-edge vortex is shed anyway: on the rigid plate at alpha = 5 deg the run ends
#: with 0 particles while 30 of 30 strips are pinned.  Against the attached thin-wing
#: value it says whether the lattice itself carries the excess normal force or the pin
#: does.  It removes the leading-edge separation model and is therefore never a
#: candidate for the production path.
_LESP_MASK_OFF = os.environ.get("FLUXV_LESP_MASK_OFF", "").strip().lower() in {
    "1", "true", "on", "yes",
}


class Q16NativeV5MOwner:
    def __init__(self, state: NativeV5MState) -> None:
        self.state = state

    def propose(
        self,
        solver: Q16NativeV5MSolver,
        structural_state: wp.array,
        structural_velocity: wp.array,
        preconvected: "NativeV5MState | None" = None,
        capture_preconvected: bool = False,
        inherited_retention: dict | None = None,
    ) -> NativeV5MProposal:
        return solver.propose(
            self.state,
            structural_state,
            structural_velocity,
            preconvected,
            capture_preconvected,
        )

    def commit(self, proposal: NativeV5MProposal) -> None:
        if proposal.parent_fingerprint != self.state.mutation_fingerprint():
            raise RuntimeError("native V5M proposal parent drift")
        if proposal.trial_state.step != self.state.step + 1:
            raise RuntimeError("native V5M commit must advance exactly one step")
        self.state = proposal.trial_state


__all__ = [
    "NATIVE_V5M_CONTRACT",
    "NativeV5MConfig",
    "NativeV5MGeometry",
    "NativeV5MLoad",
    "NativeV5MProposal",
    "NativeV5MState",
    "Q16NativePanelLoadTransfer",
    "Q16NativeV5MOwner",
    "Q16NativeV5MSolver",
    "Q16NativeV5MSurface",
    "assemble_native_ring_geometry",
    "native_aic",
    "native_ring_velocity_expanded",
]
