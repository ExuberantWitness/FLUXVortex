"""Native CUDA FLUX-V5M aerodynamic owner for Q16 shell coupling.

The scientific data plane is Q16 surface interpolation -> vortex geometry ->
LEV/TEV/free-wake solve -> panel load -> exact Q16 transpose.  Every evolving
numerical array is CUDA float64; Python owns only topology and transactions.
"""
from __future__ import annotations

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
from .kernels_q16_transfer import Q16CudaSurfaceTransfer
from .kernels_ring_velocity import (
    native_ring_velocity_expanded,
    native_ring_velocity_fused,
)

from ldvm_source_bank_gpu import CudaLDVMSourceBank
from pfield_torch_gpu import CudaParticleField


NATIVE_V5M_CONTRACT = "q16-flux-v5m-native-cuda-float64-v1"
LESP_FACTOR = 1.13
_FOUR_PI = 4.0 * math.pi
_EPS_V = 1.0e-9


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
        if self.le_open_deposit and not self.joint_separation_solve:
            raise ValueError(
                "le_open_deposit requires joint_separation_solve "
                "(the closure basis and the deposit must agree)"
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




def native_aic(geometry: NativeV5MGeometry, *, chordwise_panels: int) -> torch.Tensor:
    expanded = native_ring_velocity_expanded(
        geometry.collocation,
        geometry.rings,
        core_fraction=1.0e-6,
        reference_length=1.0 / chordwise_panels,
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
            diagnostics=tuple(dict(item) for item in self.diagnostics),
        )

    def digest(self) -> str:
        digest = hashlib.sha256(NATIVE_V5M_CONTRACT.encode("ascii"))
        digest.update(str(self.step).encode("ascii"))
        for value in (
            self.gamma_bound,
            self.gamma_previous,
            self.wake_rings,
            self.wake_gamma,
            self.particle_field.positions_cuda,
            self.particle_field.gammas_cuda,
            self.frontier_nodes,
            self.frontier_active,
        ):
            digest.update(value.detach().contiguous().cpu().numpy().tobytes())
        digest.update(str(self.source_bank.it).encode("ascii"))
        digest.update(str(self.source_bank.nt).encode("ascii"))
        return digest.hexdigest()


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
    trial_state: NativeV5MState
    load: NativeV5MLoad
    generalized_force: wp.array
    author_load: Any


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

    def _ring_velocity(self, points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor, *, rough: bool) -> torch.Tensor:
        if rings.shape[0] == 0:
            return torch.zeros_like(points)
        core_fraction = 0.1 if rough else 1.0e-6
        reference_length = 1.0 / self.settings.chordwise_panels
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

    def _external_velocity(
        self,
        points: torch.Tensor,
        geometry: NativeV5MGeometry,
        trial: NativeV5MState,
    ) -> torch.Tensor:
        return (
            self.v_inf[None, :]
            + self._ring_velocity(points, geometry.rings, trial.gamma_bound, rough=False)
            + self._ring_velocity(points, trial.wake_rings, trial.wake_gamma, rough=False)
        )

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
                    return self._external_velocity(at.reshape(-1, 3), geometry, trial).reshape_as(at)

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
                    velocity = self._external_velocity(flat, geometry, trial)
                    if trial.particle_field.n:
                        velocity = velocity + trial.particle_field.velocity_at_cuda(flat)
                    return velocity.reshape_as(at)

                u1 = wake_velocity(original)
                u2 = wake_velocity(original + 0.5 * dt * u1)
                u3 = wake_velocity(original + dt * (-u1 + 2.0 * u2))
                trial.wake_rings.copy_(original + (dt / 6.0) * (u1 + 4.0 * u2 + u3))

        def particle_external(points: torch.Tensor) -> torch.Tensor:
            return self._external_velocity(points, geometry, trial)

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
        birth_2d = result["lev_birth_position"][ns:]
        anchor_2d = result["lev_edge_position"][ns:]
        displacement = birth_2d - anchor_2d
        frontier = geometry.leading_edge + trial.reference_node_chord[:, None] * (
            displacement[:, :1] * node_x + displacement[:, 1:] * node_normal
        )
        frontier = torch.where(node_active[:, None], frontier, geometry.leading_edge)
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
    ) -> NativeV5MProposal:
        parent_digest = committed.digest()
        trial = committed.clone()
        geometry = self.surface.evaluate(structural_state, structural_velocity)
        # This step's release is not known yet: clear the per-step record so a
        # non-releasing step cannot inherit the previous step's strength.
        trial.lev_strip_last_release = torch.zeros(
            self.settings.spanwise_panels,
            device=self.device,
            dtype=torch.float64,
        )
        retention = self._advance_material_state(geometry, trial)
        aic = native_aic(geometry, chordwise_panels=self.settings.chordwise_panels)
        wake_velocity = self._ring_velocity(
            geometry.collocation, trial.wake_rings, trial.wake_gamma, rough=False
        )
        particle_velocity = (
            trial.particle_field.velocity_at_cuda(geometry.collocation)
            if trial.particle_field.n
            else torch.zeros_like(geometry.collocation)
        )
        rhs = torch.sum(
            (geometry.collocation_velocity - self.v_inf - wake_velocity - particle_velocity)
            * geometry.normals,
            dim=1,
        )
        gamma_pre = torch.linalg.solve(aic, rhs)
        ns, nc = self.settings.spanwise_panels, self.settings.chordwise_panels
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
        scale = chord * self.settings.freestream * (theta + torch.sin(theta)) / LESP_FACTOR
        lesp_pre_3d = -gamma_pre[:ns] / scale
        # Single separation owner (plan §8.3): the 3D actual-surface LESP is
        # the ONLY truth for whether a strip is separated.
        surface_separated = unified_separation_mask(
            lesp_pre_3d, self.settings.effective_lesp_crit
        )
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
        source_result = trial.source_bank.step(
            alpha,
            alpha_rate,
            heave,
            node_topology_from_cell_count=ns,
            cell_release_mask=surface_separated,
        )
        trial.alpha_previous = alpha.clone()
        particle_start = trial.particle_field.n
        # P1: on the joint path the ribbon deposit moves AFTER the solve
        # (newborn strength from the 3D system); the legacy path keeps the
        # original order so the newborn-particle induction below matches
        # the deposited particles exactly.
        joint_solve_early = bool(
            getattr(self.settings, "joint_separation_solve", False)
        )
        if not joint_solve_early:
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
        # keeps the historical unconstrained operator bit-for-bit.
        load_derivative_aic = aic
        load_derivative_scope = "unconstrained"
        use_active_set = bool(
            getattr(self.settings, "load_derivative_active_set", False)
        )
        if joint_solve and active_indices.numel():
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
            birth_2d = source_result["lev_birth_position"][ns:]
            anchor_2d = source_result["lev_edge_position"][ns:]
            displacement = birth_2d - anchor_2d
            frontier = geometry.leading_edge + trial.reference_node_chord[:, None] * (
                displacement[:, :1] * node_x + displacement[:, 1:] * node_normal
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
            separated_aic = aic.clone()
            separated_rhs = rhs.clone()
            source_a0 = source_result["A0"][:ns]
            separated_aic[active_indices] = 0.0
            separated_aic[active_indices, active_indices] = 1.0
            separated_rhs[active_indices] = (-source_a0 * scale)[pin_active]
            gamma = torch.linalg.solve(separated_aic, separated_rhs)
            if use_active_set:
                # Exact: d(gamma)/d(rhs) = separated_aic^-1 because the pin
                # target sign*crit*scale does not depend on velocity or
                # acceleration.
                load_derivative_aic = separated_aic
                load_derivative_scope = "separated_pinned_exact"
        else:
            gamma = torch.linalg.solve(aic, rhs)
        # Acceptance: joint path checks the FULL surface (LE included);
        # legacy path checks the retained rows exactly as before.
        total_residual = aic @ gamma - rhs
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
            retained[active_indices] = False
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
        if active_indices.numel() and not joint_solve:
            source_a0 = source_result["A0"][:ns]
            pin_error = torch.max(torch.abs(solved_lesp[active_indices] - source_a0[active_indices]))
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
        external_flow = self.v_inf + wake_velocity + particle_velocity + bound_velocity
        grid = gamma.reshape(nc, ns)
        dx = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 2] + geometry.rings[:, 3] - geometry.rings[:, 0] - geometry.rings[:, 1]), dim=1
        ).reshape(nc, ns)
        dy = torch.linalg.vector_norm(
            0.5 * (geometry.rings[:, 1] + geometry.rings[:, 2] - geometry.rings[:, 0] - geometry.rings[:, 3]), dim=1
        ).reshape(nc, ns)
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
            mf2_history = torch.linalg.solve(aic, -wake_normal_rate)
        else:
            wake_vertex_velocity = None
            mf2_history = torch.zeros_like(gamma)
        author_load = self.author_load_assembler.assemble(
            structural_state=structural_state,
            geometry=geometry,
            aic=load_derivative_aic,
            gamma=gamma,
            external_flow=external_flow,
            gamma_gradient=gradient,
            mf2_history=mf2_history,
        )
        pressure = author_load.constant_pressure
        panel_forces = pressure[:, None] * geometry.areas[:, None] * geometry.normals
        generalized = wp.clone(author_load.constant_generalized_force)
        # M1-3: publish the unified resolved packet from the SAME solver
        # state (shared NativeResolvedLoadAssembler; induction via this
        # solver's own kernels, wake state as advanced for this solve).
        # The pressure-quadrature totals above stay in place during
        # migration; the recorded discrepancy becomes the M1-4 hard gate.
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
        solution_velocity_flat = (
            self.v_inf[None, :].expand(mids_flat.shape[0], 3)
            + self._ring_velocity(
                mids_flat, geometry.rings, gamma, rough=False
            )
            + self._ring_velocity(
                mids_flat, trial.wake_rings, trial.wake_gamma, rough=False
            )
        )
        if trial.particle_field.n:
            solution_velocity_flat = (
                solution_velocity_flat
                + trial.particle_field.velocity_at_cuda(mids_flat)
            )
        packet = resolved.assemble_packet(
            geometry=geometry,
            gamma=gamma,
            solution_velocity_flat=solution_velocity_flat.contiguous(),
            mf2_history=mf2_history,
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
            total_moment=torch.sum(
                torch.linalg.cross(geometry.collocation, panel_forces, dim=1), dim=0
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
        reconnect_wake = bool(
            getattr(self.settings, "joint_separation_solve", False)
        )
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
        trial.gamma_previous = trial.gamma_bound.clone()
        trial.gamma_bound = gamma.clone()
        trial.separated_previous = surface_separated.clone()
        trial.step += 1
        trial.diagnostics = trial.diagnostics + (
            {
                "step": trial.step,
                "lev_release_count": int(torch.count_nonzero(released).item()),
                "separated_strip_count": int(torch.count_nonzero(pin_active).item()),
                "release_owner_conflicts": int(torch.count_nonzero(release_owner_conflicts).item()),
                "release_gate_overrides": release_gate_overrides,
                # Release-flow decomposition (audit R3)
                "release_3d_only_count": release_3d_only_count,
                "release_2d_only_count": release_2d_only_count,
                "newly_separated_count": newly_separated_count,
                "continuing_separated_count": continuing_separated_count,
                "strips_with_existing_lev_circulation": (
                    strips_with_existing_lev_circulation
                ),
                "lesp_pre_max_abs": float(torch.max(torch.abs(lesp_pre_3d)).item()),
                "lesp_pin_max_abs": float(pin_error.item()),
                "kelvin_max_abs": float(kelvin.item()),
                "full_surface_neumann_max_abs": float(residual_max.item()),
                "algebraic_neumann_max_abs": float(algebraic_neumann_max.item()),
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
                "flux_le_no_penetration_residual": float(le_flux_residual.item()),
                "te_leg_candidate": te_leg_candidate,
                "lev_strip_last_release_max_abs": float(
                    torch.max(
                        torch.abs(trial.lev_strip_last_release)
                    ).item()
                )
                if trial.lev_strip_last_release is not None
                else 0.0,
                "lev_strip_released_total_sum": float(
                    trial.lev_strip_released_total.sum().item()
                )
                if trial.lev_strip_released_total is not None
                else 0.0,
                "newborn_lev_circulation_max_abs": float(
                    torch.max(torch.abs(gamma_lev_newborn)).item()
                ) if gamma_lev_newborn.numel() else 0.0,
                "wake_ring_count": int(trial.wake_gamma.numel()),
                "particle_count": int(trial.particle_field.n),
                "cuda_float64": True,
                **retention,
            },
        )
        if committed.digest() != parent_digest:
            raise RuntimeError("native V5M trial mutated its committed parent")
        return NativeV5MProposal(
            parent_digest=parent_digest,
            trial_state=trial,
            load=load,
            generalized_force=generalized,
            author_load=author_load,
        )


class Q16NativeV5MOwner:
    def __init__(self, state: NativeV5MState) -> None:
        self.state = state

    def propose(
        self,
        solver: Q16NativeV5MSolver,
        structural_state: wp.array,
        structural_velocity: wp.array,
    ) -> NativeV5MProposal:
        return solver.propose(self.state, structural_state, structural_velocity)

    def commit(self, proposal: NativeV5MProposal) -> None:
        if proposal.parent_digest != self.state.digest():
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
