"""Rigid native FLUX-V5M: the SurfaceFrame drives the solver directly.

H3 (HANDOFF_IZRAELEVITZ2017_FIG14_SCHERER1968_20260827 §8): rigid
prescribed-motion paper cases (the Izraelevitz/Scherer rectangular wing) own
their geometry analytically, so the native V5M stepper must consume the
unified ``SurfaceFrame`` directly instead of the transitional Q16
``set_structural_state`` raw ``wp.array`` bridge.  This module provides that
path with no Q16 mesh, no Q16 structural elements and no fake wing:

- ``RigidV5MSurface`` stores the current SurfaceFrame and evaluates it into
  the native ``NativeV5MGeometry`` (the structural arguments are ignored —
  there is no structural state on this path, by construction);
- ``RigidNativeV5MSolver`` subclasses ``Q16NativeV5MSolver`` and inherits the
  frozen propose()/initialize() machinery unchanged (LEV/TEV/free-wake
  transactions, AIC, LESP release, Kelvin gate), swapping only the load
  assembler;
- ``RigidAuthorLoadAssembler`` is the SINGLE surface-load owner.  Its
  integrated panel forces, total force and world wrench follow the frozen
  Pterra oracle (``platform/warp_vpm/bing_joint_ptera_gpu.py``
  ``_calculate_loads``): Kutta-Joukowski vector forces on the panel ring
  filaments (effective shared strengths, full strength on leading-edge,
  trailing-edge and tip filaments, the trailing-edge row carrying the shed
  ``Gamma_now - Gamma_last``) plus the unsteady-Bernoulli
  ``rho dGamma/dt A n`` term.  The author pressure decomposition (lift1,
  wake-history Mf2, lift2, Mf2_1) is still assembled in full as an auditable
  ledger, but a scalar ``Delta p A n`` projection alone cannot carry the
  streamwise Kutta-Joukowski component (leading-edge suction / tilted lift)
  that produces thrust in attached flow — the exact regime where the
  pressure-only extraction missed the Scherer Fig.14 CT by ~0.25.

Every evolving numerical array stays CUDA float64; Python owns only topology
and transactions, exactly as on the Q16 path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import warp as wp

from fluxvortex.kinematics.frames import SurfaceFrame

from . import config as warp_config
from .q16_flux_v5m_author_loads import material_ring_velocity_derivative_expanded
from .q16_flux_v5m_native import (
    NativeV5MConfig,
    NativeV5MGeometry,
    NativeV5MLoad,
    Q16NativeV5MSolver,
    _require_cuda64,
    freestream_vector,
    native_ring_velocity_expanded,
)

RIGID_NATIVE_V5M_CONTRACT = "rigid-flux-v5m-native-cuda-float64-v1"


class RigidV5MSurface:
    """Native V5M surface fed directly by a rigid ``SurfaceFrame``.

    The unified kinematics chain produces the frame; ``set_frame`` stores it
    (validating topology/dtype/device/finiteness) and ``evaluate`` returns it
    wrapped as ``NativeV5MGeometry``.  The structural state arguments of the
    native surface protocol are accepted and ignored — the rigid path has no
    Q16 state anywhere, which is the point of H3.
    """

    def __init__(
        self, *, chordwise_panels: int, spanwise_panels: int, device: str
    ) -> None:
        self.nc = int(chordwise_panels)
        self.ns = int(spanwise_panels)
        self.device = torch.device(device)
        self._geometry: NativeV5MGeometry | None = None

    def set_frame(self, frame: SurfaceFrame) -> None:
        """Store one evaluated SurfaceFrame as the live native geometry."""
        if not isinstance(frame, SurfaceFrame):
            raise TypeError("RigidV5MSurface consumes SurfaceFrame instances")
        if (frame.chordwise_panels, frame.spanwise_panels) != (self.nc, self.ns):
            raise ValueError(
                f"surface frame topology {frame.chordwise_panels}x"
                f"{frame.spanwise_panels} differs from the rigid native "
                f"surface {self.nc}x{self.ns}"
            )
        count = self.nc * self.ns
        nodes = self.ns + 1
        fields = {
            "panel_rings_I": ((count, 4, 3), frame.panel_rings_I),
            "panel_ring_velocity_I": ((count, 4, 3), frame.panel_ring_velocity_I),
            "collocation_I": ((count, 3), frame.collocation_I),
            "collocation_velocity_I": ((count, 3), frame.collocation_velocity_I),
            "normals_I": ((count, 3), frame.normals_I),
            "areas": ((count,), frame.areas),
            "leading_edge_I": ((nodes, 3), frame.leading_edge_I),
            "trailing_edge_I": ((nodes, 3), frame.trailing_edge_I),
            "leading_velocity_I": ((nodes, 3), frame.leading_velocity_I),
            "trailing_velocity_I": ((nodes, 3), frame.trailing_velocity_I),
        }
        checked = {}
        for name, (shape, value) in fields.items():
            tensor = _require_cuda64(f"rigid frame {name}", value, self.device)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"rigid frame {name} has shape {tuple(tensor.shape)}, "
                    f"expected {shape}"
                )
            checked[name] = tensor
        normal_norm = torch.linalg.vector_norm(checked["normals_I"], dim=1)
        if bool(
            torch.any(
                torch.abs(normal_norm - 1.0)
                > 128.0 * torch.finfo(torch.float64).eps
            ).item()
        ):
            raise ValueError("rigid frame normals are not unit length")
        self._geometry = NativeV5MGeometry(
            rings=checked["panel_rings_I"],
            ring_velocity=checked["panel_ring_velocity_I"],
            collocation=checked["collocation_I"],
            collocation_velocity=checked["collocation_velocity_I"],
            normals=checked["normals_I"],
            areas=checked["areas"],
            leading_edge=checked["leading_edge_I"],
            trailing_edge=checked["trailing_edge_I"],
            leading_velocity=checked["leading_velocity_I"],
            trailing_velocity=checked["trailing_velocity_I"],
            quarter_points=None,
        )

    def evaluate(
        self, state: wp.array | None = None, velocity: wp.array | None = None
    ) -> NativeV5MGeometry:
        """Return the stored frame as native geometry (state args ignored)."""
        if state is not None or velocity is not None:
            raise TypeError(
                "RigidV5MSurface ignores structural state: the rigid native "
                "path must not receive a Q16 raw-state bridge"
            )
        if self._geometry is None:
            raise RuntimeError(
                "RigidV5MSurface.set_frame() must be called before evaluate()"
            )
        return self._geometry


@dataclass(frozen=True, slots=True, eq=False)
class RigidAuthorLoad:
    """Rigid native V5M endpoint load: Pterra-faithful forces + author ledger.

    ``panel_forces``/``total_force``/``total_moment``/``constant_generalized_force``
    are the integrated Kutta-Joukowski + unsteady-Bernoulli extraction of the
    frozen Pterra oracle (full vector panel forces — they include the
    streamwise thrust component that a pressure-only ``Delta p A n`` projection
    cannot represent).  The four author pressure parts remain assembled and
    summed into ``constant_pressure`` as an auditable decomposition ledger of
    the same solver state; ``constant_pressure`` is NOT the source of the
    integrated forces on this path.
    """

    lift1_pressure: torch.Tensor
    wake_history_pressure: torch.Tensor
    lift2_pressure: torch.Tensor
    mf2_1_pressure: torch.Tensor
    constant_pressure: torch.Tensor  # author audit sum of the four parts
    panel_forces: torch.Tensor  # Pterra KJ + unsteady vector forces
    total_force: torch.Tensor
    total_moment: torch.Tensor
    constant_generalized_force: wp.array  # (1, 6) world wrench [F; M]

    @property
    def total_pressure(self) -> torch.Tensor:
        return self.constant_pressure


# Tile budget for the KJ ring-influence sums, mirroring the solver's frozen
# ``Q16NativeV5MSolver._RING_TILE_BYTES`` behaviour so large grids tile over
# targets exactly like every other native velocity evaluation.
_RING_TILE_BYTES = 64 * 1024 * 1024


class RigidAuthorLoadAssembler:
    """Single surface-load owner for the rigid native V5M path.

    The integrated load is the frozen Pterra force extraction
    (``_calculate_loads`` in ``platform/warp_vpm/bing_joint_ptera_gpu.py``,
    itself a CUDA port of pterasoftware's unsteady UVLM loads):

    * Kutta-Joukowski vector forces on the four ring filaments of every
      panel, each with Pterra's effective shared strength — half the
      circulation difference with the neighbour across that edge, the full
      panel circulation on leading-edge, trailing-edge and tip filaments,
      and the freshly shed ``Gamma_now - Gamma_last`` on the trailing-edge
      row.  The relative velocity is the full solution velocity
      (freestream + wake + LEV particles + bound sheet) at the filament
      midpoint minus the moving-grid velocity, crossed with the filament
      vector: ``F = rho * Gamma_eff * (v_sol - v_grid) x dl``.
    * The unsteady-Bernoulli term ``rho * dGamma/dt * A * n`` per panel.
      The native rings are traversed clockwise about ``+n`` (the reference
      literature convention), while PterraSoftware's rings are
      counter-clockwise; converting between the two flips the circulation
      sign but not the normal, so Pterra's ``-rho dGamma/dt A n`` becomes
      ``+rho dGamma/dt A n`` here (pterasoftware issue #27).

    The author pressure decomposition (lift1 + wake-history Mf2 as the
    constant part, lift2 + Mf2_1 as the velocity part) is still assembled in
    full and returned in the ``RigidAuthorLoad`` ledger.

    The velocity split re-uses the exact same influence kernel
    (``native_ring_velocity_expanded``, core fraction 1e-6, reference length
    1/nc) the solver used to build ``external_flow`` at the collocation
    points, so the far field (freestream + wake + particles) is recovered
    exactly and only re-evaluated at the filament midpoints for the bound
    part; on-filament self/adjacent legs are suppressed by the kernel's core
    factor exactly as in Pterra's leg-center evaluations.  The generalized
    output is the 6-component world wrench [Fx, Fy, Fz, Mx, My, Mz] — there
    are no Q16 degrees of freedom to map into.
    """

    def __init__(
        self,
        *,
        density: float,
        device: str | torch.device,
        chordwise_panels: int | None = None,
        spanwise_panels: int | None = None,
        aerodynamic_dt: float | None = None,
        wake_history_mode: str = "bound_rate",
    ) -> None:
        self.density = float(density)
        self.device = torch.device(device)
        self.chordwise_panels = None if chordwise_panels is None else int(chordwise_panels)
        self.spanwise_panels = None if spanwise_panels is None else int(spanwise_panels)
        self.aerodynamic_dt = None if aerodynamic_dt is None else float(aerodynamic_dt)
        if wake_history_mode not in {"material", "bound_rate"}:
            raise ValueError(
                "wake_history_mode must be 'material' or 'bound_rate'"
            )
        self.wake_history_mode = wake_history_mode
        # Material-mode fallback for Gamma_last (the Fig.14 mandatory runner
        # uses bound_rate, where the mf2_history input IS dGamma/dt exactly).
        self._previous_gamma: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Pterra-faithful force extraction
    # ------------------------------------------------------------------
    def _panel_topology(self, geometry: NativeV5MGeometry) -> tuple[int, int]:
        count = int(geometry.normals.shape[0])
        ns = self.spanwise_panels
        if ns is None:
            ns = int(geometry.leading_edge.shape[0]) - 1
        if ns < 1 or count % ns:
            raise ValueError(
                "rigid native V5M grid topology is inconsistent with the "
                "load assembler topology"
            )
        nc = count // ns
        if self.chordwise_panels is not None and nc != self.chordwise_panels:
            raise ValueError(
                "rigid native V5M chordwise topology differs from the load "
                "assembler topology"
            )
        return nc, ns

    def _bound_velocity(
        self, points: torch.Tensor, rings: torch.Tensor, gamma: torch.Tensor, nc: int
    ) -> torch.Tensor:
        """Mirror of ``Q16NativeV5MSolver._ring_velocity(rough=False)``."""

        if rings.shape[0] == 0:
            return torch.zeros_like(points)
        core_fraction = 1.0e-6
        reference_length = 1.0 / nc
        full_bytes = points.shape[0] * rings.shape[0] * 3 * 8
        if full_bytes <= _RING_TILE_BYTES:
            expanded = native_ring_velocity_expanded(
                points,
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            return torch.sum(expanded * gamma[None, :, None], dim=1)
        total = torch.zeros_like(points)
        target_cap = max(1, _RING_TILE_BYTES // (rings.shape[0] * 3 * 8))
        for start in range(0, points.shape[0], target_cap):
            stop = min(start + target_cap, points.shape[0])
            expanded = native_ring_velocity_expanded(
                points[start:stop],
                rings,
                core_fraction=core_fraction,
                reference_length=reference_length,
            )
            total[start:stop] = torch.sum(
                expanded * gamma[None, :, None], dim=1
            )
        return total

    def _delta_grid(
        self, gamma: torch.Tensor, mf2_history: torch.Tensor, nc: int, ns: int
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return ``(Gamma_n - Gamma_{n-1} grid, dGamma/dt)``.

        In bound_rate mode the solver's mf2_history input IS
        ``(gamma - gamma_bound)/dt`` with gamma_bound the last committed
        circulation, so the identity below is exact and repeat-safe for
        re-proposals of one committed parent.  This reproduces Pterra's
        zeros-initialised first-step history too (the initial native
        gamma_bound is exactly zero).
        """
        if self.wake_history_mode == "bound_rate":
            gamma_rate = mf2_history
            return (mf2_history * self.aerodynamic_dt).reshape(nc, ns), gamma_rate
        if self._previous_gamma is not None:
            delta = (gamma - self._previous_gamma).reshape(nc, ns)
            return delta, delta.reshape(-1) / self.aerodynamic_dt
        return None, torch.zeros_like(gamma)

    @staticmethod
    def _effective_leg_strengths(
        grid: torch.Tensor, delta_grid: torch.Tensor | None
    ) -> tuple[torch.Tensor, ...]:
        """Pterra effective filament strengths on the native (nc, ns) grid.

        Pterasoftware ``_calculate_loads``: half the circulation difference
        with the neighbour across each edge, full circulation on the
        outermost filaments, and the shed ``Gamma_now - Gamma_last`` on the
        trailing-edge back filaments.
        """
        right = grid.clone()
        right[:, :-1] = 0.5 * (grid[:, :-1] - grid[:, 1:])
        front = grid.clone()
        front[1:, :] = 0.5 * (grid[1:, :] - grid[:-1, :])
        left = grid.clone()
        left[:, 1:] = 0.5 * (grid[:, 1:] - grid[:, :-1])
        back = grid.clone()
        back[:-1, :] = 0.5 * (grid[:-1, :] - grid[1:, :])
        if delta_grid is not None:
            back[-1, :] = delta_grid[-1, :]
        return tuple(part.reshape(-1) for part in (front, right, back, left))

    @staticmethod
    def _leg_layout(
        geometry: NativeV5MGeometry,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Filament midpoints, grid-motion velocities and vectors per leg.

        Filament legs follow the native ring traversal 0->1->2->3->0:
        front 0->1 (leading-edge side), right 1->2 (span +j side),
        back 2->3 (trailing-edge side), left 3->0 (span -j side).  The
        movement velocity is Pterra's ``-(x_n - x_{n-1})/dt`` at the filament
        centre; the analytic ring-corner velocities are its exact
        continuous-time form.
        """
        rings = geometry.rings
        ring_velocity = geometry.ring_velocity
        mids = []
        movements = []
        lengths = []
        for start, end in ((0, 1), (1, 2), (2, 3), (3, 0)):
            a = rings[:, start]
            b = rings[:, end]
            mids.append(0.5 * (a + b))
            movements.append(-0.5 * (ring_velocity[:, start] + ring_velocity[:, end]))
            lengths.append(b - a)
        return mids, movements, lengths

    def _integrate_kj(
        self,
        geometry: NativeV5MGeometry,
        leg_strengths: tuple[torch.Tensor, ...],
        leg_velocities: list[torch.Tensor],
        movements: list[torch.Tensor],
        lengths: list[torch.Tensor],
        mids: list[torch.Tensor],
        gamma_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """F = rho Gamma_eff (v_sol - v_grid) x dl + rho dGamma/dt A n."""
        count = geometry.normals.shape[0]
        panel_forces = torch.zeros(
            (count, 3), device=self.device, dtype=torch.float64
        )
        total_moment = torch.zeros(3, device=self.device, dtype=torch.float64)
        for strength, velocity, movement, length, mid in zip(
            leg_strengths, leg_velocities, movements, lengths, mids, strict=True
        ):
            force = (
                self.density
                * strength.reshape(-1, 1)
                * torch.linalg.cross(velocity + movement, length, dim=1)
            )
            panel_forces += force
            total_moment += torch.sum(torch.linalg.cross(mid, force, dim=1), dim=0)
        # Unsteady Bernoulli term: +rho dGamma/dt A n in the native
        # (clockwise-traversal) ring convention — Pterra's -rho dGamma/dt A n
        # for its counter-clockwise rings, same physical force.
        unsteady = (
            (self.density * gamma_rate)[:, None]
            * geometry.areas[:, None]
            * geometry.normals
        )
        panel_forces = panel_forces + unsteady
        total_moment = total_moment + torch.sum(
            torch.linalg.cross(geometry.collocation, unsteady, dim=1), dim=0
        )
        total_force = torch.sum(panel_forces, dim=0)
        return panel_forces, total_force, total_moment

    def _kj_forces(
        self,
        geometry: NativeV5MGeometry,
        gamma: torch.Tensor,
        external_flow: torch.Tensor,
        mf2_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pterra ``_calculate_loads`` from the assembler-visible inputs.

        The solution velocity at the filament midpoints is reconstructed
        from the collocation ``external_flow`` (exact bound part
        re-evaluated at the midpoints, far field from the shared-edge
        neighbour average).  ``refine_kj_with_solution_velocity`` replaces
        this reconstruction with the solver's exact filament velocities.
        """
        if self.aerodynamic_dt is None:
            raise ValueError(
                "the Pterra force extraction requires aerodynamic_dt "
                "(construct RigidAuthorLoadAssembler with the solver dt)"
            )
        nc, ns = self._panel_topology(geometry)
        count = nc * ns
        delta_grid, gamma_rate = self._delta_grid(gamma, mf2_history, nc, ns)
        leg_strengths = self._effective_leg_strengths(
            gamma.reshape(nc, ns), delta_grid
        )

        # Neighbour panel across each edge (for the far-field average at the
        # shared filament midpoint); None maps to the owning panel itself.
        index = torch.arange(count, device=self.device, dtype=torch.int64)
        ii = index // ns
        jj = index % ns
        leg_neighbours = (
            torch.where(ii > 0, index - ns, index),
            torch.where(jj < ns - 1, index + 1, index),
            torch.where(ii < nc - 1, index + ns, index),
            torch.where(jj > 0, index - 1, index),
        )

        # Split external_flow into its exact bound part at the collocation
        # points and the far field (freestream + wake + particles), then
        # re-evaluate the bound part at the filament midpoints.
        bound_collocation = self._bound_velocity(
            geometry.collocation, geometry.rings, gamma, nc
        )
        far_field = external_flow - bound_collocation
        mids, movements, lengths = self._leg_layout(geometry)
        leg_velocities = []
        for neighbour in leg_neighbours:
            far_value = 0.5 * (far_field + far_field[neighbour])
            leg_velocities.append(far_value)
        bound_mids = self._bound_velocity(
            torch.cat(mids, dim=0), geometry.rings, gamma, nc
        )
        for leg in range(4):
            leg_velocities[leg] = (
                leg_velocities[leg] + bound_mids[leg * count : (leg + 1) * count]
            )
        return self._integrate_kj(
            geometry, leg_strengths, leg_velocities, movements, lengths, mids,
            gamma_rate,
        )

    def leg_midpoints_flat(self, geometry: NativeV5MGeometry) -> torch.Tensor:
        """All four filament midpoints stacked: (4 * panel_count, 3)."""
        mids, _, _ = self._leg_layout(geometry)
        return torch.cat(mids, dim=0)

    def refine_kj_with_solution_velocity(
        self,
        geometry: NativeV5MGeometry,
        gamma: torch.Tensor,
        gamma_rate: torch.Tensor,
        solution_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pterra ``_calculate_loads`` with EXACT filament velocities.

        ``solution_velocity`` is the full solution velocity (freestream +
        wake + particles + bound sheet) evaluated at
        ``leg_midpoints_flat(geometry)``, exactly as the frozen Pterra oracle
        evaluates its four leg centres.  ``gamma_rate`` is dGamma/dt of the
        same state (bound_rate: the solver's mf2_history).
        """
        nc, ns = self._panel_topology(geometry)
        if self.aerodynamic_dt is None:
            raise ValueError(
                "the Pterra force extraction requires aerodynamic_dt "
                "(construct RigidAuthorLoadAssembler with the solver dt)"
            )
        delta_grid = (gamma_rate * self.aerodynamic_dt).reshape(nc, ns)
        leg_strengths = self._effective_leg_strengths(
            gamma.reshape(nc, ns), delta_grid
        )
        mids, movements, lengths = self._leg_layout(geometry)
        count = nc * ns
        leg_velocities = [
            solution_velocity[leg * count : (leg + 1) * count] for leg in range(4)
        ]
        panel_forces, total_force, total_moment = self._integrate_kj(
            geometry, leg_strengths, leg_velocities, movements, lengths, mids,
            gamma_rate,
        )
        if not bool(torch.isfinite(panel_forces).all().item()):
            raise FloatingPointError("rigid native V5M KJ refinement is non-finite")
        return panel_forces, total_force, total_moment

    def assemble(
        self,
        *,
        structural_state: wp.array | None,
        geometry: NativeV5MGeometry,
        aic: torch.Tensor,
        gamma: torch.Tensor,
        external_flow: torch.Tensor,
        gamma_gradient: torch.Tensor,
        mf2_history: torch.Tensor,
    ) -> RigidAuthorLoad:
        if structural_state is not None:
            raise TypeError(
                "the rigid native V5M path has no structural state; pass None"
            )
        # Constant part (author's lift1 + wake-history Mf2).
        lift1 = self.density * torch.sum(external_flow * gamma_gradient, dim=1)
        wake_history = self.density * mf2_history
        # Velocity part (author's lift2 + Mf2_1), evaluated on the prescribed
        # SurfaceFrame velocities instead of a structural substep.  Both
        # pressures carry rho exactly like lift1/Mf2 above and like the frozen
        # reference paths (warp_fsi/coupled.py dp_lift1_flat_kernel,
        # platform/warp_vpm/q16_real_fsi_coupling.py): summing a bare
        # m^2/s^2 term into a Pa total would add incompatible units.
        lift2 = -self.density * torch.sum(
            geometry.collocation_velocity * gamma_gradient, dim=1
        )
        diagonal_31 = geometry.rings[:, 3] - geometry.rings[:, 1]
        diagonal_24 = geometry.rings[:, 2] - geometry.rings[:, 0]
        diagonal_31_rate = geometry.ring_velocity[:, 3] - geometry.ring_velocity[:, 1]
        diagonal_24_rate = geometry.ring_velocity[:, 2] - geometry.ring_velocity[:, 0]
        cross = torch.linalg.cross(diagonal_31, diagonal_24, dim=1)
        cross_rate = torch.linalg.cross(
            diagonal_31_rate, diagonal_24, dim=1
        ) + torch.linalg.cross(diagonal_31, diagonal_24_rate, dim=1)
        cross_norm = torch.linalg.vector_norm(cross, dim=1)
        if bool(torch.any(cross_norm <= 0.0).item()):
            raise FloatingPointError("rigid native V5M surface has a collapsed panel")
        normal_rate_unprojected = cross_rate / cross_norm[:, None]
        normal_rate = normal_rate_unprojected - geometry.normals * torch.sum(
            geometry.normals * normal_rate_unprojected, dim=1, keepdim=True
        )
        bound_influence_rate = material_ring_velocity_derivative_expanded(
            geometry.collocation,
            geometry.collocation_velocity,
            geometry.rings,
            geometry.ring_velocity,
        )
        bound_rate = torch.sum(bound_influence_rate * gamma[None, :, None], dim=1)
        slip = geometry.collocation_velocity - external_flow
        mf2_1_rhs = torch.sum(slip * normal_rate, dim=1) - torch.sum(
            bound_rate * geometry.normals, dim=1
        )
        lu = torch.linalg.lu_factor(aic)
        mf2_1 = self.density * torch.linalg.lu_solve(
            lu[0], lu[1], mf2_1_rhs.unsqueeze(1)
        ).squeeze(1)

        pressure = lift1 + wake_history + lift2 + mf2_1
        # Integrated load: the Pterra-faithful Kutta-Joukowski + unsteady
        # extraction above — NOT the pressure projection, which has no
        # streamwise component and therefore no attached-flow thrust.
        panel_forces, total_force, total_moment = self._kj_forces(
            geometry, gamma, external_flow, mf2_history
        )
        wrench = torch.cat((total_force, total_moment)).unsqueeze(0).contiguous()
        for name, value in (
            ("rigid lift1 pressure", lift1),
            ("rigid wake-history pressure", wake_history),
            ("rigid lift2 pressure", lift2),
            ("rigid Mf2_1 pressure", mf2_1),
            ("rigid total pressure", pressure),
            ("rigid panel forces", panel_forces),
            ("rigid wrench", wrench),
        ):
            _require_cuda64(name, value, self.device)
        if not bool(
            torch.isfinite(panel_forces).all().item()
            and torch.isfinite(wrench).all().item()
        ):
            raise FloatingPointError("rigid native V5M KJ load is non-finite")
        self._previous_gamma = gamma.detach().clone()
        return RigidAuthorLoad(
            lift1_pressure=lift1.detach().clone(),
            wake_history_pressure=wake_history.detach().clone(),
            lift2_pressure=lift2.detach().clone(),
            mf2_1_pressure=mf2_1.detach().clone(),
            constant_pressure=pressure.detach().clone(),
            panel_forces=panel_forces.detach().clone(),
            total_force=total_force.detach().clone(),
            total_moment=total_moment.detach().clone(),
            constant_generalized_force=wp.from_torch(
                wrench, dtype=warp_config.DTYPE, requires_grad=False
            ),
        )


class RigidNativeV5MSolver(Q16NativeV5MSolver):
    """``Q16NativeV5MSolver`` driven by SurfaceFrame geometry, not Q16 state.

    ``propose()``/``initialize()`` are inherited unchanged, so the frozen
    native V5M numerics (bound/LEV/TEV/free-wake transactional state, AIC,
    LESP release, Kelvin and Neumann gates) run on exactly the same code
    path; only construction differs — no Q16 mesh, no panel-load transfer,
    no author-load Q16 quadrature — and the load ownership moves to the
    rigid assembler's Pterra-faithful Kutta-Joukowski wrench.  ``propose``
    itself is wrapped (not copied): the inherited solve produces the frozen
    state proposal, then the wrapper republishes the load from the rigid
    assembler's integrated panel forces — in bound_rate mode re-integrated
    with the exact solution velocities at the filament midpoints — because
    the inherited packaging projects the author audit pressure onto
    ``p A n``, a projection with no streamwise component and therefore no
    attached-flow thrust.
    """

    def __init__(self, surface: RigidV5MSurface, settings: NativeV5MConfig) -> None:
        if not isinstance(surface, RigidV5MSurface):
            raise TypeError("RigidNativeV5MSolver requires a RigidV5MSurface")
        if surface.nc != settings.chordwise_panels or surface.ns != settings.spanwise_panels:
            raise ValueError("surface and aerodynamic topology differ")
        self.surface = surface
        self.settings = settings
        self.device = torch.device(settings.device)
        self.author_load_assembler = RigidAuthorLoadAssembler(
            density=settings.density,
            device=self.device,
            chordwise_panels=settings.chordwise_panels,
            spanwise_panels=settings.spanwise_panels,
            aerodynamic_dt=settings.aerodynamic_dt,
            wake_history_mode=settings.wake_history_mode,
        )
        self.v_inf = freestream_vector(
            settings.freestream, settings.freestream_angle_deg, self.device
        )

    def propose(
        self,
        committed: Any,
        structural_state: wp.array | None,
        structural_velocity: wp.array | None,
    ) -> Any:
        proposal = super().propose(committed, structural_state, structural_velocity)
        author = proposal.author_load
        if not isinstance(author, RigidAuthorLoad):
            raise TypeError(
                "the rigid native V5M solver received a non-rigid author load"
            )
        if self.settings.wake_history_mode == "bound_rate":
            author, load = self._refine_load_with_filament_velocity(
                proposal, author
            )
            return replace(proposal, author_load=author, load=load)
        kj_load = NativeV5MLoad(
            panel_positions=proposal.load.panel_positions,
            panel_forces=author.panel_forces,
            pressure=author.constant_pressure,
            total_force=author.total_force,
            total_moment=author.total_moment,
        )
        return replace(proposal, load=kj_load)

    def _refine_load_with_filament_velocity(
        self, proposal: Any, author: RigidAuthorLoad
    ) -> tuple[RigidAuthorLoad, NativeV5MLoad]:
        """Re-integrate the KJ load at the EXACT filament velocities.

        The assembler sees only collocation-based inputs, so its in-solve
        pass reconstructs the filament solution velocity approximately (far
        field from the neighbour-average collocation values).  This wrapper
        has the post-solve trial state, so it re-evaluates the full solution
        velocity — freestream + bound sheet (Gamma_n) + pre-shed wake + LEV
        particles — at the four filament midpoints of every panel, exactly
        as the frozen Pterra oracle evaluates its leg centres, and
        re-integrates the same effective-strength KJ + unsteady load.  In
        bound_rate mode the author ledger's ``wake_history_pressure``
        recovers the solver's mf2_history (== dGamma/dt) exactly, so the
        refinement changes no state and stays repeat-safe.
        """
        trial = proposal.trial_state
        geometry = self.surface.evaluate(None, None)
        assembler = self.author_load_assembler
        gamma = trial.gamma_bound  # the base propose() stored Gamma_n here
        gamma_rate = author.wake_history_pressure / self.settings.density
        mids = assembler.leg_midpoints_flat(geometry)
        velocity = self.v_inf[None, :].expand(mids.shape[0], 3) + self._ring_velocity(
            mids, geometry.rings, gamma, rough=False
        )
        ns = self.settings.spanwise_panels
        if trial.wake_rings.shape[0] > ns:
            # The load was assembled BEFORE this step's TEV row was shed;
            # skip the newest ns rings so the field matches that state.
            velocity = velocity + self._ring_velocity(
                mids,
                trial.wake_rings[ns:],
                trial.wake_gamma[ns:],
                rough=False,
            )
        if trial.particle_field.n:
            velocity = velocity + trial.particle_field.velocity_at_cuda(mids)
        panel_forces, total_force, total_moment = (
            assembler.refine_kj_with_solution_velocity(
                geometry, gamma, gamma_rate, velocity
            )
        )
        wrench = torch.cat((total_force, total_moment)).unsqueeze(0).contiguous()
        refined = replace(
            author,
            panel_forces=panel_forces.detach().clone(),
            total_force=total_force.detach().clone(),
            total_moment=total_moment.detach().clone(),
            constant_generalized_force=wp.from_torch(
                wrench, dtype=warp_config.DTYPE, requires_grad=False
            ),
        )
        load = NativeV5MLoad(
            panel_positions=proposal.load.panel_positions,
            panel_forces=refined.panel_forces,
            pressure=refined.constant_pressure,
            total_force=refined.total_force,
            total_moment=refined.total_moment,
        )
        return refined, load


__all__ = [
    "RIGID_NATIVE_V5M_CONTRACT",
    "RigidAuthorLoad",
    "RigidAuthorLoadAssembler",
    "RigidNativeV5MSolver",
    "RigidV5MSurface",
]
