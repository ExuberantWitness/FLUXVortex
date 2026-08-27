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
- ``RigidAuthorLoadAssembler`` is the SINGLE surface-load owner: for a
  prescribed rigid surface the motion is analytic at assembly time, so the
  complete author pressure — the constant part (lift1 + wake-history Mf2)
  plus the velocity part (lift2 + Mf2_1) — is assembled once and integrated
  into panel forces and a 6-component world wrench.

Every evolving numerical array stays CUDA float64; Python owns only topology
and transactions, exactly as on the Q16 path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from fluxvortex.kinematics.frames import SurfaceFrame

from . import config as warp_config
from .q16_flux_v5m_author_loads import material_ring_velocity_derivative_expanded
from .q16_flux_v5m_native import (
    NativeV5MConfig,
    NativeV5MGeometry,
    Q16NativeV5MSolver,
    _require_cuda64,
    freestream_vector,
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
    """Full author pressure decomposition at one rigid native V5M endpoint.

    Unlike the Q16 FSI endpoint — whose velocity load must be re-evaluated
    per structural substep — a prescribed rigid surface knows its motion
    analytically, so the complete pressure is assembled once.
    ``constant_pressure`` (the field the inherited propose() protocol reads)
    therefore carries the TOTAL pressure, and every part stays separately
    auditable for the H4/H5 load-ledger evidence.
    """

    lift1_pressure: torch.Tensor
    wake_history_pressure: torch.Tensor
    lift2_pressure: torch.Tensor
    mf2_1_pressure: torch.Tensor
    constant_pressure: torch.Tensor  # total = the four parts summed
    panel_forces: torch.Tensor
    total_force: torch.Tensor
    total_moment: torch.Tensor
    constant_generalized_force: wp.array  # (1, 6) world wrench [F; M]

    @property
    def total_pressure(self) -> torch.Tensor:
        return self.constant_pressure


class RigidAuthorLoadAssembler:
    """Single surface-load owner for the rigid native V5M path.

    Assembles the complete author pressure from the same formulas as
    ``Q16NativeAuthorLoadAssembler``/``Q16NativeAuthorEndpointLoad`` with the
    "live" velocity evaluation folded in (the SurfaceFrame velocities ARE the
    live structural velocity for a prescribed rigid case).  The generalized
    output is the 6-component world wrench [Fx, Fy, Fz, Mx, My, Mz] — there
    are no Q16 degrees of freedom to map into.
    """

    def __init__(self, *, density: float, device: str | torch.device) -> None:
        self.density = float(density)
        self.device = torch.device(device)

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
        panel_forces = pressure[:, None] * geometry.areas[:, None] * geometry.normals
        total_force = torch.sum(panel_forces, dim=0)
        total_moment = torch.sum(
            torch.linalg.cross(geometry.collocation, panel_forces, dim=1), dim=0
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
    rigid assembler's complete-pressure wrench.
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
            density=settings.density, device=self.device
        )
        self.v_inf = freestream_vector(
            settings.freestream, settings.freestream_angle_deg, self.device
        )


__all__ = [
    "RIGID_NATIVE_V5M_CONTRACT",
    "RigidAuthorLoad",
    "RigidAuthorLoadAssembler",
    "RigidNativeV5MSolver",
    "RigidV5MSurface",
]
