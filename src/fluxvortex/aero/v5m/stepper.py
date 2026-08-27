"""V5M3DStepper: wraps the existing Q16NativeV5MSolver into the unified
AerodynamicStepper protocol. For single-surface cases (Roj), this is a thin
adapter. For multi-surface cases (Meng), it concatenates frames and runs
the solver on the global panel set.

The adapter does NOT duplicate any physics — it delegates to the existing
propose()/commit() and only handles the surface-frame conversion.

RigidV5MStepper (HANDOFF_IZRAELEVITZ2017 §8 H3) is the rigid-prescribed
sibling: it drives the same native solver through a RigidV5MSurface that
consumes the SurfaceFrame directly, with no Q16 structural-state bridge.
"""
from __future__ import annotations

from typing import Any

from ..protocol import AerodynamicStepper
from .topology import MultiSurfaceTopology


class V5M3DStepper(AerodynamicStepper):
    """Unified 3D V5M stepper adapter.

    Single-surface mode (Roj): wraps Q16NativeV5MSolver directly. The
    SurfaceFrame drives topology bookkeeping while the raw structural state
    (the transitional ``set_structural_state`` bridge) drives the numerics.

    Multi-surface mode (Meng): concatenates SurfaceFrames, runs the solver
    on the global panel set, splits results back per surface.
    """

    def __init__(self, native_solver, *, device: str = "cuda:0"):
        self._solver = native_solver
        self.device = device
        self._topology: MultiSurfaceTopology | None = None
        self._structural_state = None
        self._structural_velocity = None

    @staticmethod
    def _native_owner(owner: Any) -> Any:
        """Unwrap an FSI owner (``owner.aerodynamic``) to the native V5M owner.

        Accepts either a Q16NativeV5MFSIOwner (whose aerodynamic member is
        the Q16NativeV5MOwner) or the native owner itself.
        """
        native = getattr(owner, "aerodynamic", None)
        return owner if native is None else native

    def initialize(self, surfaces: tuple[Any, ...]) -> Any:
        """Initialize the adapter's topology from the surface frames.

        Returns the wrapped native solver (which owns its NativeV5MState).
        """
        if not surfaces:
            raise ValueError(
                "V5M3DStepper.initialize requires at least one surface frame"
            )
        self._topology = MultiSurfaceTopology.from_surface_frames(surfaces)
        if len(surfaces) == 1:
            # Single-surface: direct delegation. The native solver holds the
            # evolving state; the frame only fixes the panel topology.
            frame = surfaces[0]
            surface = self._solver.surface
            if (frame.chordwise_panels, frame.spanwise_panels) != (
                surface.nc,
                surface.ns,
            ):
                raise ValueError(
                    f"surface frame topology "
                    f"{frame.chordwise_panels}x{frame.spanwise_panels} differs "
                    f"from the native solver {surface.nc}x{surface.ns}"
                )
            return self._solver
        # Multi-surface: the AIC must be assembled over the global panel set
        # with offsets so mutual induction is inside one solve (plan §8.6).
        # TODO U7+: multi-surface AIC assembly. For now, raise a clear error
        # so callers know the limit.
        raise NotImplementedError(
            f"Multi-surface V5M is not yet implemented "
            f"({len(surfaces)} surfaces); single-surface only in U7-1"
        )

    def propose(self, committed: Any, surfaces: tuple[Any, ...], dt: float) -> Any:
        """Propose one aero step.

        For single-surface: delegate to the existing propose().  The committed
        object is a Q16NativeV5MFSIOwner (or Q16NativeV5MOwner) and the raw
        structural state comes from the transitional set_structural_state()
        bridge — the unified framework produces SurfaceFrames, but the native
        solver still consumes raw wp.arrays.  ``dt`` is informational: the
        native solver advances on its frozen ``aerodynamic_dt``.  This is a
        transitional interface that will be cleaned up when the full
        kinematic chain is wired in U7-2.
        """
        if len(surfaces) != 1:
            raise NotImplementedError("multi-surface propose is U7+ work")
        if self._topology is None:
            raise RuntimeError(
                "V5M3DStepper.initialize() must be called before propose()"
            )
        if self._structural_state is None or self._structural_velocity is None:
            raise RuntimeError(
                "V5M3DStepper.set_structural_state() must be called before propose()"
            )
        return self._solver.propose(
            self._native_owner(committed).state,
            self._structural_state,
            self._structural_velocity,
        )

    def commit(self, owner: Any, proposal: Any) -> None:
        """Commit a proposal."""
        self._native_owner(owner).commit(proposal)

    def set_structural_state(self, state, velocity) -> None:
        """Provide the raw Q16 structural state for the native solver.

        This is the transitional bridge: the unified framework produces
        SurfaceFrames, but the native solver still consumes raw wp.arrays.
        """
        self._structural_state = state
        self._structural_velocity = velocity

    @property
    def topology(self) -> MultiSurfaceTopology | None:
        return self._topology


class RigidV5MStepper(AerodynamicStepper):
    """V5M stepper for rigid prescribed-motion cases (HANDOFF H3).

    Feeds the SurfaceFrame geometry DIRECTLY into the native solver's
    propose(): each propose stores the frame on the solver's
    ``RigidV5MSurface`` and advances the frozen native V5M machinery one
    aerodynamic step.  There is no Q16 structural-state bridge — this class
    has no ``set_structural_state`` at all, and the rigid surface/assembler
    reject any non-None structural state, so a raw wp.array cannot leak back
    onto the rigid path.  ``dt`` is informational: the native solver advances
    on its frozen ``aerodynamic_dt`` (same contract as ``V5M3DStepper``).
    """

    def __init__(self, native_solver, *, device: str = "cuda:0"):
        from ...warp_fsi.rigid_flux_v5m_native import (
            RigidNativeV5MSolver,
            RigidV5MSurface,
        )

        if not isinstance(native_solver, RigidNativeV5MSolver):
            raise TypeError("RigidV5MStepper requires a RigidNativeV5MSolver")
        if not isinstance(native_solver.surface, RigidV5MSurface):
            raise TypeError("RigidNativeV5MSolver must carry a RigidV5MSurface")
        self._solver = native_solver
        self.device = device
        self._topology: MultiSurfaceTopology | None = None

    def initialize(self, surfaces: tuple[Any, ...]) -> Any:
        """Initialize from the surface frames; return the state owner.

        The owner wraps the solver's fresh ``NativeV5MState`` (built from the
        frame stored on the rigid surface) and is what ``propose`` expects as
        ``committed`` — either the owner itself or its ``.state``.
        """
        if not surfaces:
            raise ValueError(
                "RigidV5MStepper.initialize requires at least one surface frame"
            )
        self._topology = MultiSurfaceTopology.from_surface_frames(surfaces)
        if len(surfaces) != 1:
            raise NotImplementedError(
                f"Multi-surface rigid V5M is not yet implemented "
                f"({len(surfaces)} surfaces); single-surface only"
            )
        frame = surfaces[0]
        surface = self._solver.surface
        if (frame.chordwise_panels, frame.spanwise_panels) != (surface.nc, surface.ns):
            raise ValueError(
                f"surface frame topology "
                f"{frame.chordwise_panels}x{frame.spanwise_panels} differs "
                f"from the native solver {surface.nc}x{surface.ns}"
            )
        from ...warp_fsi.q16_flux_v5m_native import Q16NativeV5MOwner

        surface.set_frame(frame)
        return Q16NativeV5MOwner(self._solver.initialize(None, None))

    def propose(self, committed: Any, surfaces: tuple[Any, ...], dt: float) -> Any:
        """Propose one aero step driven by the current SurfaceFrame."""
        if len(surfaces) != 1:
            raise NotImplementedError("multi-surface propose is not supported")
        if self._topology is None:
            raise RuntimeError(
                "RigidV5MStepper.initialize() must be called before propose()"
            )
        frame = surfaces[0]
        surface = self._solver.surface
        if (frame.chordwise_panels, frame.spanwise_panels) != (surface.nc, surface.ns):
            raise ValueError(
                f"surface frame topology "
                f"{frame.chordwise_panels}x{frame.spanwise_panels} differs "
                f"from the native solver {surface.nc}x{surface.ns}"
            )
        state = self._committed_state(committed)
        surface.set_frame(frame)
        return self._solver.propose(state, None, None)

    def commit(self, owner: Any, proposal: Any) -> None:
        """Commit a proposal."""
        self._committed_owner(owner).commit(proposal)

    @staticmethod
    def _committed_owner(committed: Any) -> Any:
        """Resolve the native owner from a WorldOwner/FSI wrapper.

        Accepts the ``Q16NativeV5MOwner`` returned by ``initialize`` (directly
        or as ``WorldOwner.aero_state``, the one-way coupling contract) or an
        FSI wrapper exposing ``.aerodynamic``.
        """
        candidate = getattr(committed, "aerodynamic", None)
        if not hasattr(candidate, "commit"):
            candidate = getattr(committed, "aero_state", None)
        if not hasattr(candidate, "commit") and hasattr(committed, "state"):
            candidate = committed
        if not hasattr(candidate, "commit"):
            raise TypeError(
                "RigidV5MStepper expects the native owner returned by "
                "initialize(), a WorldOwner carrying it as aero_state, or an "
                "FSI wrapper around it"
            )
        return candidate

    @classmethod
    def _committed_state(cls, committed: Any) -> Any:
        owner = cls._committed_owner(committed)
        return owner.state

    @property
    def topology(self) -> MultiSurfaceTopology | None:
        return self._topology
