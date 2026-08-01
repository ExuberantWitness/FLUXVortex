"""N3-only spatial P2 shadow with a unified panel-pressure difference.

The production V4.1 trajectory is a read-only counterfactual.  This module
owns a private :class:`P2SpatialLEVCandidate` and a private previous-bound
state.  It returns only the panelwise difference

``coupled P2 unified pressure - untouched baseline unified pressure``.

No force is inferred from LESP, and no impulse, fitted coefficient, pressure
cap, decay, self-advection, or target-load input is present.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributed_doublet import DistributedDoubletError
from .p2_spatial_candidate import P2SpatialLEVCandidate, P2SpatialStep
from .unified_panel_pressure import (
    structured_uvlm_surface_gradient,
    unified_panel_pressure,
)


def _finite(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise DistributedDoubletError(
            f"{name} must be finite with shape {shape}, got {array.shape}"
        )
    return array


def _readonly(value) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_bool(value) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True)
    result.setflags(write=False)
    return result


def _readonly_p2_step(step: P2SpatialStep) -> P2SpatialStep:
    return P2SpatialStep(
        bound_gamma=_readonly(step.bound_gamma),
        bound_pre=_readonly(step.bound_pre),
        induced_velocity=_readonly(step.induced_velocity),
        release_previous=_readonly(step.release_previous),
        release_current=_readonly(step.release_current),
        a0_geometry=_readonly(step.a0_geometry),
        a0_pre=_readonly(step.a0_pre),
        a0_post=_readonly(step.a0_post),
        active=_readonly_bool(step.active),
        bound_residual=float(step.bound_residual),
        lesp_residual=float(step.lesp_residual),
        source_condition_number=float(step.source_condition_number),
        bands=int(step.bands),
        outflow_bands=int(step.outflow_bands),
        appended=bool(step.appended),
        self_advection_included=bool(step.self_advection_included),
    )


@dataclass(frozen=True)
class N3OnlyPressureGuards:
    """Numerical and identity guards for one panel-pressure difference."""

    baseline_ledger_passed: bool
    coupled_ledger_passed: bool
    baseline_pressure_ledger_residual: float
    baseline_force_ledger_residual: float
    coupled_pressure_ledger_residual: float
    coupled_force_ledger_residual: float
    pressure_decomposition_residual: float
    force_decomposition_residual: float
    attached_zero_required: bool
    attached_zero_bitwise: bool
    all_finite: bool
    bound_reaction_included: bool = True
    self_advection_included: bool = False

    @property
    def passed(self) -> bool:
        return bool(
            self.baseline_ledger_passed
            and self.coupled_ledger_passed
            and self.all_finite
            and self.pressure_decomposition_residual <= 1.0e-12
            and self.force_decomposition_residual <= 1.0e-12
            and (
                not self.attached_zero_required
                or self.attached_zero_bitwise
            )
            and self.bound_reaction_included
            and not self.self_advection_included
        )


@dataclass(frozen=True)
class N3OnlyPressureIncrement:
    """Per-panel N3 pressure/force increment and its two operands."""

    pressure_increment: np.ndarray
    force_increment: np.ndarray
    baseline_pressure: np.ndarray
    baseline_force: np.ndarray
    coupled_pressure: np.ndarray
    coupled_force: np.ndarray
    bound_reaction: np.ndarray
    guards: N3OnlyPressureGuards


@dataclass(frozen=True)
class N3OnlyShadowStep:
    """Private P2 state advanced from one read-only V4.1 time step."""

    p2: P2SpatialStep
    baseline_bound_gamma: np.ndarray
    coupled_bound_gamma: np.ndarray
    private_bound_previous: np.ndarray
    bound_reaction: np.ndarray
    inventory_absent: bool
    serial: int


def n3_only_unified_pressure_increment(
    *,
    density: float,
    dt: float,
    nc: int,
    ns: int,
    baseline_local_velocity,
    p2_induced_velocity,
    baseline_bound_gamma,
    baseline_bound_previous,
    coupled_bound_gamma,
    coupled_bound_previous,
    release_current,
    release_previous,
    panel_chord_tangent,
    panel_span_tangent,
    chord_step,
    span_step,
    area,
    normal,
) -> N3OnlyPressureIncrement:
    """Evaluate the exact coupled-minus-baseline unified pressure.

    ``coupled_bound_previous`` is the shadow's private history, not the
    production previous-bound array.  Including it and the coupled current
    state makes the induced bound reaction part of N3 rather than silently
    assigning it back to N1.
    """

    if (
        not isinstance(nc, (int, np.integer))
        or not isinstance(ns, (int, np.integer))
        or nc < 1
        or ns < 1
    ):
        raise DistributedDoubletError("nc and ns must be positive integers")
    if not np.isfinite(dt) or dt <= 0.0:
        raise DistributedDoubletError("dt must be positive and finite")
    count = int(nc) * int(ns)
    velocity_baseline = _finite(
        "baseline_local_velocity",
        baseline_local_velocity,
        (count, 3),
    )
    velocity_p2 = _finite(
        "p2_induced_velocity",
        p2_induced_velocity,
        (count, 3),
    )
    gamma_baseline = _finite(
        "baseline_bound_gamma",
        baseline_bound_gamma,
        (count,),
    )
    gamma_baseline_previous = _finite(
        "baseline_bound_previous",
        baseline_bound_previous,
        (count,),
    )
    gamma_coupled = _finite(
        "coupled_bound_gamma",
        coupled_bound_gamma,
        (count,),
    )
    gamma_coupled_previous = _finite(
        "coupled_bound_previous",
        coupled_bound_previous,
        (count,),
    )
    q_current = _finite("release_current", release_current, (int(ns),))
    q_previous = _finite(
        "release_previous",
        release_previous,
        (int(ns),),
    )
    chord_tangent = _finite(
        "panel_chord_tangent",
        panel_chord_tangent,
        (count, 3),
    )
    span_tangent = _finite(
        "panel_span_tangent",
        panel_span_tangent,
        (count, 3),
    )
    chord_delta = _finite("chord_step", chord_step, (count,))
    span_delta = _finite("span_step", span_step, (count,))
    panel_area = _finite("area", area, (count,))
    panel_normal = _finite("normal", normal, (count, 3))

    release_panel_current = np.tile(q_current, int(nc))
    release_panel_previous = np.tile(q_previous, int(nc))
    baseline_gradient = structured_uvlm_surface_gradient(
        gamma_baseline,
        chord_tangent=chord_tangent,
        span_tangent=span_tangent,
        chord_step=chord_delta,
        span_step=span_delta,
        nc=int(nc),
        ns=int(ns),
    )
    coupled_gradient = structured_uvlm_surface_gradient(
        gamma_coupled + release_panel_current,
        chord_tangent=chord_tangent,
        span_tangent=span_tangent,
        chord_step=chord_delta,
        span_step=span_delta,
        nc=int(nc),
        ns=int(ns),
    )
    baseline = unified_panel_pressure(
        density=density,
        local_velocity=velocity_baseline,
        surface_gradient=baseline_gradient,
        potential_rate_channels={
            "bound_unsteady": (
                gamma_baseline - gamma_baseline_previous
            )
            / dt,
        },
        area=panel_area,
        normal=panel_normal,
    )
    coupled = unified_panel_pressure(
        density=density,
        local_velocity=velocity_baseline + velocity_p2,
        surface_gradient=coupled_gradient,
        potential_rate_channels={
            "bound_unsteady": (
                gamma_coupled - gamma_coupled_previous
            )
            / dt,
            "lev_release_unsteady": (
                release_panel_current - release_panel_previous
            )
            / dt,
        },
        area=panel_area,
        normal=panel_normal,
    )

    pressure_increment = coupled.total_pressure - baseline.total_pressure
    force_increment = coupled.total_force - baseline.total_force
    pressure_residual = float(
        np.max(
            np.abs(
                baseline.total_pressure
                + pressure_increment
                - coupled.total_pressure
            ),
            initial=0.0,
        )
    )
    force_residual = float(
        np.max(
            np.abs(
                baseline.total_force
                + force_increment
                - coupled.total_force
            ),
            initial=0.0,
        )
    )
    attached_zero_required = bool(
        not np.any(velocity_p2)
        and not np.any(q_current)
        and not np.any(q_previous)
        and np.array_equal(gamma_coupled, gamma_baseline)
        and np.array_equal(
            gamma_coupled_previous,
            gamma_baseline_previous,
        )
    )
    attached_zero_bitwise = bool(
        np.array_equal(
            pressure_increment,
            np.zeros_like(pressure_increment),
        )
        and np.array_equal(
            force_increment,
            np.zeros_like(force_increment),
        )
    )
    baseline_report = baseline.ledger_report()
    coupled_report = coupled.ledger_report()
    all_finite = bool(
        np.all(np.isfinite(pressure_increment))
        and np.all(np.isfinite(force_increment))
    )
    guards = N3OnlyPressureGuards(
        baseline_ledger_passed=baseline_report.passed,
        coupled_ledger_passed=coupled_report.passed,
        baseline_pressure_ledger_residual=(
            baseline_report.pressure_residual
        ),
        baseline_force_ledger_residual=baseline_report.force_residual,
        coupled_pressure_ledger_residual=(
            coupled_report.pressure_residual
        ),
        coupled_force_ledger_residual=coupled_report.force_residual,
        pressure_decomposition_residual=pressure_residual,
        force_decomposition_residual=force_residual,
        attached_zero_required=attached_zero_required,
        attached_zero_bitwise=attached_zero_bitwise,
        all_finite=all_finite,
    )
    if not guards.passed:
        raise DistributedDoubletError(
            "N3-only unified pressure guards failed: "
            f"pressure={pressure_residual:.3e}, "
            f"force={force_residual:.3e}, "
            f"attached_zero={attached_zero_bitwise}"
        )
    return N3OnlyPressureIncrement(
        pressure_increment=_readonly(pressure_increment),
        force_increment=_readonly(force_increment),
        baseline_pressure=_readonly(baseline.total_pressure),
        baseline_force=_readonly(baseline.total_force),
        coupled_pressure=_readonly(coupled.total_pressure),
        coupled_force=_readonly(coupled.total_force),
        bound_reaction=_readonly(gamma_coupled - gamma_baseline),
        guards=guards,
    )


class P2SpatialN3OnlyShadow:
    """Stateful N3-only wrapper around the frozen spatial P2 candidate."""

    def __init__(
        self,
        *,
        nc: int,
        ns: int,
        span_edges,
        u_infinity: float,
        dt: float,
        lesp_crit: float,
        quadrature_order: int,
        max_bands: int,
        mirror_halfwing: bool,
    ):
        if (
            not isinstance(nc, (int, np.integer))
            or not isinstance(ns, (int, np.integer))
            or nc < 1
            or ns < 1
        ):
            raise DistributedDoubletError(
                "nc and ns must be positive integers"
            )
        self.nc = int(nc)
        self.ns = int(ns)
        self.dt = float(dt)
        self._state_model = P2SpatialLEVCandidate(
            ns=self.ns,
            span_edges=span_edges,
            u_infinity=u_infinity,
            dt=dt,
            lesp_crit=lesp_crit,
            quadrature_order=quadrature_order,
            max_bands=max_bands,
            mirror_halfwing=mirror_halfwing,
        )
        self._private_bound_previous = np.zeros(self.nc * self.ns)
        self._pending: N3OnlyShadowStep | None = None
        self._serial = 0
        self._pressure_steps = 0
        self._convection_steps = 0
        self._guard_failures = 0
        self._maximum_pressure_residual = 0.0
        self._maximum_force_residual = 0.0
        self._maximum_bound_reaction = 0.0

    def advance_state(
        self,
        *,
        time: float,
        aic,
        rhs_without_p2,
        baseline_bound_gamma,
        collocation,
        normals,
        leading_edge,
        strip_chord_tangent,
        suction_normal,
        alpha_rad,
        chord,
        delta_x_front,
    ) -> N3OnlyShadowStep:
        """Advance only the private P2/bound counterfactual state."""

        if self._pending is not None:
            raise DistributedDoubletError(
                "pressure_increment must consume the previous shadow state "
                "before advance_state is called again"
            )
        count = self.nc * self.ns
        baseline_gamma = _finite(
            "baseline_bound_gamma",
            baseline_bound_gamma,
            (count,),
        ).copy()
        matrix = _finite("aic", aic, (count, count)).copy()
        rhs = _finite(
            "rhs_without_p2",
            rhs_without_p2,
            (count,),
        ).copy()
        points = _finite(
            "collocation",
            collocation,
            (count, 3),
        ).copy()
        panel_normals = _finite(
            "normals",
            normals,
            (count, 3),
        ).copy()
        edge = _finite(
            "leading_edge",
            leading_edge,
            (self.ns + 1, 3),
        ).copy()
        strip_tangent = _finite(
            "strip_chord_tangent",
            strip_chord_tangent,
            (self.ns, 3),
        ).copy()
        strip_suction = _finite(
            "suction_normal",
            suction_normal,
            (self.ns, 3),
        ).copy()
        alpha = _finite(
            "alpha_rad",
            alpha_rad,
            (self.ns,),
        ).copy()
        local_chord = _finite(
            "chord",
            chord,
            (self.ns,),
        ).copy()
        front_delta = _finite(
            "delta_x_front",
            delta_x_front,
            (self.ns,),
        ).copy()

        p2_step = _readonly_p2_step(self._state_model.solve_step(
            time=time,
            aic=matrix,
            rhs_without_p2=rhs,
            collocation=points,
            normals=panel_normals,
            leading_edge=edge,
            chord_tangent=strip_tangent,
            suction_normal=strip_suction,
            alpha_rad=alpha,
            chord=local_chord,
            delta_x_front=front_delta,
        ))
        inventory_absent = bool(
            p2_step.bands == 0
            and not p2_step.appended
            and not np.any(p2_step.release_current)
            and not np.any(p2_step.release_previous)
        )
        if inventory_absent:
            # With no P2 source or material history, both pressure operands
            # are physically identical.  Reuse the read-only production
            # state to make the preregistered attached-limit identity exact
            # instead of exposing CPU/GPU linear-solver roundoff.
            coupled_gamma = baseline_gamma.copy()
            induced_velocity = np.zeros((count, 3))
            private_previous = self._private_bound_previous.copy()
        else:
            coupled_gamma = _finite(
                "p2 bound_gamma",
                p2_step.bound_gamma,
                (count,),
            ).copy()
            induced_velocity = _finite(
                "p2 induced_velocity",
                p2_step.induced_velocity,
                (count, 3),
            ).copy()
            private_previous = self._private_bound_previous.copy()

        # ``induced_velocity`` is checked here even though it is read again in
        # pressure_increment; keeping it in the step prevents recomputation
        # after material convection.
        if inventory_absent and np.any(induced_velocity):
            raise DistributedDoubletError(
                "absent P2 inventory produced nonzero induced velocity"
            )
        state = N3OnlyShadowStep(
            p2=p2_step,
            baseline_bound_gamma=_readonly(baseline_gamma),
            coupled_bound_gamma=_readonly(coupled_gamma),
            private_bound_previous=_readonly(private_previous),
            bound_reaction=_readonly(coupled_gamma - baseline_gamma),
            inventory_absent=inventory_absent,
            serial=self._serial,
        )
        self._pending = state
        self._serial += 1
        return state

    def pressure_increment(
        self,
        *,
        density: float,
        baseline_local_velocity,
        baseline_bound_previous,
        panel_chord_tangent,
        panel_span_tangent,
        chord_step,
        span_step,
        area,
        normal,
    ) -> N3OnlyPressureIncrement:
        """Consume the pending state and return the panelwise N3 increment."""

        state = self._pending
        if state is None:
            raise DistributedDoubletError(
                "advance_state must be called before pressure_increment"
            )
        count = self.nc * self.ns
        baseline_previous = _finite(
            "baseline_bound_previous",
            baseline_bound_previous,
            (count,),
        )
        private_previous = np.asarray(state.private_bound_previous)
        if state.inventory_absent or self._pressure_steps == 0:
            # Preserve exact equality with an arbitrary, read-only production
            # initial history.  Once a P2 event exists, later steps use only
            # the shadow's independently saved previous-bound state.
            private_previous = baseline_previous
        try:
            result = n3_only_unified_pressure_increment(
                density=density,
                dt=self.dt,
                nc=self.nc,
                ns=self.ns,
                baseline_local_velocity=baseline_local_velocity,
                p2_induced_velocity=state.p2.induced_velocity
                if not state.inventory_absent
                else np.zeros((count, 3)),
                baseline_bound_gamma=state.baseline_bound_gamma,
                baseline_bound_previous=baseline_previous,
                coupled_bound_gamma=state.coupled_bound_gamma,
                coupled_bound_previous=private_previous,
                release_current=state.p2.release_current,
                release_previous=state.p2.release_previous,
                panel_chord_tangent=panel_chord_tangent,
                panel_span_tangent=panel_span_tangent,
                chord_step=chord_step,
                span_step=span_step,
                area=area,
                normal=normal,
            )
        except Exception:
            self._guard_failures += 1
            raise
        self._private_bound_previous = np.array(
            state.coupled_bound_gamma,
            dtype=float,
            copy=True,
        )
        self._pending = None
        self._pressure_steps += 1
        self._maximum_pressure_residual = max(
            self._maximum_pressure_residual,
            result.guards.pressure_decomposition_residual,
        )
        self._maximum_force_residual = max(
            self._maximum_force_residual,
            result.guards.force_decomposition_residual,
        )
        self._maximum_bound_reaction = max(
            self._maximum_bound_reaction,
            float(
                np.max(np.abs(result.bound_reaction), initial=0.0)
            ),
        )
        return result

    def convect(self, external_velocity) -> None:
        """Convect material P2 geometry with the caller's N1 field only."""

        if self._pending is not None:
            raise DistributedDoubletError(
                "pressure_increment must precede convect"
            )
        if not callable(external_velocity):
            raise DistributedDoubletError(
                "external_velocity must be callable"
            )

        def checked_external_velocity(points):
            point_array = np.asarray(points, dtype=float)
            if point_array.ndim != 2 or point_array.shape[1] != 3:
                raise DistributedDoubletError(
                    "convection points must have shape (n,3)"
                )
            return _finite(
                "external convection velocity",
                external_velocity(point_array.copy()),
                point_array.shape,
            )

        self._state_model.convect_heun(checked_external_velocity)
        self._convection_steps += 1

    def diagnostics(self) -> dict:
        """Return state identity, limitations, and accumulated guards."""

        state_diagnostics = self._state_model.diagnostics()
        return {
            "model": "n3-spatial-edge-pressure-v1-shadow",
            "role": "N3-only-diagnostic-shadow",
            "pressure_operator": "unified_panel_pressure",
            "pressure_definition": "coupled_p2_minus_v41_baseline",
            "bound_reaction_included": True,
            "private_bound_previous": (
                self._private_bound_previous.copy().tolist()
            ),
            "pending_pressure": self._pending is not None,
            "advanced_steps": self._serial,
            "pressure_steps": self._pressure_steps,
            "convection_steps": self._convection_steps,
            "guard_failures": self._guard_failures,
            "maximum_pressure_decomposition_residual": (
                self._maximum_pressure_residual
            ),
            "maximum_force_decomposition_residual": (
                self._maximum_force_residual
            ),
            "maximum_bound_reaction": self._maximum_bound_reaction,
            "self_advection_included": False,
            "pressure_clipping_included": False,
            "force_rescaling_included": False,
            "target_data_access": False,
            "state_model": state_diagnostics,
        }
