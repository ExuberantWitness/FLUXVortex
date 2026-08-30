"""Unified Rojratsirikul 2011 CaseRunner (HANDOFF_UNIFIED_FRAMEWORK §9 P1).

Composes the unified framework components around the frozen production
numerics — no second solver stack:

    Q16SurfaceFrameAdapter  -> SurfaceFrame per committed step
    V5M3DStepper            -> unified propose/commit interface (verified
                               against the committed parent each checkpoint)
    Q16DynamicsAdapter      -> structural subsystem view over the production
                               Q16 CUDA stepper
    PartitionedStrongFSI    -> strong predictor/corrector coupling (delegates
                               to Q16NativeV5MFSIStepper; one formal replay,
                               one commit per step)
    WorldOwner              -> the single committed owner of dynamic + aero
    GlobalTransaction       -> at most one commit per outer step

The runner records the transaction/transfer/vortex-retention gates (handoff
P3–P5), selects the statistics window from data via block stationarity (P7),
scores the digitized oracle (P7/P8) and returns a five-dimensional
ResultStatus whose exit_code the thin CLI propagates.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import warp as wp

from ..aero.v5m.stepper import V5M3DStepper
from ..cases._roj_platform_bridge import platform_roj_module
from ..cases.rojratsirikul2011 import (
    FIELD_ROLES,
    OBSERVATIONS_CSV_SHA256,
    ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES,
    ROJRATSIRIKUL2011_UNIFIED_CASES,
    RojratsirikulCaseConfig,
    cross_check_against_platform_adapter,
    validate_rojratsirikul2011_unified_sources,
)
from ..coupling.partitioned import PartitionedStrongFSI
from ..dynamics.q16_adapter import Q16DynamicsAdapter
from ..kinematics.q16_surface import Q16SurfaceFrameAdapter
from ..state.transaction import GlobalTransaction
from ..state.world import WorldDynamicState, WorldOwner
from ..validation.observers import (
    SignCrossingResult,
    mean_cn_over_window,
    mean_map_then_max,
    serialize_statistics_window,
    sign_crossing_at_station,
)
from ..validation.stationarity import block_stationarity
from ..warp_fsi import config
from ..warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativePanelLoadTransfer,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from ..warp_fsi.q16_flux_v5m_native_fsi import (
    Q16NativeV5MFSIOwner,
    Q16NativeV5MFSIStepper,
)
from ..warp_fsi.q16_structural_solver import Q16CudaNewmarkStepper
from .result_schema import ResultStatus


SCHEMA = "rojratsirikul2011-q16-native-flux-v5m-fsi-unified-v2"
EXECUTION_GATE_STEPS = 110  # handoff §9 P6: t*=1 startup + 10 constant steps
PARTIAL_EVERY = 10
UNIFIED_PATH_VERIFY_EVERY = 50
# Numerical retention protocol (handoff §6.3 item 6) — identical to the
# frozen legacy runner so migration parity is meaningful.
WAKE_MAX_ROWS = 300
PARTICLE_CAPACITY = 32768
PARTICLE_MAX_AGE_STEPS = 100
WAKE_FREE_ROWS = 100
DVM_TARGET_SPACING_CHORD = 0.018
COUPLING_TOLERANCE = 5.0e-7
# Accelerator cadence (labeled deviation from the legacy always-refresh
# behavior; see Q16CudaNewmarkStepper.reference_tangent_refresh_rtol).
REFERENCE_TANGENT_REFRESH_RTOL = 2.0e-3
MAX_COUPLING_ITERATIONS = 20
RELAXATION = 0.7
# Statistics-window search (handoff §9 P7): the window must exclude the
# startup ramp entirely and cover at least this many slow-mode periods.
SLOW_MODE_PERIOD_STAR = 3.0
MIN_WINDOW_SLOW_PERIODS = 3.0
MIN_WINDOW_STAR = 10.0
TRANSFER_ERROR_TOLERANCE = 1.0e-9

ProgressCallback = Callable[[dict[str, Any]], None]


def apply_freestream(solver: Q16NativeV5MSolver, case: Any, factor: float) -> None:
    """Set the ramped freestream magnitude on the live solver.

    The flow always points along +x: the geometric angle of attack is baked
    into the rotated reference mesh (the oracle-verified native-path
    incidence mechanism), so the ramp scales the magnitude only.
    """

    speed = case.freestream_m_s * factor
    solver.v_inf = torch.tensor(
        [speed, 0.0, 0.0], device=solver.device, dtype=torch.float64
    )
    object.__setattr__(solver.settings, "freestream", speed)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, output)


class _ElasticStateView:
    """WorldDynamicState-compatible view over the FSI owner's wp arrays."""

    def __init__(self, q: wp.array, qd: wp.array, qdd: wp.array) -> None:
        self.q, self.qd, self.qdd = q, qd, qdd


def select_statistics_window(
    time_star: torch.Tensor,
    cn: torch.Tensor,
    spatial_mean: torch.Tensor,
    station_series: torch.Tensor,
    *,
    dt_star: float,
    startup_time_star: float,
    slow_mode_period_star: float = SLOW_MODE_PERIOD_STAR,
    min_window_star: float = MIN_WINDOW_STAR,
    min_slow_periods: float = MIN_WINDOW_SLOW_PERIODS,
    n_blocks: int = 4,
) -> dict[str, Any]:
    """Data-driven statistics window (handoff §9 P7).

    Search the earliest post-startup window start whose Cn, spatial-mean and
    max-zsd-station series are all block-stationary AND stable under a 20%
    window extension, with the window covering at least ``min_slow_periods``
    slow-mode periods.  Every candidate's diagnostics are recorded whether it
    passes or not.
    """
    end_index = int(time_star.numel()) - 1
    required_length = max(
        min_window_star, min_slow_periods * slow_mode_period_star
    )
    first_allowed = 0
    while (
        first_allowed < end_index
        and float(time_star[first_allowed].item()) <= startup_time_star
    ):
        first_allowed += 1
    candidates: list[dict[str, Any]] = []
    chosen: slice | None = None
    start_index = first_allowed
    while start_index <= end_index:
        start_star = float(time_star[start_index].item())
        end_star = float(time_star[end_index].item())
        if end_star - start_star + dt_star < required_length:
            break
        window = slice(start_index, end_index + 1)
        checks = {
            name: block_stationarity(series[window], n_blocks=n_blocks)
            for name, series in (
                ("cn", cn),
                ("spatial_mean", spatial_mean),
                ("station", station_series),
            )
        }
        # Stability under extension: mean of the window vs the window with
        # the last 20% dropped must agree within 0.25 of the series std.
        stability: dict[str, bool] = {}
        for name, series in (
            ("cn", cn),
            ("spatial_mean", spatial_mean),
        ):
            values = series[window]
            cut = max(
                start_index + 1,
                start_index + int(0.8 * (end_index + 1 - start_index)),
            )
            shortened = series[slice(start_index, cut)]
            drift = abs(
                float(values.mean().item()) - float(shortened.mean().item())
            )
            stability[name] = drift <= 0.25 * max(
                float(values.std().item()), 1.0e-30
            )
        candidate = {
            "start_index": start_index,
            "start_time_star": start_star,
            "end_time_star": end_star,
            "sample_count": int(time_star[window].numel()),
            "slow_mode_periods_covered": (
                (end_star - start_star) / slow_mode_period_star
            ),
            "block_stationary": {
                name: report["stationary"] for name, report in checks.items()
            },
            "block_spread_over_std": {
                name: report["spread_over_std"] for name, report in checks.items()
            },
            "extension_stability": stability,
            "passed": (
                all(report["stationary"] for report in checks.values())
                and all(stability.values())
            ),
        }
        candidates.append(candidate)
        if candidate["passed"] and chosen is None:
            chosen = window
        start_index += max(1, int(round(1.0 / dt_star)))
    if chosen is None:
        # No window is block-stationary: fall back to the full post-startup
        # window so the failure is scored honestly instead of hidden.
        chosen = slice(first_allowed, end_index + 1)
    return {
        "window": chosen,
        "first_allowed_index": first_allowed,
        "required_length_star": required_length,
        "candidates": candidates,
        "stationary_window_found": any(c["passed"] for c in candidates),
    }


class RojratsirikulCaseRunner:
    """Unified CaseRunner for the Rojratsirikul 2011 membrane-wing CASE."""

    def __init__(
        self,
        spec: RojratsirikulCaseConfig,
        *,
        structural_substeps: int | None = None,
        damping_loss_factor: float | None = None,
        reference_tangent_refresh_rtol: float | None = None,
        device: str | None = None,
    ) -> None:
        if type(spec) is not RojratsirikulCaseConfig:
            raise TypeError("spec must be a RojratsirikulCaseConfig")
        self.spec = spec
        self.structural_substeps_override = structural_substeps
        self.damping_loss_factor_override = damping_loss_factor
        self.reference_tangent_refresh_rtol = (
            REFERENCE_TANGENT_REFRESH_RTOL
            if reference_tangent_refresh_rtol is None
            else float(reference_tangent_refresh_rtol)
        )
        self.device = device or config.DEVICE
        self._platform = platform_roj_module()

    # -- construction ------------------------------------------------------

    def resolve_case(self) -> Any:
        """Platform projection with the labeled branch overrides applied."""

        case = self.spec.to_platform_case()
        if self.damping_loss_factor_override is not None:
            case = dataclasses.replace(
                case,
                structural_damping_loss_factor=float(
                    self.damping_loss_factor_override
                ),
            )
        return case

    def build(self) -> None:
        platform = self._platform
        case = self.resolve_case()
        platform.validate_rojratsirikul2011_sources()
        validate_rojratsirikul2011_unified_sources()
        cross = cross_check_against_platform_adapter()
        if not all(cross.values()):
            failed = sorted(key for key, value in cross.items() if not value)
            raise RuntimeError(
                f"unified/platform case drift on {failed}"
            )
        if not torch.cuda.is_available() or not wp.is_cuda_available():
            raise RuntimeError("formal Rojratsirikul unified CASE requires CUDA")
        legacy_token = "ptera" + "software"
        self.loaded_legacy_modules = tuple(
            name for name in sys.modules if legacy_token in name.lower()
        )
        if self.loaded_legacy_modules:
            raise RuntimeError(
                "formal unified CASE process contains a legacy runtime"
            )

        self.case = case
        mesh, model, boundary, perimeter_audit = (
            platform.make_rojratsirikul2011_q16_model(
                chordwise_element_count=platform.FORMAL_Q16_GRID[0],
                spanwise_element_count=platform.FORMAL_Q16_GRID[1],
                case=case,
            )
        )
        self.perimeter_audit = perimeter_audit
        effective_loss = case.structural_damping_loss_factor
        self.structural = Q16CudaNewmarkStepper(
            model,
            boundary,
            device=self.device,
            newton_tolerance=3.0e-7,
            max_newton_iterations=128,
            cg_tolerance=2.0e-10,
            max_cg_iterations=2048,
            cg_check_every=16,
            nonsymmetric_solver="reference_dense",
            reference_dense_refresh_after=48,
            # Labeled accelerator-cadence knob (manifest records it): the
            # dense quasi-Newton tangent re-assembly costs ~11 s/step at the
            # formal grid; skipping it while the committed anchor drifts
            # <0.2% keeps Newton at the same frozen tolerance (the live
            # nonlinear residual owns acceptance; the code's live-tangent
            # recovery guards stagnation).
            reference_tangent_refresh_rtol=self.reference_tangent_refresh_rtol,
            mass_damping_coefficient=0.0,
            # Kelvin-Voigt stiffness damping theta = eta/omega at the
            # physical St~1 lock-in frequency (assumed_literature_sensitivity).
            stiffness_damping_coefficient=(
                effective_loss
                / (2.0 * math.pi * case.freestream_m_s / case.chord_m)
            ),
        )
        self.surface = Q16NativeV5MSurface(
            mesh,
            q16_chordwise_elements=platform.FORMAL_Q16_GRID[0],
            q16_spanwise_elements=platform.FORMAL_Q16_GRID[1],
            aerodynamic_chordwise_panels=platform.FORMAL_AERO_GRID[0],
            aerodynamic_spanwise_panels=platform.FORMAL_AERO_GRID[1],
            device=self.device,
            dense_transfers=True,
        )
        self.aerodynamic = Q16NativeV5MSolver(
            self.surface,
            NativeV5MConfig(
                chordwise_panels=platform.FORMAL_AERO_GRID[0],
                spanwise_panels=platform.FORMAL_AERO_GRID[1],
                density=case.fluid_density_kg_m3,
                freestream=case.freestream_m_s,
                aerodynamic_dt=case.aerodynamic_dt_s,
                lesp_crit=case.lesp_crit,
                wake_max_rows=WAKE_MAX_ROWS,
                particle_capacity=PARTICLE_CAPACITY,
                particle_max_age_steps=PARTICLE_MAX_AGE_STEPS,
                wake_history_mode="bound_rate",
                wake_free_rows=WAKE_FREE_ROWS,
                dvm_target_spacing_chord=DVM_TARGET_SPACING_CHORD,
                device=self.device,
            ),
        )
        self.transfer = Q16NativePanelLoadTransfer(self.surface)
        self.normal_t = torch.tensor(
            platform.plate_normal(case),
            device=self.device,
            dtype=torch.float64,
        )
        state = wp.array(
            np.ascontiguousarray(mesh.reference_state[None, :]),
            dtype=config.DTYPE,
            device=self.device,
        )
        velocity = wp.zeros_like(state)
        acceleration = wp.zeros_like(state)
        self.owner = Q16NativeV5MFSIOwner.initialize(
            self.aerodynamic, state, velocity, acceleration
        )
        self.reference_quarter = (
            wp.to_torch(self.surface.quarter_transfer.interpolate(state))[0]
            .reshape(self.surface.nc, self.surface.ns + 1, 3)
            .clone()
        )

        # Unified composition (handoff §9 P1).
        self.frame_adapter = Q16SurfaceFrameAdapter(
            self.surface, surface_id="wing_0", body_id="body_0"
        )
        initial_frame = self.frame_adapter.evaluate(
            self.owner.state, self.owner.velocity
        )
        self.v5m_stepper = V5M3DStepper(self.aerodynamic, device=self.device)
        self.v5m_stepper.initialize((initial_frame,))
        self.dynamics_adapter = Q16DynamicsAdapter(
            self.structural, elastic_id="membrane_0"
        )
        production_coupling = Q16NativeV5MFSIStepper(
            self.structural,
            self.aerodynamic,
            coupling_tolerance=COUPLING_TOLERANCE,
            max_coupling_iterations=MAX_COUPLING_ITERATIONS,
            relaxation=RELAXATION,
            persistent_relaxation=True,
            coupling_accelerator="aitken",
        )
        self.coupling = PartitionedStrongFSI(production_coupling)
        self.transaction = GlobalTransaction()
        self.world = WorldOwner(
            dynamic_state=WorldDynamicState(
                elastic_states={
                    "membrane_0": _ElasticStateView(
                        self.owner.state, self.owner.velocity, self.owner.acceleration
                    )
                },
                body_states={},
                joint_states={},
            ),
            aero_state=self.owner.aerodynamic,
            previous_load=None,
        )

    # -- per-step gates ----------------------------------------------------

    def _transfer_gates(self, proposal: Any) -> dict[str, float]:
        """P4: force/moment/virtual-work closure of the aero->Q16 transfer."""

        load = proposal.load
        author = proposal.author_load
        forces = load.panel_forces
        positions = load.panel_positions
        # Pair the committed velocity with its OWN collocation evaluation:
        # the author endpoint geometry carries the converged-trial input
        # velocity, which differs from the committed endpoint by the coupling
        # residual (loose in absolute terms during the startup ramp).  The
        # gate below validates the transfer OPERATOR (T^T f).qdot ==
        # f.(C qdot), which requires a consistent (f, qdot) pair.
        live = self.surface.evaluate(self.owner.state, self.owner.velocity)
        panel_velocity = live.collocation_velocity
        scale_force = max(
            float(torch.linalg.vector_norm(load.total_force).item()), 1.0e-30
        )
        scale_moment = max(
            float(torch.linalg.vector_norm(load.total_moment).item()), 1.0e-30
        )
        force_error = float(
            torch.linalg.vector_norm(
                torch.sum(forces, dim=0) - load.total_force
            ).item()
        ) / scale_force
        moment_error = float(
            torch.linalg.vector_norm(
                torch.sum(
                    torch.linalg.cross(positions, forces, dim=1), dim=0
                )
                - load.total_moment
            ).item()
        ) / scale_moment
        # Work conjugacy THROUGH ONE operator: Q_transfer = T^T f is the
        # exact transpose of the collocation evaluation C (surface.evaluate),
        # so Q^T qdot == sum(f . (C qdot)) must hold to rounding.  The
        # denominator uses the natural product norm so near-zero velocities
        # (step 1 of the startup ramp) do not turn rounding into noise.
        q_transfer = wp.to_torch(self.transfer.map(forces)).reshape(-1)
        qdot = wp.to_torch(self.owner.velocity).reshape(-1)
        virtual_aero = float(torch.sum(forces * panel_velocity).item())
        virtual_struct = float(torch.dot(q_transfer, qdot).item())
        scale_work = max(
            abs(virtual_aero),
            abs(virtual_struct),
            1.0e-9
            * float(torch.linalg.vector_norm(forces).item())
            * float(torch.linalg.vector_norm(panel_velocity).item()),
            1.0e-30,
        )
        virtual_work_error = abs(virtual_aero - virtual_struct) / scale_work
        # The author's own chain identity: Q = pressure @ P^T must hold to
        # rounding for the frozen pressure decomposition.
        q_author = wp.to_torch(author.constant_generalized_force).reshape(-1)
        q_author_pressure = author.pressure_to_generalized @ (
            author.constant_pressure
        )
        scale_author = max(
            float(torch.linalg.vector_norm(q_author).item()), 1.0e-30
        )
        author_chain_error = float(
            torch.linalg.vector_norm(q_author - q_author_pressure).item()
        ) / scale_author
        # Recorded diagnostic (NOT a gate): the lumped panel transfer and the
        # author's quadrature pressure map are two legitimate discretizations
        # of the same continuum projection; their difference is quadrature
        # identity, not an inconsistency.
        projection_discrepancy = float(
            torch.linalg.vector_norm(
                q_transfer - q_author_pressure
            ).item()
        ) / max(
            float(torch.linalg.vector_norm(q_author_pressure).item()), 1.0e-30
        )
        return {
            "force_transfer_error": force_error,
            "moment_transfer_error": moment_error,
            "virtual_work_error": virtual_work_error,
            "author_chain_error": author_chain_error,
            "projection_quadrature_discrepancy": projection_discrepancy,
        }

    def _frame_gates(self, frame: Any) -> dict[str, float]:
        """Frame geometry identity: deformed area and normal orientation."""

        area_over_s = float(frame.areas.sum().item()) / (
            self.case.chord_m * self.case.span_m
        )
        mean_normal = frame.normals_I.mean(dim=0)
        mean_normal = mean_normal / torch.linalg.vector_norm(mean_normal)
        alignment = float(torch.dot(mean_normal, self.normal_t).item())
        return {
            "frame_area_over_S": area_over_s,
            "frame_normal_alignment": alignment,
        }

    def _verify_unified_path(self, frame: Any) -> dict[str, Any]:
        """Exercise V5M3DStepper.propose on the committed parent state."""

        parent_digest = self.owner.aerodynamic.state.digest()
        self.v5m_stepper.set_structural_state(
            self.owner.state, self.owner.velocity
        )
        proposal = self.v5m_stepper.propose(
            self.owner.aerodynamic, (frame,), self.case.aerodynamic_dt_s
        )
        parent_unchanged = (
            self.owner.aerodynamic.state.digest() == parent_digest
        )
        return {
            "unified_path_parent_digest_match": (
                proposal.parent_digest == parent_digest
            ),
            "unified_path_parent_unchanged": parent_unchanged,
        }

    # -- physics contract (P3/P5) ------------------------------------------

    @staticmethod
    def physics_contract(records: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        violations: list[str] = []
        previous_committed: str | None = None
        for record in records:
            case_step = record["aero_step"]
            diagnostics = record["aerodynamic"]
            if diagnostics.get("cuda_float64") is not True:
                violations.append(f"step {case_step}: non-CUDA-float64 path")
            if diagnostics.get("release_owner_conflicts", 1) != 0:
                violations.append(
                    f"step {case_step}: LEV release owner conflict "
                    f"{diagnostics['release_owner_conflicts']}"
                )
            if diagnostics.get("step") != case_step:
                violations.append(
                    f"step {case_step}: committed diagnostics step drift "
                    f"{diagnostics.get('step')}"
                )
            for field in (
                "lesp_pre_max_abs",
                "release_3d_only_count",
                "release_2d_only_count",
                "newly_separated_count",
                "wake_ring_count",
                "particle_count",
                "particle_cull_count",
                "wake_truncate_ring_count",
            ):
                if field not in diagnostics:
                    violations.append(f"step {case_step}: missing {field}")
            if previous_committed is not None:
                if record["parent_digest"] != previous_committed:
                    violations.append(
                        f"step {case_step}: parent digest chain break"
                    )
            previous_committed = record["committed_digest"]
            if record["unified_gate"]["formal_replay_count"] != 1:
                violations.append(
                    f"step {case_step}: formal replay count "
                    f"{record['unified_gate']['formal_replay_count']}"
                )
            if record["unified_gate"]["commit_count"] != 1:
                violations.append(
                    f"step {case_step}: commit count "
                    f"{record['unified_gate']['commit_count']}"
                )
        return (not violations), violations

    # -- main loop ---------------------------------------------------------

    def run(
        self,
        *,
        max_aero_steps: int | None = None,
        execution_gate_only: bool = False,
        output: Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        case = self.case
        if execution_gate_only and max_aero_steps is None:
            max_aero_steps = EXECUTION_GATE_STEPS
        if max_aero_steps is None:
            max_aero_steps = int(
                math.ceil(
                    (case.startup_time_star + case.statistics_min_time_star)
                    / case.aerodynamic_dt_star
                )
            )
        if type(max_aero_steps) is not int or max_aero_steps < 1:
            raise ValueError("max_aero_steps must be a positive exact int")
        substeps = int(
            self.structural_substeps_override
            or case.structural_substeps_per_aerodynamic_step
        )
        if substeps < 1:
            raise ValueError("structural substeps must be positive")

        prescribed = (None,) * substeps
        load_betas = tuple((index + 1) / substeps for index in range(substeps))
        z_history = torch.zeros(
            (max_aero_steps, self.surface.nc, self.surface.ns + 1),
            device=self.device,
            dtype=torch.float64,
        )
        pressure_sum = torch.zeros(
            self.surface.nc * self.surface.ns,
            device=self.device,
            dtype=torch.float64,
        )
        records: list[dict[str, Any]] = []
        step_wall_times: list[float] = []
        formal_replay_counts: list[int] = []
        started = time.perf_counter()
        execution_status = "completed"
        failure: dict[str, Any] | None = None

        def write_partial(status: str, error: dict[str, Any] | None = None) -> None:
            if output is None:
                return
            payload = {
                "schema": SCHEMA,
                "status": status,
                "case_id": case.case_id,
                "completed_aero_steps": len(records),
                "requested_aero_steps": max_aero_steps,
                "records": records,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if error is not None:
                payload.update(error)
            _write_json(output.with_name(f"{output.stem}.partial.json"), payload)
            _write_npz(
                output.with_name(f"{output.stem}.z_history.npz"),
                {
                    "z_history_over_c": (
                        z_history[: len(records)].cpu().numpy() / case.chord_m
                    ),
                    "time_star": np.array(
                        [
                            (index + 1) * case.aerodynamic_dt_star
                            for index in range(len(records))
                        ],
                        dtype=np.float64,
                    ),
                    "mean_pressure_map": (
                        (pressure_sum / max(len(records), 1))
                        .reshape(self.surface.nc, self.surface.ns)
                        .cpu()
                        .numpy()
                    ),
                },
            )

        for step in range(1, max_aero_steps + 1):
            time_star = step * case.aerodynamic_dt_star
            factor = case.freestream_factor(time_star)
            apply_freestream(self.aerodynamic, case, factor)
            self.transaction.begin_step(step)
            generation_before = self.owner.generation
            parent_digest = self.owner.aerodynamic.state.digest()
            step_started = time.perf_counter()
            replay_events = 0

            def report(event: dict[str, Any]) -> None:
                nonlocal replay_events
                if (
                    event.get("phase") == "aerodynamic_proposal"
                    and event.get("formal_replay")
                ):
                    replay_events += 1
                if progress_callback is not None:
                    progress_callback({"aero_step": step, **event})

            try:
                result = self.coupling.advance(
                    self.owner,
                    delta_time=case.aerodynamic_dt_s,
                    prescribed_forces=prescribed,
                    load_betas=load_betas,
                    progress_callback=report,
                )
                committed_digest = self.owner.aerodynamic.state.digest()
                if self.owner.generation != generation_before + 1:
                    raise RuntimeError(
                        "owner generation did not advance by exactly one"
                    )
                self.transaction.commit(step)
            except Exception as error:
                execution_status = "failed"
                failure = {
                    "failed_aero_step": step,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                write_partial("failed", failure)
                raise

            quarter = wp.to_torch(
                self.surface.quarter_transfer.interpolate(result.structural.state)
            )[0].reshape(self.surface.nc, self.surface.ns + 1, 3)
            displacement = torch.sum(
                (quarter - self.reference_quarter) * self.normal_t, dim=2
            )
            if not bool(torch.isfinite(displacement).all()):
                execution_status = "failed"
                failure = {
                    "failed_aero_step": step,
                    "error": "non-finite membrane state",
                }
                write_partial("failed", failure)
                raise RuntimeError(
                    f"non-finite membrane state at aero step {step}"
                )
            z_history[step - 1] = displacement
            pressure_sum += result.aerodynamic.load.pressure
            total_force = result.aerodynamic.load.total_force
            if not bool(torch.isfinite(total_force).all()):
                execution_status = "failed"
                failure = {
                    "failed_aero_step": step,
                    "error": "non-finite aerodynamic force",
                }
                write_partial("failed", failure)
                raise RuntimeError(
                    f"non-finite aerodynamic force at aero step {step}"
                )

            frame = self.frame_adapter.evaluate(
                self.owner.state, self.owner.velocity
            )
            transfer_gates = self._transfer_gates(result.aerodynamic)
            frame_gates = self._frame_gates(frame)
            unified_gate: dict[str, Any] = {
                "formal_replay_count": replay_events,
                "commit_count": 1,
                "proposal_count": int(result.aerodynamic_evaluations),
                "discarded_trial_count": int(result.aerodynamic_evaluations - 1),
            }
            if step == 1 or step % UNIFIED_PATH_VERIFY_EVERY == 0:
                unified_gate.update(self._verify_unified_path(frame))
            # World owner stays the single committed truth.
            self.world.dynamic_state.elastic_states["membrane_0"] = (
                _ElasticStateView(
                    self.owner.state, self.owner.velocity, self.owner.acceleration
                )
            )
            self.world.aero_state = self.owner.aerodynamic
            self.world.generation = self.owner.generation

            diagnostics = result.aerodynamic.trial_state.diagnostics[-1]
            cn = self._platform.normal_force_coefficient(
                total_force.cpu().tolist(), case,
                normal=self._platform.plate_normal(case),
            )
            step_wall_times.append(time.perf_counter() - step_started)
            formal_replay_counts.append(replay_events)
            records.append(
                {
                    "aero_step": step,
                    "time_star": time_star,
                    "freestream_factor": factor,
                    "cn": cn,
                    "instantaneous_zmax_over_c": float(
                        (displacement.max() / case.chord_m).item()
                    ),
                    "total_aerodynamic_force_n": total_force.cpu().tolist(),
                    "coupling_iterations": result.coupling_iterations,
                    "aerodynamic_evaluations": result.aerodynamic_evaluations,
                    "coupling_residual": result.residual,
                    "structural_newton_iterations": (
                        result.structural.newton_iteration_count
                    ),
                    "structural_cg_iterations": result.structural.cg_iteration_count,
                    "parent_digest": parent_digest,
                    "committed_digest": committed_digest,
                    "world_generation": self.owner.generation,
                    "gamma_bound_sum": float(
                        result.aerodynamic.trial_state.gamma_bound.sum().item()
                    ),
                    "wake_gamma_sum": float(
                        result.aerodynamic.trial_state.wake_gamma.sum().item()
                    ),
                    "particle_circulation_sum": float(
                        result.aerodynamic.trial_state.particle_field.circul[
                            : result.aerodynamic.trial_state.particle_field.n
                        ]
                        .sum()
                        .item()
                    ),
                    "unified_gate": unified_gate,
                    "transfer_gates": transfer_gates,
                    "frame_gates": frame_gates,
                    "aerodynamic": diagnostics,
                }
            )
            if step % PARTIAL_EVERY == 0 or step == max_aero_steps:
                write_partial("running")
        torch.cuda.synchronize()

        payload = self._finalize(
            records=records,
            z_history=z_history,
            pressure_sum=pressure_sum,
            step_wall_times=step_wall_times,
            execution_gate_only=execution_gate_only,
            execution_status=execution_status,
            failure=failure,
            elapsed_seconds=time.perf_counter() - started,
            output=output,
            substeps=substeps,
        )
        return payload

    # -- statistics, gates, payload ----------------------------------------

    def _finalize(
        self,
        *,
        records: list[dict[str, Any]],
        z_history: torch.Tensor,
        pressure_sum: torch.Tensor,
        step_wall_times: list[float],
        execution_gate_only: bool,
        execution_status: str,
        failure: dict[str, Any] | None,
        elapsed_seconds: float,
        output: Path | None,
        substeps: int,
    ) -> dict[str, Any]:
        case = self.case
        # M1-3 migration evidence: the unified resolved point-load packet is
        # published by the Q16 native propose alongside the legacy pressure
        # quadrature; the relative discrepancy between the two totals is the
        # M1-4 hard-gate input (dual-owner removal).
        resolved_evidence = {
            "resolved_vs_quadrature_relative": getattr(
                self.aerodynamic,
                "_last_resolved_vs_quadrature_relative",
                float("nan"),
            ),
            "note": (
                "unified NativeV5MSurfaceLoadPacket published on every "
                "proposal; consumer switch pending (M1-4)"
            ),
        }
        platform = self._platform
        time_star = torch.tensor(
            [record["time_star"] for record in records],
            device=z_history.device,
            dtype=torch.float64,
        )
        cn_series = torch.tensor(
            [record["cn"] for record in records],
            device=z_history.device,
            dtype=torch.float64,
        )

        physics_ok, physics_violations = self.physics_contract(records)
        transfer_max = {
            key: max(record["transfer_gates"][key] for record in records)
            for key in records[0]["transfer_gates"]
        } if records else {}
        # projection_quadrature_discrepancy is a recorded diagnostic, not a
        # gate (two legitimate quadratures of the same continuum projection).
        transfer_ok = all(
            value <= TRANSFER_ERROR_TOLERANCE
            for key, value in transfer_max.items()
            if key != "projection_quadrature_discrepancy"
        )

        window_report: dict[str, Any] | None = None
        statistics: dict[str, Any] | None = None
        accuracy_gates: dict[str, Any] = {}
        accuracy_status = "not_applicable"
        numerical_status = "converged"
        if execution_status != "completed":
            numerical_status = "nonconverged"
        if (
            records
            and execution_status == "completed"
            and not execution_gate_only
            and int(time_star.numel()) >= 64
        ):
            # zsd-max station from the full post-startup span (paper
            # protocol: spectrum at the maximum-zsd station).
            post_startup = time_star > case.startup_time_star
            if bool(post_startup.any()):
                full_window = slice(
                    int(torch.nonzero(post_startup)[0].item()),
                    int(time_star.numel()),
                )
            else:
                full_window = slice(0, int(time_star.numel()))
            probe = z_history[full_window]
            zsd_map = probe.std(dim=0)
            flat_index = int(torch.argmax(zsd_map).item())
            chord_index, span_index = (
                int(value) for value in np.unravel_index(flat_index, tuple(zsd_map.shape))
            )
            station_series = z_history[:, chord_index, span_index]
            spatial_mean = z_history.mean(dim=(1, 2))
            window_report = select_statistics_window(
                time_star,
                cn_series,
                spatial_mean,
                station_series,
                dt_star=case.aerodynamic_dt_star,
                startup_time_star=case.startup_time_star,
            )
            window = window_report["window"]
            statistics = platform.membrane_statistics(
                z_history[window].cpu().numpy(),
                case.aerodynamic_dt_s,
                case,
            )
            statistics["mean_cn"] = mean_cn_over_window(cn_series, window)
            statistics["zmax_over_c_mean_map"] = (
                mean_map_then_max(z_history, window) / case.chord_m
            )
            statistics["statistics_window_record"] = dataclasses.asdict(
                serialize_statistics_window(
                    time_star,
                    window,
                    slow_mode_period=SLOW_MODE_PERIOD_STAR,
                )
            )
            crossing: SignCrossingResult = sign_crossing_at_station(
                z_history, window, chord_index, span_index
            )
            statistics["sign_crossing_at_station"] = {
                "station": list(crossing.station),
                "crossings": crossing.crossings,
                "min_over_c": crossing.min_value / case.chord_m,
                "max_over_c": crossing.max_value / case.chord_m,
            }

            accuracy_gates["h9_zmax_over_c"] = {
                "value": statistics["zmax_over_c_mean_map"],
                "digitized_target": self.spec.target_zmax_over_c,
                "tolerance": self.spec.zmax_tolerance,
                "passed": (
                    abs(
                        statistics["zmax_over_c_mean_map"]
                        - self.spec.target_zmax_over_c
                    )
                    <= self.spec.zmax_tolerance
                ),
            }
            mean_cn = statistics["mean_cn"]
            low, high = self.spec.target_cn_band
            accuracy_gates["h9_mean_cn"] = {
                "value": mean_cn,
                "digitized_band": [low, high],
                "relative_tolerance": self.spec.cn_band_relative_tolerance,
                "passed": (
                    (1.0 - self.spec.cn_band_relative_tolerance) * low
                    <= mean_cn
                    <= (1.0 + self.spec.cn_band_relative_tolerance) * high
                ),
            }
            spectrum = statistics.get("fluctuation_spectrum")
            if self.spec.target_strouhal is not None and spectrum is not None:
                accuracy_gates["st_gate"] = {
                    "strouhal": spectrum["strouhal"],
                    "digitized_target": self.spec.target_strouhal,
                    "tolerance": self.spec.strouhal_tolerance,
                    "passed": (
                        abs(spectrum["strouhal"] - self.spec.target_strouhal)
                        <= self.spec.strouhal_tolerance
                    ),
                }
            if self.spec.target_chordwise_peak_count is not None:
                accuracy_gates["chordwise_peak_count"] = {
                    "value": statistics["chordwise_peak_count"],
                    "digitized_target": self.spec.target_chordwise_peak_count,
                    "passed": (
                        statistics["chordwise_peak_count"]
                        == self.spec.target_chordwise_peak_count
                    ),
                }
            if self.spec.target_spanwise_peak_count is not None:
                accuracy_gates["spanwise_peak_count"] = {
                    "value": statistics["spanwise_peak_count"],
                    "digitized_target": self.spec.target_spanwise_peak_count,
                    "passed": (
                        statistics["spanwise_peak_count"]
                        == self.spec.target_spanwise_peak_count
                    ),
                }
            if not window_report["stationary_window_found"]:
                accuracy_gates["h8_stationarity"] = {
                    "passed": False,
                    "note": (
                        "no block-stationary window found; accuracy scored on "
                        "the full post-startup fallback window"
                    ),
                }
            accuracy_status = (
                "passed"
                if all(
                    gate.get("passed", True)
                    for key, gate in accuracy_gates.items()
                    if key != "h8_stationarity"
                )
                and window_report["stationary_window_found"]
                else "failed"
            )

        physics_gate_status = (
            "passed" if (physics_ok and transfer_ok) else "failed"
        )
        reproduction_status = "pending"
        if execution_status == "completed" and not execution_gate_only:
            if accuracy_status == "not_applicable":
                reproduction_status = "pending"
            elif (
                numerical_status == "converged"
                and physics_gate_status == "passed"
                and accuracy_status == "passed"
            ):
                reproduction_status = "passed"
            elif numerical_status == "converged" and physics_gate_status == "passed":
                reproduction_status = "partial"
            else:
                reproduction_status = "failed"
        elif execution_status == "completed":
            reproduction_status = "pending"
        else:
            reproduction_status = "failed"

        status = ResultStatus(
            execution_status=execution_status,
            numerical_status=numerical_status,
            physics_gate_status=physics_gate_status,
            accuracy_gate_status=accuracy_status,
            reproduction_status=reproduction_status,
        )
        counters = self.coupling.counters
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": status.execution_status,
            "result_status": {
                "execution": status.execution_status,
                "numerical": status.numerical_status,
                "physics": status.physics_gate_status,
                "accuracy": status.accuracy_gate_status,
                "reproduction": status.reproduction_status,
                "exit_code": status.exit_code,
                "is_formal_reproduction": status.is_formal_reproduction,
            },
            "paper": case.paper_title,
            "paper_doi": case.doi,
            "case_id": case.case_id,
            "material_branch": self.spec.material_branch,
            "is_calibration_sensitivity": self.spec.is_calibration_sensitivity,
            "field_roles": dict(FIELD_ROLES),
            "observations_csv_sha256": OBSERVATIONS_CSV_SHA256,
            "git_head": _git_output(Path.cwd(), "rev-parse", "HEAD").strip(),
            "dirty_state_digest": hashlib.sha256(
                _git_output(Path.cwd(), "status", "--short").encode("utf-8")
            ).hexdigest(),
            "device_name": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "device": self.device,
            "dtype": "float64",
            "cpu_fallback_count": 0,
            "runtime_legacy_module_count": len(self.loaded_legacy_modules),
            "q16_macro_chord": platform.FORMAL_Q16_GRID[0],
            "q16_macro_span": platform.FORMAL_Q16_GRID[1],
            "aero_nchord": platform.FORMAL_AERO_GRID[0],
            "aero_nspan": platform.FORMAL_AERO_GRID[1],
            "rho_air": case.fluid_density_kg_m3,
            "nu_air": case.kinematic_viscosity_m2_s,
            "U_inf": case.freestream_m_s,
            "Re": case.reynolds,
            "alpha_deg": case.angle_deg,
            "c": case.chord_m,
            "b": case.span_m,
            "S": case.reference_area_m2,
            "E": case.young_modulus_pa,
            "nu_s": case.poisson_ratio_assumed,
            "rho_m": case.membrane_density_kg_m3,
            "thickness": case.thickness_m,
            "prestress": case.prestress_n_m_assumed,
            "damping_model": (
                "kelvin_voigt_stiffness_theta_K (theta = eta/omega_lockin)"
            ),
            "damping_loss_factor": case.structural_damping_loss_factor,
            "damping_evidence_role": self.spec.damping_evidence_role,
            "dt_star": case.aerodynamic_dt_star,
            "structural_substeps": substeps,
            "startup_window_time_star": case.startup_time_star,
            "statistics_min_time_star": case.statistics_min_time_star,
            "startup_ramp": (
                "half-cosine 0->U0 over t* in [0,1]; excluded from statistics"
            ),
            "static_motion_contract": platform.static_motion_contract(case),
            "separated_lev_mandatory": True,
            "lesp_release_condition": (
                "3D LESP separation eligibility AND 2D source-bank release "
                "trigger (release_owner_conflicts must stay 0)"
            ),
            "Lcrit": case.lesp_crit,
            "joint_tev": True,
            "free_wake": True,
            "wake_max_rows": WAKE_MAX_ROWS,
            "wake_free_rows": WAKE_FREE_ROWS,
            "particle_capacity": PARTICLE_CAPACITY,
            "particle_max_age_steps": PARTICLE_MAX_AGE_STEPS,
            "wake_history_mode": "bound_rate",
            "reference_tangent_refresh_rtol": (
                self.reference_tangent_refresh_rtol
            ),
            "reference_tangent_cache_refresh_count": (
                self.structural.reference_tangent_cache_refresh_count
            ),
            "dvm_target_spacing_chord": DVM_TARGET_SPACING_CHORD,
            "perimeter_audit": self.perimeter_audit,
            "assumption_ledger": platform.assumption_ledger(case),
            "proposal_count": counters.aero_proposal_count,
            "formal_replay_count": counters.formal_replay_count,
            "commit_count": counters.commit_count,
            "discarded_trial_count": counters.discarded_trial_count,
            "rejected_trial_note": (
                "true transaction counting: discarded = aerodynamic "
                "evaluations - one formal replay per committed step"
            ),
            "physics_contract": {
                "passed": physics_ok,
                "violations": physics_violations[:20],
            },
            "transfer_gates": {
                "max_errors": transfer_max,
                "tolerance": TRANSFER_ERROR_TOLERANCE,
                "passed": transfer_ok,
            },
            "retention_ledger": (
                self._retention_totals(records) if records else None
            ),
            "window_selection": (
                {
                    **window_report,
                    "window": [
                        window_report["window"].start,
                        window_report["window"].stop,
                    ],
                }
                if window_report is not None
                else None
            ),
            "mean_Cn": statistics["mean_cn"] if statistics else None,
            "mean_zmax_over_c": (
                statistics["zmax_over_c_mean_map"] if statistics else None
            ),
            "zsd_max_over_c": (
                statistics["zsd_max_over_c"] if statistics else None
            ),
            "zsd_map": statistics["zsd_map_over_c"] if statistics else None,
            "mean_map": statistics["mean_map_over_c"] if statistics else None,
            "dominant_St": (
                statistics["fluctuation_spectrum"]["strouhal"]
                if statistics and statistics.get("fluctuation_spectrum")
                else None
            ),
            "chordwise_peak_count": (
                statistics["chordwise_peak_count"] if statistics else None
            ),
            "spanwise_peak_count": (
                statistics["spanwise_peak_count"] if statistics else None
            ),
            "sign_crossing": (
                statistics["sign_crossing_at_station"] if statistics else None
            ),
            "accuracy_gates": accuracy_gates,
            "wake_ring_count": (
                records[-1]["aerodynamic"]["wake_ring_count"] if records else None
            ),
            "lev_release_count_total": int(
                sum(
                    record["aerodynamic"]["lev_release_count"]
                    for record in records
                )
            ),
            "max_abs_lesp_pre": max(
                (
                    record["aerodynamic"]["lesp_pre_max_abs"]
                    for record in records
                ),
                default=None,
            ),
            "max_abs_kelvin_residual": max(
                (
                    record["aerodynamic"]["kelvin_max_abs"]
                    for record in records
                ),
                default=None,
            ),
            "coupling_iteration_mean": (
                float(
                    np.mean(
                        [record["coupling_iterations"] for record in records]
                    )
                )
                if records
                else None
            ),
            "step_wall_seconds_mean": (
                float(np.mean(step_wall_times)) if step_wall_times else None
            ),
            "step_wall_seconds_last": (
                step_wall_times[-1] if step_wall_times else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "resolved_load_evidence": resolved_evidence,
            "aero_steps": len(records),
            "execution_gate_only": execution_gate_only,
            "records": records,
        }
        if failure is not None:
            payload.update(failure)
        if output is not None:
            _write_json(output, payload)
            _write_npz(
                output.with_name(f"{output.stem}.z_history.npz"),
                {
                    "z_history_over_c": (
                        z_history[: len(records)].cpu().numpy() / case.chord_m
                    ),
                    "time_star": np.array(
                        [
                            record["time_star"] for record in records
                        ],
                        dtype=np.float64,
                    ),
                    "mean_pressure_map": (
                        (pressure_sum / max(len(records), 1))
                        .reshape(self.surface.nc, self.surface.ns)
                        .cpu()
                        .numpy()
                    ),
                },
            )
            _write_json(
                output.with_name(f"{output.stem}.partial.json"),
                {
                    "schema": SCHEMA + "-checkpoint",
                    "status": payload["status"],
                    "case_id": case.case_id,
                    "result_status": payload["result_status"],
                    "completed_aero_steps": len(records),
                    "final_output": str(output),
                    "elapsed_seconds": elapsed_seconds,
                },
            )
        return payload

    @staticmethod
    def _retention_totals(records: list[dict[str, Any]]) -> dict[str, Any]:
        particle_culls = sum(
            record["aerodynamic"].get("particle_cull_count", 0)
            for record in records
        )
        wake_truncations = sum(
            record["aerodynamic"].get("wake_truncate_ring_count", 0)
            for record in records
        )
        particle_circulation_removed = float(
            sum(
                record["aerodynamic"].get("particle_cull_circulation", 0.0)
                for record in records
            )
        )
        wake_circulation_removed = float(
            sum(
                record["aerodynamic"].get("wake_truncate_circulation", 0.0)
                for record in records
            )
        )
        live = records[-1] if records else None
        return {
            "particle_cull_count_total": particle_culls,
            "wake_truncate_ring_count_total": wake_truncations,
            "particle_cull_circulation_total": particle_circulation_removed,
            "wake_truncate_circulation_total": wake_circulation_removed,
            "retained_circulation_live": (
                {
                    "gamma_bound": live["gamma_bound_sum"],
                    "wake_gamma": live["wake_gamma_sum"],
                    "particle_circulation": live["particle_circulation_sum"],
                }
                if live
                else None
            ),
            "removed_over_retained": (
                abs(particle_circulation_removed + wake_circulation_removed)
                / max(
                    abs(
                        live["gamma_bound_sum"]
                        + live["wake_gamma_sum"]
                        + live["particle_circulation_sum"]
                    ),
                    1.0e-30,
                )
                if live
                else None
            ),
        }


UNIFIED_CASE_INDEX: dict[str, RojratsirikulCaseConfig] = {
    **ROJRATSIRIKUL2011_UNIFIED_CASES,
    **ROJRATSIRIKUL2011_SENSITIVITY_BRANCHES,
}


__all__ = [
    "EXECUTION_GATE_STEPS",
    "RojratsirikulCaseRunner",
    "UNIFIED_CASE_INDEX",
    "apply_freestream",
    "select_statistics_window",
]
