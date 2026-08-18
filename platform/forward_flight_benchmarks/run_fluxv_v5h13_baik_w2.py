"""No-observation runner for the V5H13 Baik-W2 execution-only repair.

The implementation itself has no Ptera dependency.  The real coupling is
represented by :class:`FormalCouplingExecutor`; until the audited dependency
hashes are bound, :func:`run_formal_attempt` publishes a checksummed STOP
bundle before invoking that callback.  The artifact writer and verifier are
fully usable with synthetic protocol records and never call a solver, source,
scorer, or paper observation.  (Importing through the surrounding package may
still execute that package's existing initialiser.)
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import csv
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import errno
from hashlib import sha256
import io
from importlib import metadata as importlib_metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Final, Mapping, Protocol, Sequence
from uuid import uuid4

import numpy as np


SYNTHETIC_EXECUTION_MODE: Final = "synthetic_fixture"
FORMAL_EXECUTION_MODE: Final = "formal"
EXECUTION_MODES: Final = (SYNTHETIC_EXECUTION_MODE, FORMAL_EXECUTION_MODE)
SUMMARY_SCHEMA_BY_MODE: Final = {
    SYNTHETIC_EXECUTION_MODE: "fluxv-v5h13-baik-w2-synthetic-summary-v2",
    FORMAL_EXECUTION_MODE: "fluxv-v5h13-baik-w2-formal-summary-v2",
}
RUN_MANIFEST_SCHEMA_BY_MODE: Final = {
    SYNTHETIC_EXECUTION_MODE: "fluxv-v5h13-baik-w2-synthetic-run-manifest-v2",
    FORMAL_EXECUTION_MODE: "fluxv-v5h13-baik-w2-formal-run-manifest-v2",
}
TRAJECTORY_SCHEMA_BY_MODE: Final = {
    SYNTHETIC_EXECUTION_MODE: "fluxv-v5h13-baik-w2-synthetic-trajectories-v2",
    FORMAL_EXECUTION_MODE: "fluxv-v5h13-baik-w2-formal-trajectories-v2",
}
CONVERGENCE_SCHEMA_BY_MODE: Final = {
    SYNTHETIC_EXECUTION_MODE: "fluxv-v5h13-baik-w2-synthetic-convergence-v2",
    FORMAL_EXECUTION_MODE: "fluxv-v5h13-baik-w2-formal-convergence-v2",
}
DEPENDENCY_MANIFEST_SCHEMA_ID: Final = "fluxv-v5h13-dependency-manifest-v2"
DEPENDENCY_AUDIT_TOKEN_SCHEMA_ID: Final = "fluxv-v5h13-dependency-audit-token-v1"
DEPENDENCY_SET_SCOPE: Final = "bounded_declared_leaf_set_plus_runtime_module_inventory"
AUDITED_RUNTIME_MODULE_PREFIXES: Final = (
    "fluxvortex",
    "forward_flight_benchmarks",
    "pterasoftware",
    "ldvm_fourier",
    "flap_ldvm",
)
FORMAL_EXECUTOR_FACTORY_NAME: Final = "build_fluxv_v5h13_w2_executor"
FORMAL_AUDIT_SCOPE: Final = "v5h13_execution_repair_no_gt_formal_execution"
CASE_ID: Final = "W2"
OBSERVATION_ACCESS: Final = "none"
FORCE_SCORING_STATUS: Final = "blocked_no_gt_inner_mechanics_only"
BIRTH_WINDOW_K: Final = 5
BIRTH_WINDOW_R: Final = 4
FORMAL_LEVELS: Final = (
    32 - BIRTH_WINDOW_K + BIRTH_WINDOW_K * BIRTH_WINDOW_R,
    64 - BIRTH_WINDOW_K + BIRTH_WINDOW_K * BIRTH_WINDOW_R,
    128 - BIRTH_WINDOW_K + BIRTH_WINDOW_K * BIRTH_WINDOW_R,
)
FORMAL_NOMINAL_LEVELS: Final = (32, 64, 128)


def graded_substep_delta_time(level: int, substep: int) -> float:
    """The frozen V5H13 birth-window dt of one graded sub-step slot."""

    nominal = level - BIRTH_WINDOW_K * BIRTH_WINDOW_R + BIRTH_WINDOW_K
    if substep <= BIRTH_WINDOW_K * BIRTH_WINDOW_R:
        return DELTA_TIME_S / (nominal * BIRTH_WINDOW_R)
    return DELTA_TIME_S / nominal


LAYERS: Final = (1, 2, 3)
SOURCE_STEPS: Final = (4, 5, 6)
PTERA_STEPS: Final = (3, 4, 5)
PANEL_IDS: Final = tuple(
    f"airplane:0/wing:0/chord:{chord}/span:{span}"
    for chord in range(2)
    for span in range(8)
)
TOTAL_PANEL_ID: Final = "TOTAL"
FRONTIER_NODE_IDS: Final = tuple(f"frontier-node:{index}" for index in range(9))
FIXED_PROBES_GP1_M: Final = (
    (0.05, 0.30, 0.30),
    (0.10, 0.15, 0.40),
    (0.20, 0.45, 0.50),
)
FIXED_PROBE_SOURCE_SHA256: Final = {
    "run_fluxv_v5h7_temporal_topology_oracle.py": (
        "22a90cc17cd8e47d212ebcc2d7f71bdb8108cd3a0e61a8a3d2310bede9127ac9"
    ),
    "run_fluxv_v5h8_incremental_sheet_oracle.py": (
        "f2079cb457b883fd35892434745d8a3e96c6a6587ba23f1caf710193723f6e98"
    ),
}
DELTA_TIME_S: Final = 0.11125
STATE_RELATIVE_L2_MAX: Final = 1.0e-6
ACTIVE_GAMMA_MAXABS_MIN: Final = math.sqrt(np.finfo(np.float64).tiny)
PROBE_RELATIVE_L2_MAX: Final = 1.0e-4
PROBE_DIRECT_ORACLE_RTOL: Final = 512.0 * np.finfo(np.float64).eps
PROBE_DIRECT_ORACLE_ATOL: Final = 512.0 * np.finfo(np.float64).eps
LOAD_RELATIVE_L2_MAX: Final = 2.0e-3
MIN_DIFFERENCE_REDUCTION_RATIO: Final = 1.5
ROUNDOFF_DIFFERENCE_MAX: Final = 1.0e-14
NO_PENETRATION_MAX: Final = 1.0e-12
KELVIN_RESIDUAL_MAX: Final = 1.0e-10
STAGE_INVARIANT_RESIDUAL_OVER_SLOG_MAX: Final = 512.0 * np.finfo(np.float64).eps
STAGE_H_JACOBIAN_FROBENIUS_MAX: Final = 1.5
STAGE_H_GALILEAN_OVER_SIGMA_MAX: Final = 0.5
STREAM_STAGE_CHAIN_GENESIS: Final = sha256(
    b"fluxv-ir-wrk3-stream-stage-chain-v1"
).hexdigest()

ARTIFACT_FILES: Final = (
    "raw_steps.csv",
    "source_events.csv",
    "owner_events.jsonl",
    "particle_counts.csv",
    "raw_loads.csv",
    "transport_stages.csv",
    "trajectory_arrays.json",
    "convergence.json",
    "summary.json",
    "run_manifest.json",
    "SHA256SUMS",
    "run.log",
)
SEMANTIC_FILES: Final = ARTIFACT_FILES[:9]
RAW_REPLAY_INPUT_FILES: Final = ARTIFACT_FILES[:7]

# Formal readiness is external and acyclic: an audit token hashes a dependency
# manifest, and that manifest hashes only leaf files.  No source file embeds its
# own digest and neither the token nor manifest is allowed to appear as a leaf.
DEFAULT_DEPENDENCY_AUDIT_TOKEN: Final[Path | None] = None
REQUIRED_DEPENDENCY_LEAVES: Final = (
    "prereg_freeze",
    "runner",
    "runner_test",
    "formal_executor_module",
    "formal_executor_test",
    "probe_source_v5h7",
    "probe_source_v5h8",
    "b3_execution_amendment",
    "rvpm_ir_wrk3",
    "rvpm_ir_wrk3_fd_adapter",
    "rvpm_ir_wrk3_v5h13_stream",
    "rvpm_reference",
    "v5h10_row_owner",
    "v5h13_baik_coupling",
    "v5h3_native_feedback",
    "v5h4_ptera_transport",
    "v5h_dvm_source",
    "v5h_node_placement",
    "rvpm_edge_bridge",
    "rvpm_transport",
    "baik2012",
    "ldvm_fourier",
    "uvlm_correction",
    "pterasoftware_distribution_metadata",
    "pterasoftware_solver_source",
    "fluxvortex_distribution_metadata",
    "fluxvortex_solver_source",
    "v5h2_dyadic_cumulative_cloud_transport",
    "v5h_cumulative_cloud_transport",
    "rvpm_dyadic_edge_bridge",
    "causal_incidence_owner",
    "ptera_adapter",
    "uvlm_polar_correction",
    "fluxvortex_particles",
    "fluxvortex_kernel",
    "flap_ldvm",
    "v5h_node_ribbon",
    "v5h_passive_frontier_transport",
    "forward_flight_cases",
    "numpy_distribution_metadata",
    "scipy_distribution_metadata",
)

# These in-tree identities are not inferred from an audit-supplied label.  A
# formal manifest must resolve each named project leaf to this exact path in
# the executing checkout.  The three not-yet-bound external deliverables
# (executor module/test and B3 amendment) and installed-distribution leaves
# are additionally constrained below by their frozen path identities.
CANONICAL_PROJECT_LEAF_RELATIVE_PATH: Final = {
    "prereg_freeze": (
        "docs/forward_flight_large_pitch/reproductions/"
        "fluxv_v5_nextgen_20260814/refine-logs/v5h13_birth_window_timegrid/"
        "FREEZE_INPUTS.json"
    ),
    "runner": "platform/forward_flight_benchmarks/run_fluxv_v5h13_baik_w2.py",
    "runner_test": "platform/tests/test_run_fluxv_v5h13_baik_w2.py",
    "formal_executor_module": (
        "platform/forward_flight_benchmarks/fluxv_v5h13_baik_w2_executor.py"
    ),
    "formal_executor_test": "platform/tests/test_fluxv_v5h13_baik_w2_executor.py",
    "probe_source_v5h7": (
        "platform/forward_flight_benchmarks/"
        "run_fluxv_v5h7_temporal_topology_oracle.py"
    ),
    "probe_source_v5h8": (
        "platform/forward_flight_benchmarks/"
        "run_fluxv_v5h8_incremental_sheet_oracle.py"
    ),
    "b3_execution_amendment": (
        "docs/forward_flight_large_pitch/reproductions/"
        "fluxv_v5_nextgen_20260814/refine-logs/v5h11_ir_wrk3/"
        "B3_EXECUTION_AMENDMENT_20260816.md"
    ),
    "rvpm_ir_wrk3": "src/fluxvortex/rvpm_ir_wrk3.py",
    "rvpm_ir_wrk3_fd_adapter": "src/fluxvortex/rvpm_ir_wrk3_fd_adapter.py",
    "rvpm_ir_wrk3_v5h13_stream": "src/fluxvortex/rvpm_ir_wrk3_v5h13_stream.py",
    "rvpm_reference": "src/fluxvortex/rvpm_reference.py",
    "v5h10_row_owner": ("platform/forward_flight_benchmarks/fluxv_v5h10_row_owner.py"),
    "v5h13_baik_coupling": (
        "platform/forward_flight_benchmarks/fluxv_v5h13_baik_coupling.py"
    ),
    "v5h3_native_feedback": (
        "platform/forward_flight_benchmarks/fluxv_v5h3_native_feedback.py"
    ),
    "v5h4_ptera_transport": (
        "platform/forward_flight_benchmarks/fluxv_v5h4_ptera_rvpm_transport.py"
    ),
    "v5h_dvm_source": "platform/forward_flight_benchmarks/v5h_dvm_source.py",
    "v5h_node_placement": (
        "platform/forward_flight_benchmarks/v5h_dvm_node_placement.py"
    ),
    "rvpm_edge_bridge": "src/fluxvortex/rvpm_edge_bridge.py",
    "rvpm_transport": "src/fluxvortex/rvpm_transport.py",
    "baik2012": "platform/forward_flight_benchmarks/baik2012.py",
    "ldvm_fourier": "platform/ldvm_fourier.py",
    "uvlm_correction": ("platform/forward_flight_benchmarks/ldvm_uvlm_correction.py"),
    "fluxvortex_solver_source": "src/fluxvortex/solver.py",
    "v5h2_dyadic_cumulative_cloud_transport": (
        "platform/forward_flight_benchmarks/"
        "v5h2_dyadic_cumulative_cloud_transport.py"
    ),
    "v5h_cumulative_cloud_transport": (
        "platform/forward_flight_benchmarks/v5h_cumulative_cloud_transport.py"
    ),
    "rvpm_dyadic_edge_bridge": "src/fluxvortex/rvpm_dyadic_edge_bridge.py",
    "causal_incidence_owner": (
        "platform/forward_flight_benchmarks/causal_incidence_owner.py"
    ),
    "ptera_adapter": "platform/forward_flight_benchmarks/ptera_adapter.py",
    "uvlm_polar_correction": (
        "platform/forward_flight_benchmarks/uvlm_polar_correction.py"
    ),
    "fluxvortex_particles": "src/fluxvortex/particles.py",
    "fluxvortex_kernel": "src/fluxvortex/kernel.py",
    "flap_ldvm": "platform/flap_ldvm.py",
    "v5h_node_ribbon": ("platform/forward_flight_benchmarks/v5h_dvm_node_ribbon.py"),
    "v5h_passive_frontier_transport": (
        "platform/forward_flight_benchmarks/v5h_passive_frontier_transport.py"
    ),
    "forward_flight_cases": "platform/forward_flight_benchmarks/cases.py",
}
EXTERNAL_DEPENDENCY_PATH_SUFFIX: Final = {
    "pterasoftware_solver_source": (
        "pterasoftware/unsteady_ring_vortex_lattice_method.py"
    ),
}
DISTRIBUTION_LEAF_TO_NAME: Final = {
    "pterasoftware_distribution_metadata": "pterasoftware",
    "fluxvortex_distribution_metadata": "fluxvortex",
    "numpy_distribution_metadata": "numpy",
    "scipy_distribution_metadata": "scipy",
}

RAW_STEP_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "status",
    "particle_count",
    "material_tracer_count",
    "material_support_tracer_count",
    "frontier_node_tracer_count",
    "stream_result_sha256",
    "stream_stage_chain_sha256",
    "fd_ledger_sha256",
    "load_ledger_sha256",
    "layer_result_sha256",
    "row_owner_before_sha256",
    "advanced_owner_sha256",
    "ptera_parent_sha256_before",
    "ptera_parent_sha256_after",
    "no_penetration_max_abs",
    "kelvin_residual_max_abs",
    "raw_cl",
    "raw_cd",
    "direct_field_call_count",
    "ptera_center_call_count",
    "ptera_offset_call_count",
    "fd_physical_evaluation_count",
    "fd_tracer_evaluation_count",
    "fd_evaluator_call_count",
    "transport_substep_count",
    "transport_stage_count",
    "physical_field_call_count",
    "tracer_field_call_count",
    "observer_call_count",
    "stage_pre_reconstruction_count",
    "stage_post_reconstruction_count",
    "physical_rhs_call_count",
    "storage_reset_count",
    "tracer_storage_reset_count",
    "invariant_reference_freeze_count",
    "sigma_storage_update_count",
    "relaxation_call_count",
    "compact_stage_record_count",
    "retained_stage_array_count",
    "evidence_byte_count",
    "native_collocation_evaluation_count",
    "native_load_batch_evaluation_count",
    "native_load_call_count",
    "transport_macro_call_count",
    "row_commit_count",
)

SOURCE_EVENT_FIELDS: Final = (
    "transport_substeps",
    "source_step_index",
    "ptera_step_index",
    "source_time_s",
    "cell_count",
    "status",
    "birth_modes",
    "a0_pre",
    "a0_post",
    "gamma_lev_new_m2_s",
    "gamma_lev_persisted_m2_s",
    "gamma_tev_new_m2_s",
    "kelvin_residual_m2_s",
    "event_sha256",
    "parent_event_sha256",
)
SOURCE_VECTOR_FIELDS: Final = (
    "a0_pre",
    "a0_post",
    "gamma_lev_new_m2_s",
    "gamma_lev_persisted_m2_s",
    "gamma_tev_new_m2_s",
    "kelvin_residual_m2_s",
)
SOURCE_INTERFACE_ID: Final = "fluxv-v5h-dvm-source-only-v3"
SOURCE_PLACEMENT_SCHEMA_ID: Final = "fluxv-v5h-dvm-source-placement-v3"
SOURCE_BACKEND_ID: Final = (
    "platform.ldvm_fourier.LDVM2D-source-parity-clean-linear-provisional-tev-v3"
)
SOURCE_EVENT_CHAIN_DOMAIN: Final = "fluxv-v5h-dvm-event-chain-v3"
SOURCE_EVENT_DIGEST_PREFIX: Final = b"fluxv-v5h-direct-dvm-event-v3\0"
SOURCE_POSITION_FRAME: Final = (
    "2D inertial LDVM world frame, x downstream and z up, origin at "
    "the section pivot at t*=0; birth position is pre-convection"
)
SOURCE_CIRCULATION_SIGN: Final = (
    "positive is counter-clockwise in the x-z plane; this is the "
    "negative of the author-Fortran clockwise strength"
)
SOURCE_TEV_BIRTH_LAW: Final = (
    "source-parity TEV: first source column is TE + 0.5*U*dt and "
    "its solved strength is zeroed before persistence; subsequent "
    "columns are TE + (previous newest TEV - TE)/3"
)
SOURCE_LEV_BIRTH_LAW: Final = (
    "source-parity LEV: first/restart is LE + 0.5*v_LE*dt with "
    "v_LE induced by old wake plus the same-step provisional TEV; "
    "continuous shedding is LE + (previous newest LEV - LE)/3"
)
SOURCE_BIRTH_TIME_LAYER: Final = (
    "birth coordinates are captured from the actual pre-convection "
    "constraint columns at t_n"
)
SOURCE_DIMENSIONALIZATION_LIMITATIONS: Final = (
    "multiply circulation by U_ref*c_ref and position by c_ref; no "
    "span coordinate, edge endpoints, strip width, or 3D orientation "
    "is supplied, so a separately gated node-owned mapper is required"
)
SOURCE_OWNERSHIP_SCOPE: Final = "source quantities only; FluxV/Ptera is sole load owner"
SOURCE_CANONICAL_BLOCKER: Final = (
    "strict D0 source-proof artifact has not been independently consumed by "
    "the D1 integration gate; local 174-LEV event parity alone is insufficient"
)
SOURCE_BOTTOM_MODEL_PARITY: Final = (
    "local strict source-parity regression reproduces the author event "
    "mask (174 LEVs, output rows 116..289) and first-TEV persistence; "
    "strengths use a clean linear solve rather than author Newton, and "
    "the independent D0 proof has not yet been consumed"
)
FORMAL_W2_THRESHOLD_SOURCE: Final = (
    "Ramesh 2013 thesis flat-plate Re=1000 sections 4.3.5 and "
    "Figures 4.19/4.21 use Lcrit=0.11; frozen cross-Re/thickness "
    "transfer hypothesis with no Baik force fit"
)
FORMAL_W2_GEOMETRY_IDENTITY: Final = (
    "explicit zero-camber rounded-flat-plate mean-line surrogate"
)
FORMAL_W2_GEOMETRY_SHA256: Final = (
    "0625b6a543002743d4ea38b68ac69bbc5016956bef9a4637063b691c1ac46810"
)
FORMAL_W2_REFERENCE_SPEED_M_S: Final = float(np.pi * (1.0 / 3.56) * 0.076)
FORMAL_W2_STEPS_PER_CYCLE: Final = 32
FORMAL_W2_CONVECTIVE_DT: Final = float(
    FORMAL_W2_REFERENCE_SPEED_M_S * 3.56 / 0.076 / FORMAL_W2_STEPS_PER_CYCLE
)
FORMAL_W2_CIRCULATION_SCALE: Final = float(FORMAL_W2_REFERENCE_SPEED_M_S * 0.076)

OWNER_EVENT_FIELDS: Final = (
    "schema_id",
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "status",
    "row_owner_before_sha256",
    "row_state_before_sha256",
    "common_transport_sha256",
    "transport_attestation_sha256",
    "transport_parent_digest",
    "advanced_owner_sha256",
    "advanced_state_sha256",
    "changed_particle_ids",
    "appended_particle_ids",
    "commit_event",
    "transport_event",
)
COMMIT_EVENT_FIELDS: Final = (
    "proposal_id",
    "release_index",
    "changed_indices",
    "appended_indices",
    "parent_state_sha256",
    "parent_transport_digest",
    "parent_transport_event_sha256",
    "upstream_nodes_sha256",
    "committed_state_sha256",
    "row_sha256",
    "before_gamma_sha256",
    "added_gamma_sha256",
    "after_gamma_sha256",
    "operator_order",
    "global_graph_build_count",
    "clone_count",
    "counter_particle_count",
    "fresh_upstream_particle_count",
    "remesh_count",
    "ptera_call_count",
    "load_call_count",
    "feedback_call_count",
    "transport_call_count",
    "parent_owner_sha256",
    "previous_event_sha256",
    "event_sha256",
)
TRANSPORT_EVENT_FIELDS: Final = (
    "release_index",
    "source_step_index",
    "transport_end_time_s",
    "parent_state_sha256",
    "transported_state_sha256",
    "parent_transport_digest",
    "common_transport_attestation_sha256",
    "transported_arrays_sha256",
    "live_boundary_nodes_sha256",
    "previous_transport_event_sha256",
    "transport_event_sha256",
)

PARTICLE_COUNT_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "status",
    "particle_count",
    "material_tracer_count",
    "material_support_tracer_count",
    "frontier_node_tracer_count",
    "changed_particle_count",
    "appended_particle_count",
)

RAW_LOAD_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "scope",
    "panel_id",
    "force_x_n",
    "force_y_n",
    "force_z_n",
    "moment_x_nm",
    "moment_y_nm",
    "moment_z_nm",
    "force_coefficient_x",
    "force_coefficient_y",
    "force_coefficient_z",
    "raw_cl",
    "raw_cd",
)

STAGE_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "substep",
    "stage",
    "status",
    "substep_delta_time",
    "rk_a",
    "rk_b",
    "pre_state_sha256",
    "post_state_sha256",
    "tracer_pre_sha256",
    "tracer_post_sha256",
    "velocity_sha256",
    "jacobian_sha256",
    "gamma_rate_sha256",
    "tracer_velocity_sha256",
    "invariant_residual_sha256",
    "invariant_residual_over_slog_max",
    "h_jacobian_frobenius",
    "h_convective_over_sigma",
    "position_storage_pre_sha256",
    "gamma_storage_pre_sha256",
    "tracer_storage_pre_sha256",
    "position_storage_post_sha256",
    "gamma_storage_post_sha256",
    "tracer_storage_post_sha256",
    "fd_physical_evaluation_sha256",
    "fd_tracer_evaluation_sha256",
    "ptera_parent_sha256_before",
    "ptera_parent_sha256_after",
    "direct_field_call_count",
    "ptera_center_call_count",
    "ptera_offset_call_count",
    "stream_record_sha256",
    "previous_chain_sha256",
    "chain_sha256",
    "failure_type",
    "failure_message",
)

REQUIRED_TRAJECTORY_ARRAYS: Final = (
    "start_positions",
    "start_gamma",
    "start_sigma",
    "end_positions",
    "end_gamma",
    "end_sigma",
    "material_tracer_positions",
    "frontier_tracer_positions",
    "probe_velocity",
    "probe_jacobian",
    "force",
    "moment",
    "invariant_start",
    "invariant_end",
    "no_penetration_residual",
)
TRAJECTORY_METADATA_FIELDS: Final = (
    "source_cell_manifest_sha256",
    "source_cell_manifests",
    "source_prehistory_manifest_sha256",
    "source_prehistory_manifests",
    "source_kelvin_ledger_sha256",
    "source_kelvin_evidence_sha256",
    "particle_id_sequence_sha256",
    "material_tracer_id_sequence_sha256",
    "frontier_start_positions_sha256",
    "fixed_probe_contract",
)
SOURCE_CELL_MANIFEST_FIELDS: Final = (
    "enabled",
    "status",
    "delta_time_convective",
    "a0_pre",
    "a0_post",
    "lesp_critical",
    "lesp_active",
    "lesp_signed_target",
    "lesp_constraint_residual",
    "lev_birth_mode",
    "restart",
    "gamma_lev_new_over_u_c",
    "gamma_tev_new_solved_over_u_c",
    "gamma_tev_new_persisted_over_u_c",
    "lev_placement",
    "tev_placement",
    "lev_birth_position_over_chord_backend_world",
    "tev_birth_position_over_chord_backend_world",
    "kelvin_residual_over_u_c",
    "kelvin_ledger",
    "lineage",
    "provenance",
    "parent_event_manifest_sha256",
    "producer_manifest_sha256",
)
SOURCE_PLACEMENT_FIELDS: Final = (
    "schema_id",
    "vortex_family",
    "placement_mode",
    "edge_anchor_position_over_chord_backend_world",
    "birth_position_over_chord_backend_world",
    "birth_displacement_from_edge_over_chord_backend_world",
    "q_birth_over_u_backend_world",
    "q_kinematic_over_u_backend_world",
    "q_old_wake_over_u_backend_world",
    "q_provisional_tev_over_u_backend_world",
    "continuous_parent_source_id",
    "continuous_parent_position_over_chord_backend_world",
    "used_for_topology_eligible",
)
SOURCE_KELVIN_LEDGER_FIELDS: Final = (
    "circulation_units",
    "gamma_bound_post",
    "gamma_old_tev_persisted",
    "gamma_old_lev_persisted",
    "gamma_deleted_before",
    "gamma_tev_new_te_only_provisional",
    "gamma_tev_new_solved",
    "gamma_tev_new_persisted",
    "gamma_lev_new_solved",
    "gamma_lev_new_persisted",
    "gamma_deleted_after",
    "gamma_deleted_delta",
    "gamma_tev_persisted_after",
    "gamma_lev_persisted_after",
    "tev_solved_to_persisted_delta",
    "first_tev_zeroed",
    "kelvin_solve_residual",
    "persistence_residual",
)
SOURCE_PROVENANCE_FIELDS: Final = (
    "interface_id",
    "backend_id",
    "physical_section_id",
    "physical_strip_id",
    "section_family",
    "reynolds",
    "lesp_critical",
    "threshold_source",
    "threshold_source_role",
    "delta_time_convective_nominal",
    "pivot_fraction_chord",
    "ndiv",
    "naterm",
    "resolved_core_radius_chord",
    "max_wake_steps",
    "geometry_identity",
    "geometry_hash_sha256",
    "geometry_role",
    "geometry_station_count",
    "circulation_units",
    "circulation_scale_u_times_c_m2_per_s",
    "position_units",
    "position_scale_chord_m",
    "position_frame",
    "circulation_sign",
    "tev_birth_law",
    "lev_birth_law",
    "birth_time_layer",
    "dimensionalization_limitations",
    "source_parity",
    "source_solver",
    "canonical",
    "canonical_blocker",
    "bottom_model_parity",
    "ownership_scope",
    "observation_access",
    "target_case_branch",
)
SOURCE_LINEAGE_FIELDS: Final = (
    "physical_section_id",
    "physical_strip_id",
    "section_lineage_id",
    "source_step_index",
    "parent_state_step_index",
    "newborn_tev_source_id",
    "newborn_lev_source_id",
    "newborn_tev_role",
    "newborn_lev_role",
    "persistent_tev_history_role",
    "persistent_lev_history_role",
    "persistent_tev_count_before",
    "persistent_tev_count_after",
    "persistent_lev_count_before",
    "persistent_lev_count_after",
    "persistent_history_exported",
)

TERMINAL_COORDINATE_FIELDS: Final = (
    "transport_substeps",
    "layer",
    "source_step_index",
    "ptera_step_index",
    "substep",
    "stage",
    "phase",
    "stage_began",
)
CONVERSION_STOP_CODE: Final = "stage_evidence_conversion_error"
CONVERSION_PHASE: Final = "artifact_stage_conversion"

POST_MATRIX_PASSING_CONVERGENCE_STOP_PAIRS: Final = frozenset(
    {
        ("coupling_callback_error", "coupling_callback"),
        ("coupling_stop_contract_error", "coupling_callback"),
        ("coupling_callback_contract_error", "coupling_completion"),
        ("dependency_drift", "dependency_postflight"),
        ("post_matrix_runtime_failure", "runtime_postflight"),
        ("post_matrix_audit_failure", "audit_postflight"),
    }
)


def _expected_layer_keys() -> tuple[tuple[int, int], ...]:
    return tuple((level, layer) for level in FORMAL_LEVELS for layer in LAYERS)


def _expected_source_keys() -> tuple[tuple[int, int], ...]:
    return tuple((level, source) for level in FORMAL_LEVELS for source in SOURCE_STEPS)


def _expected_load_keys() -> tuple[tuple[int, int, str, str], ...]:
    rows: list[tuple[int, int, str, str]] = []
    for level, layer in _expected_layer_keys():
        rows.extend((level, layer, "panel", panel_id) for panel_id in PANEL_IDS)
        rows.append((level, layer, "total", TOTAL_PANEL_ID))
    return tuple(rows)


def _expected_stage_keys() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(
        (level, layer, substep, stage)
        for level in FORMAL_LEVELS
        for layer in LAYERS
        for substep in range(1, level + 1)
        for stage in (1, 2, 3)
    )


EXPECTED_LAYER_KEYS: Final = _expected_layer_keys()
EXPECTED_SOURCE_KEYS: Final = _expected_source_keys()
EXPECTED_LOAD_KEYS: Final = _expected_load_keys()
EXPECTED_STAGE_KEYS: Final = _expected_stage_keys()


class DependencyFreezeError(RuntimeError):
    """Formal execution is blocked until every dependency hash is frozen."""


class FormalRunStopped(RuntimeError):
    """Structured stop raised by the future real coupling executor."""

    def __init__(
        self,
        stop_code: str,
        terminal_coordinate: Mapping[str, object],
        message: str,
    ) -> None:
        super().__init__(message)
        self.stop_code = stop_code
        self.terminal_coordinate = dict(terminal_coordinate)


class FormalCouplingExecutor(Protocol):
    """The sole future boundary between this runner and live coupling code.

    The executor must create fresh N=32/64/128 trajectories and append durable
    primitive records to ``sink`` as work commits.  It may raise only after the
    sink contains the exact completed prefix.
    """

    def run_formal_matrix(
        self,
        *,
        levels: tuple[int, int, int],
        sink: "ArtifactSink",
    ) -> None:
        ...

    def attest_dependency_origins(self) -> None:
        """Fail closed if any already-loaded dependency differs from the audit."""

        ...


@dataclass(frozen=True, slots=True)
class ArtifactRecords:
    execution_mode: str
    status: str
    raw_steps: tuple[Mapping[str, object], ...]
    source_events: tuple[Mapping[str, object], ...]
    owner_events: tuple[Mapping[str, object], ...]
    particle_counts: tuple[Mapping[str, object], ...]
    raw_loads: tuple[Mapping[str, object], ...]
    transport_stages: tuple[Mapping[str, object], ...]
    trajectories: tuple[Mapping[str, object], ...]
    terminal_coordinate: Mapping[str, object] | None = None
    stop_code: str | None = None
    stop_message: str | None = None


class ArtifactSink:
    """Append-only JSON-primitive capture surface for the final executor.

    Every append is validated and frozen immediately as canonical JSON bytes.
    This both rejects NumPy/non-finite/foreign objects in ordinary rows and
    prevents later caller mutation from rewriting an already completed prefix.
    Trajectory arrays use :meth:`add_trajectory_array_record`, which copies and
    encodes ndarray raw bytes at that same boundary; :meth:`add_trajectory`
    accepts only already encoded and independently validated array payloads.
    """

    __slots__ = (
        "_raw_steps",
        "_source_events",
        "_owner_events",
        "_particle_counts",
        "_raw_loads",
        "_transport_stages",
        "_trajectories",
    )

    def __init__(self) -> None:
        self._raw_steps: list[bytes] = []
        self._source_events: list[bytes] = []
        self._owner_events: list[bytes] = []
        self._particle_counts: list[bytes] = []
        self._raw_loads: list[bytes] = []
        self._transport_stages: list[bytes] = []
        self._trajectories: list[bytes] = []

    @staticmethod
    def _append(target: list[bytes], row: Mapping[str, object]) -> None:
        if not isinstance(row, Mapping):
            raise TypeError("artifact record must be a mapping")
        target.append(_canonical_json_record_bytes(row))

    @staticmethod
    def _snapshot(target: Sequence[bytes]) -> tuple[Mapping[str, object], ...]:
        return tuple(_decode_canonical_json_record(payload) for payload in target)

    @property
    def raw_steps(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._raw_steps)

    @property
    def source_events(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._source_events)

    @property
    def owner_events(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._owner_events)

    @property
    def particle_counts(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._particle_counts)

    @property
    def raw_loads(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._raw_loads)

    @property
    def transport_stages(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._transport_stages)

    @property
    def trajectories(self) -> tuple[Mapping[str, object], ...]:
        return self._snapshot(self._trajectories)

    def add_raw_step(self, row: Mapping[str, object]) -> None:
        self._append(self._raw_steps, row)

    def add_source_event(self, row: Mapping[str, object]) -> None:
        payload = _canonical_json_record_bytes(row)
        frozen = _decode_canonical_json_record(payload)
        if set(frozen) != set(SOURCE_EVENT_FIELDS):
            raise ValueError("source_events row schema mismatch")
        if (
            frozen["status"] != "completed"
            or frozen["cell_count"] != 8
            or frozen["ptera_step_index"] != frozen["source_step_index"] - 1
        ):
            raise ValueError("source-event aggregate/status is invalid")
        candidate_sources = (*self.source_events, frozen)
        candidate_source_keys = tuple(_source_key(item) for item in candidate_sources)
        _assert_prefix("source", candidate_source_keys, EXPECTED_SOURCE_KEYS)
        _validate_source_semantics(candidate_sources)
        _validate_cross_table_progress(
            tuple(_layer_key(item) for item in self.raw_steps),
            candidate_source_keys,
            self.transport_stages,
        )
        self._source_events.append(payload)

    def add_owner_event(self, row: Mapping[str, object]) -> None:
        self._append(self._owner_events, row)

    def add_particle_count(self, row: Mapping[str, object]) -> None:
        self._append(self._particle_counts, row)

    def add_raw_load(self, row: Mapping[str, object]) -> None:
        self._append(self._raw_loads, row)

    def _add_transport_stage(self, row: Mapping[str, object]) -> None:
        self._append(self._transport_stages, row)

    def add_transport_stage_from_compact_evidence(
        self,
        row: Mapping[str, object],
        compact_evidence: Mapping[str, object],
    ) -> None:
        """Bind persisted stage gates directly to the coupling compact evidence."""

        if row.get("status") != "completed":
            raise ValueError("compact stage append accepts only completed stages")
        stability = compact_stage_stability_fields(compact_evidence)
        if any(row.get(name) != value for name, value in stability.items()):
            raise ValueError("stage row stability fields differ from compact evidence")
        payload = _canonical_json_record_bytes(row)
        frozen = _decode_canonical_json_record(payload)
        if set(frozen) != set(STAGE_FIELDS):
            raise ValueError("transport_stages row schema mismatch")
        candidate = (*self.transport_stages, frozen)
        _validate_stage_sequence(candidate, status="PREFIX", terminal=None)
        source_keys = tuple(_source_key(source) for source in self.source_events)
        _validate_cross_table_progress(
            tuple(_layer_key(item) for item in self.raw_steps),
            source_keys,
            candidate,
        )
        self._transport_stages.append(payload)

    def add_failed_transport_stage(self, row: Mapping[str, object]) -> None:
        """Append a terminal coordinate without inventing unavailable evidence."""

        if row.get("status") != "failed":
            raise ValueError("failed-stage append requires status=failed")
        payload = _canonical_json_record_bytes(row)
        frozen = _decode_canonical_json_record(payload)
        if set(frozen) != set(STAGE_FIELDS):
            raise ValueError("transport_stages row schema mismatch")
        coordinate = {
            "transport_substeps": frozen.get("transport_substeps"),
            "layer": frozen.get("layer"),
            "source_step_index": frozen.get("source_step_index"),
            "ptera_step_index": frozen.get("ptera_step_index"),
            "substep": frozen.get("substep"),
            "stage": frozen.get("stage"),
            "phase": "transport_stage",
            "stage_began": True,
        }
        source_keys = tuple(_source_key(source) for source in self.source_events)
        candidate = (*self.transport_stages, frozen)
        _validate_stage_sequence(
            candidate,
            status="STOP",
            terminal=_validate_terminal_coordinate(coordinate),
            durable_source_keys=source_keys,
        )
        _validate_cross_table_progress(
            tuple(_layer_key(item) for item in self.raw_steps),
            source_keys,
            candidate,
        )
        self._transport_stages.append(payload)

    def add_trajectory(self, row: Mapping[str, object]) -> None:
        self._append(
            self._trajectories,
            _validate_preencoded_trajectory_record(row),
        )

    def add_trajectory_array_record(self, row: Mapping[str, object]) -> None:
        """Freeze ndarray channels immediately without a Python-list round trip."""

        self.add_trajectory(_encode_trajectory_array_record(row))

    def commit_completed_layer(
        self,
        *,
        raw_step: Mapping[str, object],
        source_event: Mapping[str, object],
        owner_event: Mapping[str, object],
        particle_count: Mapping[str, object],
        raw_loads: Sequence[Mapping[str, object]],
        trajectory_array_record: Mapping[str, object],
    ) -> None:
        """Atomically append one layer bundle after its full stage prefix exists."""

        layer_index = len(self._raw_steps)
        if layer_index >= len(EXPECTED_LAYER_KEYS):
            raise ValueError("layer sink is already complete")
        expected_layer = EXPECTED_LAYER_KEYS[layer_index]
        expected_source = EXPECTED_SOURCE_KEYS[layer_index]
        if _layer_key(raw_step) != expected_layer:
            raise ValueError("atomic layer bundle is not the next layer key")
        if _source_key(source_event) != expected_source:
            raise ValueError("atomic layer bundle is not the next source key")
        current_sources = self.source_events
        if len(current_sources) != layer_index + 1:
            raise ValueError(
                "layer commit requires its durably pre-appended source event"
            )
        source_bytes = _canonical_json_record_bytes(source_event)
        if self._source_events[layer_index] != source_bytes:
            raise ValueError("precommitted source event differs from layer bundle")
        if any(
            _layer_key(row) != expected_layer
            for row in (owner_event, particle_count, trajectory_array_record)
        ):
            raise ValueError("atomic layer bundle keys disagree")
        if len(raw_loads) != 17:
            raise ValueError("atomic layer bundle requires exactly 17 load rows")
        expected_loads = EXPECTED_LOAD_KEYS[17 * layer_index : 17 * (layer_index + 1)]
        if tuple(_load_key(row) for row in raw_loads) != expected_loads:
            raise ValueError("atomic layer bundle load topology/order is invalid")

        stage_rows = self._snapshot(self._transport_stages)
        _validate_stage_sequence(stage_rows, status="PREFIX", terminal=None)
        level, layer = expected_layer
        required_stage_keys = {
            (level, layer, substep, stage)
            for substep in range(1, level + 1)
            for stage in (1, 2, 3)
        }
        completed_stage_keys = {
            _stage_key(row) for row in stage_rows if row["status"] == "completed"
        }
        if not required_stage_keys.issubset(completed_stage_keys):
            raise ValueError("atomic layer commit precedes all 3N completed stages")

        trajectory_encoded = _validate_preencoded_trajectory_record(
            _encode_trajectory_array_record(trajectory_array_record)
        )
        candidate_layer_keys = EXPECTED_LAYER_KEYS[: layer_index + 1]
        candidate_source_keys = EXPECTED_SOURCE_KEYS[: layer_index + 1]
        terminal_six = _expected_unbegun_terminal_six(
            completed_stage_count=len(stage_rows),
            layer_keys=candidate_layer_keys,
            source_keys=candidate_source_keys,
        )
        prepared = (
            _canonical_json_record_bytes(raw_step),
            source_bytes,
            _canonical_json_record_bytes(owner_event),
            _canonical_json_record_bytes(particle_count),
            tuple(_canonical_json_record_bytes(row) for row in raw_loads),
            _canonical_json_record_bytes(trajectory_encoded),
        )
        candidate = ArtifactRecords(
            execution_mode=SYNTHETIC_EXECUTION_MODE,
            status="STOP",
            raw_steps=(*self.raw_steps, dict(raw_step)),
            source_events=self.source_events,
            owner_events=(*self.owner_events, dict(owner_event)),
            particle_counts=(*self.particle_counts, dict(particle_count)),
            raw_loads=(*self.raw_loads, *(dict(row) for row in raw_loads)),
            transport_stages=stage_rows,
            trajectories=(*self.trajectories, trajectory_encoded),
            terminal_coordinate={
                **dict(zip(TERMINAL_COORDINATE_FIELDS[:6], terminal_six)),
                "phase": "layer_commit",
                "stage_began": False,
            },
            stop_code="layer_commit_validation",
            stop_message="",
        )
        _validate_records(candidate)

        (
            raw_bytes,
            source_bytes,
            owner_bytes,
            count_bytes,
            load_bytes,
            trajectory_bytes,
        ) = prepared
        self._raw_steps.append(raw_bytes)
        self._owner_events.append(owner_bytes)
        self._particle_counts.append(count_bytes)
        self._raw_loads.extend(load_bytes)
        self._trajectories.append(trajectory_bytes)

    def freeze(
        self,
        *,
        execution_mode: str,
        status: str,
        terminal_coordinate: Mapping[str, object] | None = None,
        stop_code: str | None = None,
        stop_message: str | None = None,
    ) -> ArtifactRecords:
        return ArtifactRecords(
            execution_mode=execution_mode,
            status=status,
            raw_steps=self._snapshot(self._raw_steps),
            source_events=self._snapshot(self._source_events),
            owner_events=self._snapshot(self._owner_events),
            particle_counts=self._snapshot(self._particle_counts),
            raw_loads=self._snapshot(self._raw_loads),
            transport_stages=self._snapshot(self._transport_stages),
            trajectories=self._snapshot(self._trajectories),
            terminal_coordinate=(
                None if terminal_coordinate is None else dict(terminal_coordinate)
            ),
            stop_code=stop_code,
            stop_message=stop_message,
        )


@dataclass(frozen=True, slots=True)
class FormalExecutorAPI:
    """Exact class-identity-safe ABI injected into the audited executor factory."""

    schema_id: str
    formal_levels: tuple[int, int, int]
    artifact_sink_type: type[ArtifactSink]
    stop_constructor: type[FormalRunStopped]
    source_event_fields: tuple[str, ...]
    stage_fields: tuple[str, ...]
    frontier_node_ids: tuple[str, ...]
    fixed_probes_gp1_m: tuple[tuple[float, float, float], ...]
    dependency_leaf_file_path: Mapping[str, str]
    dependency_leaf_file_sha256: Mapping[str, str]
    dependency_runtime_module_file_path: Mapping[str, str]
    dependency_runtime_module_file_sha256: Mapping[str, str]


FORMAL_EXECUTOR_API: Final = FormalExecutorAPI(
    schema_id="fluxv-v5h13-baik-w2-formal-executor-api-v1",
    formal_levels=FORMAL_LEVELS,
    artifact_sink_type=ArtifactSink,
    stop_constructor=FormalRunStopped,
    source_event_fields=SOURCE_EVENT_FIELDS,
    stage_fields=STAGE_FIELDS,
    frontier_node_ids=FRONTIER_NODE_IDS,
    fixed_probes_gp1_m=FIXED_PROBES_GP1_M,
    dependency_leaf_file_path=MappingProxyType({}),
    dependency_leaf_file_sha256=MappingProxyType({}),
    dependency_runtime_module_file_path=MappingProxyType({}),
    dependency_runtime_module_file_sha256=MappingProxyType({}),
)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_json_primitive_tree(value: object, path: str = "record") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string mapping key")
            _validate_json_primitive_tree(item, f"{path}.{key}")
        return
    if type(value) in (list, tuple):
        for index, item in enumerate(value):
            _validate_json_primitive_tree(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains a non-JSON primitive: {type(value).__name__}")


def _canonical_json_record_bytes(value: Mapping[str, object]) -> bytes:
    _validate_json_primitive_tree(value)
    payload = _json_bytes(value)
    decoded = json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(decoded, dict):
        raise TypeError("canonical artifact record must be a JSON object")
    return payload


def _decode_canonical_json_record(payload: bytes) -> dict[str, object]:
    decoded = json.loads(
        payload,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(decoded, dict):
        raise TypeError("canonical artifact record must be a JSON object")
    return decoded


def _loads_json_bytes(payload: bytes) -> object:
    return json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def _load_json(path: Path) -> object:
    return _loads_json_bytes(path.read_bytes())


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError("JSONL must end with a newline")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError(f"JSONL contains a blank record at line {line_number}")
        value = json.loads(
            line,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record {line_number} is not an object")
        rows.append(value)
    return rows


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _installed_distribution_identity(
    distribution_name: str,
) -> tuple[Path, str, str]:
    """Resolve installed metadata without importing the distribution package."""

    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError as error:
        raise DependencyFreezeError(
            f"required distribution is unavailable: {distribution_name}"
        ) from error
    metadata_files = [
        Path(distribution.locate_file(item)).resolve(strict=True)
        for item in (distribution.files or ())
        if str(item).replace("\\", "/").endswith(".dist-info/METADATA")
    ]
    installed_name = distribution.metadata.get("Name")
    installed_version = distribution.version
    if (
        len(metadata_files) != 1
        or type(installed_name) is not str
        or not installed_name
        or type(installed_version) is not str
        or not installed_version
    ):
        raise DependencyFreezeError(
            f"installed distribution identity is ambiguous: {distribution_name}"
        )
    return metadata_files[0], installed_name, installed_version


def _installed_distribution_file(distribution_name: str, relative_suffix: str) -> Path:
    distribution = importlib_metadata.distribution(distribution_name)
    matches = [
        Path(distribution.locate_file(item)).resolve(strict=True)
        for item in (distribution.files or ())
        if str(item).replace("\\", "/").endswith(relative_suffix)
    ]
    if len(matches) != 1:
        raise DependencyFreezeError(
            f"installed distribution source identity is ambiguous: {relative_suffix}"
        )
    return matches[0]


def _expected_local_module_paths(root: Path, tail: Sequence[str]) -> set[Path]:
    if not tail:
        candidates = (root / "__init__.py",)
    else:
        stem = root.joinpath(*tail)
        candidates = (stem.with_suffix(".py"), stem / "__init__.py")
    return {
        candidate.resolve(strict=True) for candidate in candidates if candidate.exists()
    }


def _validate_runtime_module_path_identity(
    module_name: str, path: Path, repository_root: Path
) -> None:
    if module_name in {"ldvm_fourier", "flap_ldvm"}:
        expected = (repository_root / "platform" / f"{module_name}.py").resolve(
            strict=True
        )
        if path != expected:
            raise DependencyFreezeError(
                f"runtime module has a non-canonical project path: {module_name}"
            )
        return
    if module_name == "fluxvortex" or module_name.startswith("fluxvortex."):
        tail = module_name.split(".")[1:]
        candidates = _expected_local_module_paths(
            repository_root / "src" / "fluxvortex", tail
        )
    elif module_name == "forward_flight_benchmarks" or module_name.startswith(
        "forward_flight_benchmarks."
    ):
        tail = module_name.split(".")[1:]
        candidates = _expected_local_module_paths(
            repository_root / "platform" / "forward_flight_benchmarks", tail
        )
    elif module_name == "pterasoftware" or module_name.startswith("pterasoftware."):
        tail = module_name.split(".")[1:]
        suffixes = (
            ("pterasoftware/__init__.py",)
            if not tail
            else (
                "pterasoftware/" + "/".join(tail) + ".py",
                "pterasoftware/" + "/".join(tail) + "/__init__.py",
            )
        )
        candidates = set()
        for suffix in suffixes:
            try:
                candidates.add(_installed_distribution_file("pterasoftware", suffix))
            except DependencyFreezeError:
                continue
    else:
        raise DependencyFreezeError(
            f"runtime module namespace is not audited: {module_name}"
        )
    if path not in candidates:
        raise DependencyFreezeError(
            f"runtime module has a non-canonical origin: {module_name}"
        )


def _verified_dependency_audit(audit_token_path: Path) -> dict[str, object]:
    """Strictly load and runtime-rehash the external acyclic audit DAG."""

    token_path = Path(audit_token_path).resolve(strict=True)
    token_payload = token_path.read_bytes()
    token_sha256 = sha256(token_payload).hexdigest()
    token = _loads_json_bytes(token_payload)
    token_fields = {
        "schema_id",
        "audit_id",
        "audit_scope",
        "audit_verdict",
        "dependency_manifest_path",
        "dependency_manifest_sha256",
        "observation_access",
    }
    if not isinstance(token, Mapping) or set(token) != token_fields:
        raise DependencyFreezeError("dependency audit token schema is invalid")
    if token["schema_id"] != DEPENDENCY_AUDIT_TOKEN_SCHEMA_ID:
        raise DependencyFreezeError("dependency audit token schema_id is invalid")
    if type(token["audit_id"]) is not str or not token["audit_id"]:
        raise DependencyFreezeError("dependency audit_id is invalid")
    if (
        token["audit_scope"] != FORMAL_AUDIT_SCOPE
        or token["audit_verdict"] != "PASS"
        or token["observation_access"] != OBSERVATION_ACCESS
    ):
        raise DependencyFreezeError("dependency audit verdict/scope is invalid")
    if not _is_sha256(token["dependency_manifest_sha256"]):
        raise DependencyFreezeError("dependency manifest digest is invalid")
    manifest_text = token["dependency_manifest_path"]
    if type(manifest_text) is not str:
        raise DependencyFreezeError("dependency manifest path is invalid")
    manifest_path = Path(manifest_text)
    if not manifest_path.is_absolute() or str(manifest_path.resolve()) != manifest_text:
        raise DependencyFreezeError(
            "dependency manifest path must be canonical absolute"
        )
    manifest_path = manifest_path.resolve(strict=True)
    if manifest_path == token_path:
        raise DependencyFreezeError("audit token cannot point to itself")
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = sha256(manifest_payload).hexdigest()
    if manifest_sha256 != token["dependency_manifest_sha256"]:
        raise DependencyFreezeError("dependency manifest runtime hash mismatch")

    manifest = _loads_json_bytes(manifest_payload)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_id",
        "dependency_set_scope",
        "leaf_files",
        "runtime_module_files",
    }:
        raise DependencyFreezeError("dependency manifest schema is invalid")
    if manifest["schema_id"] != DEPENDENCY_MANIFEST_SCHEMA_ID:
        raise DependencyFreezeError("dependency manifest schema_id is invalid")
    if manifest["dependency_set_scope"] != DEPENDENCY_SET_SCOPE:
        raise DependencyFreezeError("dependency manifest bounded-set scope is invalid")
    leaves = manifest["leaf_files"]
    if type(leaves) is not list:
        raise DependencyFreezeError("dependency manifest leaves must be a list")
    names: list[str] = []
    paths: set[Path] = set()
    runtime_hashes: dict[str, str] = {}
    runtime_paths: dict[str, str] = {}
    for leaf in leaves:
        if not isinstance(leaf, Mapping) or set(leaf) != {"name", "path", "sha256"}:
            raise DependencyFreezeError("dependency leaf schema is invalid")
        name, path_text, expected_sha256 = leaf["name"], leaf["path"], leaf["sha256"]
        if (
            type(name) is not str
            or type(path_text) is not str
            or not _is_sha256(expected_sha256)
        ):
            raise DependencyFreezeError("dependency leaf values are invalid")
        path = Path(path_text)
        if not path.is_absolute() or str(path.resolve()) != path_text:
            raise DependencyFreezeError(
                "dependency leaf path must be canonical absolute"
            )
        path = path.resolve(strict=True)
        if path in (token_path, manifest_path):
            raise DependencyFreezeError(
                "dependency DAG contains a token/manifest back-edge"
            )
        if path in paths:
            raise DependencyFreezeError("dependency manifest contains a duplicate path")
        paths.add(path)
        names.append(name)
        payload = path.read_bytes()
        observed_sha256 = sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise DependencyFreezeError(f"dependency runtime hash mismatch: {name}")
        if (
            manifest_sha256.encode("ascii") in payload
            or token_sha256.encode("ascii") in payload
        ):
            raise DependencyFreezeError(
                f"dependency leaf contains a manifest/token hash back-edge: {name}"
            )
        runtime_hashes[name] = observed_sha256
        runtime_paths[name] = str(path)
    if tuple(names) != REQUIRED_DEPENDENCY_LEAVES:
        raise DependencyFreezeError("dependency manifest leaf names/order are invalid")
    repository_root = Path(__file__).resolve().parents[2]
    for leaf_name, configured_path in CANONICAL_PROJECT_LEAF_RELATIVE_PATH.items():
        configured = Path(configured_path)
        expected_path = (
            configured if configured.is_absolute() else repository_root / configured
        ).resolve(strict=True)
        if Path(runtime_paths[leaf_name]) != expected_path:
            raise DependencyFreezeError(
                f"dependency project leaf has a non-canonical identity: {leaf_name}"
            )
    for leaf_name, suffix in EXTERNAL_DEPENDENCY_PATH_SUFFIX.items():
        if not Path(runtime_paths[leaf_name]).as_posix().endswith(suffix):
            raise DependencyFreezeError(
                f"dependency external leaf has an invalid path identity: {leaf_name}"
            )
    distribution_name_version: dict[str, dict[str, str]] = {}
    for leaf_name, distribution_name in DISTRIBUTION_LEAF_TO_NAME.items():
        (
            expected_metadata,
            installed_name,
            installed_version,
        ) = _installed_distribution_identity(distribution_name)
        if Path(runtime_paths[leaf_name]) != expected_metadata:
            raise DependencyFreezeError(
                f"dependency distribution metadata identity is invalid: {leaf_name}"
            )
        distribution_name_version[leaf_name] = {
            "name": installed_name,
            "version": installed_version,
        }
    expected_ptera_solver = _installed_distribution_file(
        "pterasoftware",
        "pterasoftware/unsteady_ring_vortex_lattice_method.py",
    )
    if Path(runtime_paths["pterasoftware_solver_source"]) != expected_ptera_solver:
        raise DependencyFreezeError(
            "dependency Ptera solver source is not the installed distribution source"
        )
    expected_runner = Path(__file__).resolve()
    expected_probe_sources = {
        "probe_source_v5h7": expected_runner.with_name(
            "run_fluxv_v5h7_temporal_topology_oracle.py"
        ),
        "probe_source_v5h8": expected_runner.with_name(
            "run_fluxv_v5h8_incremental_sheet_oracle.py"
        ),
    }
    for leaf_name, expected_path in expected_probe_sources.items():
        source_name = expected_path.name
        if (
            Path(runtime_paths[leaf_name]) != expected_path.resolve(strict=True)
            or runtime_hashes[leaf_name] != FIXED_PROBE_SOURCE_SHA256[source_name]
        ):
            raise DependencyFreezeError(
                f"fixed-probe source leaf/hash is invalid: {leaf_name}"
            )
    runtime_modules = manifest["runtime_module_files"]
    if type(runtime_modules) is not list:
        raise DependencyFreezeError("runtime module inventory must be a list")
    runtime_module_paths: dict[str, str] = {}
    runtime_module_hashes: dict[str, str] = {}
    seen_runtime_paths: set[Path] = set()
    for item in runtime_modules:
        if not isinstance(item, Mapping) or set(item) != {
            "module_name",
            "path",
            "sha256",
        }:
            raise DependencyFreezeError("runtime module inventory schema is invalid")
        module_name = item["module_name"]
        path_text = item["path"]
        expected_sha256 = item["sha256"]
        if (
            type(module_name) is not str
            or not any(
                module_name == prefix or module_name.startswith(prefix + ".")
                for prefix in AUDITED_RUNTIME_MODULE_PREFIXES
            )
            or type(path_text) is not str
            or not _is_sha256(expected_sha256)
        ):
            raise DependencyFreezeError("runtime module inventory values are invalid")
        path = Path(path_text)
        if not path.is_absolute() or str(path.resolve()) != path_text:
            raise DependencyFreezeError(
                "runtime module path must be canonical absolute"
            )
        path = path.resolve(strict=True)
        _validate_runtime_module_path_identity(module_name, path, repository_root)
        if (
            module_name in runtime_module_paths
            or path in seen_runtime_paths
            or path in (token_path, manifest_path)
        ):
            raise DependencyFreezeError("runtime module inventory is cyclic/duplicated")
        payload = path.read_bytes()
        observed_sha256 = sha256(payload).hexdigest()
        if observed_sha256 != expected_sha256:
            raise DependencyFreezeError(f"runtime module hash mismatch: {module_name}")
        if (
            manifest_sha256.encode("ascii") in payload
            or token_sha256.encode("ascii") in payload
        ):
            raise DependencyFreezeError(
                f"runtime module contains a manifest/token hash back-edge: {module_name}"
            )
        runtime_module_paths[module_name] = str(path)
        runtime_module_hashes[module_name] = observed_sha256
        seen_runtime_paths.add(path)
    if tuple(runtime_module_paths) != tuple(sorted(runtime_module_paths)):
        raise DependencyFreezeError("runtime module inventory order is not canonical")
    return {
        "audit_id": token["audit_id"],
        "audit_scope": token["audit_scope"],
        "audit_token_path": str(token_path),
        "audit_token_sha256": token_sha256,
        "audit_verdict": token["audit_verdict"],
        "dependency_manifest_path": str(manifest_path),
        "dependency_manifest_sha256": manifest_sha256,
        "leaf_file_path": runtime_paths,
        "leaf_file_sha256": runtime_hashes,
        "runtime_module_file_path": runtime_module_paths,
        "runtime_module_file_sha256": runtime_module_hashes,
        "observed_runtime_module_file_path": {},
        "observed_runtime_module_file_sha256": {},
        "dependency_set_scope": manifest["dependency_set_scope"],
        "distribution_name_version": distribution_name_version,
        "observation_access": token["observation_access"],
        "status": "verified",
    }


def _runtime_reverify_dependency_audit(
    value: Mapping[str, object]
) -> dict[str, object]:
    expected_fields = {
        "audit_id",
        "audit_scope",
        "audit_token_path",
        "audit_token_sha256",
        "audit_verdict",
        "dependency_manifest_path",
        "dependency_manifest_sha256",
        "leaf_file_path",
        "leaf_file_sha256",
        "runtime_module_file_path",
        "runtime_module_file_sha256",
        "observed_runtime_module_file_path",
        "observed_runtime_module_file_sha256",
        "dependency_set_scope",
        "distribution_name_version",
        "observation_access",
        "status",
    }
    if set(value) != expected_fields or value.get("status") != "verified":
        raise DependencyFreezeError("verified dependency evidence schema is invalid")
    token_path = value.get("audit_token_path")
    if type(token_path) is not str:
        raise DependencyFreezeError("verified audit token path is invalid")
    try:
        observed = _verified_dependency_audit(Path(token_path))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError) as error:
        raise DependencyFreezeError(
            f"dependency postflight rehash failed: {type(error).__name__}: {error}"
        ) from error
    observed_path = value.get("observed_runtime_module_file_path")
    observed_hash = value.get("observed_runtime_module_file_sha256")
    if not isinstance(observed_path, Mapping) or not isinstance(observed_hash, Mapping):
        raise DependencyFreezeError("observed runtime module evidence is invalid")
    expected_static = dict(value)
    expected_static["observed_runtime_module_file_path"] = {}
    expected_static["observed_runtime_module_file_sha256"] = {}
    if observed != expected_static:
        raise DependencyFreezeError("dependency evidence differs from runtime rehash")
    declared_path = observed["runtime_module_file_path"]
    declared_hash = observed["runtime_module_file_sha256"]
    assert isinstance(declared_path, Mapping) and isinstance(declared_hash, Mapping)
    if tuple(observed_path) != tuple(sorted(observed_path)) or set(
        observed_path
    ) != set(observed_hash):
        raise DependencyFreezeError(
            "observed runtime module inventory is non-canonical"
        )
    for module_name, path_text in observed_path.items():
        if (
            declared_path.get(module_name) != path_text
            or declared_hash.get(module_name) != observed_hash[module_name]
            or _sha256_file(Path(str(path_text)).resolve(strict=True))
            != observed_hash[module_name]
        ):
            raise DependencyFreezeError(
                f"observed runtime module differs from manifest: {module_name}"
            )
    observed["observed_runtime_module_file_path"] = dict(observed_path)
    observed["observed_runtime_module_file_sha256"] = dict(observed_hash)
    return observed


def _capture_observed_runtime_modules(
    dependency_audit: Mapping[str, object],
) -> dict[str, object]:
    """Bind loaded project/Ptera module origins without importing any module."""

    reverified = _runtime_reverify_dependency_audit(dependency_audit)
    declared_path = reverified["runtime_module_file_path"]
    declared_hash = reverified["runtime_module_file_sha256"]
    assert isinstance(declared_path, Mapping) and isinstance(declared_hash, Mapping)
    observed_paths: dict[str, str] = {}
    observed_hashes: dict[str, str] = {}
    for module_name in sorted(sys.modules):
        if not any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in AUDITED_RUNTIME_MODULE_PREFIXES
        ):
            continue
        module = sys.modules[module_name]
        file_text = getattr(module, "__file__", None)
        if type(file_text) is not str:
            continue
        path = Path(file_text)
        if path.suffix in {".pyc", ".pyo"}:
            source_path = Path(importlib.util.source_from_cache(str(path)))
            if source_path.exists():
                path = source_path
        path = path.resolve(strict=True)
        digest = _sha256_file(path)
        if (
            declared_path.get(module_name) != str(path)
            or declared_hash.get(module_name) != digest
        ):
            raise DependencyFreezeError(
                f"loaded runtime module is unmanifested or drifted: {module_name}"
            )
        observed_paths[module_name] = str(path)
        observed_hashes[module_name] = digest
    reverified["observed_runtime_module_file_path"] = observed_paths
    reverified["observed_runtime_module_file_sha256"] = observed_hashes
    return reverified


def _load_formal_executor(
    dependency_audit: Mapping[str, object],
) -> FormalCouplingExecutor:
    """Spec-load the audited formal executor only after dependency preflight."""

    reverified = _runtime_reverify_dependency_audit(dependency_audit)
    leaf_paths = reverified.get("leaf_file_path")
    leaf_hashes = reverified.get("leaf_file_sha256")
    runtime_paths = reverified.get("runtime_module_file_path")
    runtime_hashes = reverified.get("runtime_module_file_sha256")
    if (
        not isinstance(leaf_paths, Mapping)
        or not isinstance(leaf_hashes, Mapping)
        or not isinstance(runtime_paths, Mapping)
        or not isinstance(runtime_hashes, Mapping)
    ):
        raise DependencyFreezeError("formal executor leaf map is invalid")
    module_text = leaf_paths.get("formal_executor_module")
    if type(module_text) is not str:
        raise DependencyFreezeError("formal executor module leaf is invalid")
    module_path = Path(module_text).resolve(strict=True)
    if module_path.suffix != ".py":
        raise DependencyFreezeError(
            "formal executor module must be a Python source file"
        )
    source_bytes = module_path.read_bytes()
    if sha256(source_bytes).hexdigest() != leaf_hashes.get("formal_executor_module"):
        raise DependencyFreezeError("formal executor changed before module load")
    source = source_bytes.decode("utf-8")
    tree = ast.parse(source, filename=str(module_path))
    runner_module_stem = Path(__file__).stem
    for node in ast.walk(tree):
        imported: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            imported = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "" if node.module is None else node.module
            imported = tuple(
                name
                for alias in node.names
                for name in (
                    base,
                    alias.name,
                    f"{base}.{alias.name}" if base else alias.name,
                )
                if name
            )
        if any(name.split(".")[-1] == runner_module_stem for name in imported):
            raise DependencyFreezeError(
                "formal executor must use the injected API and cannot import the runner"
            )
    module_name = (
        "fluxv_v5h13_w2_audited_executor_" + str(reverified["audit_token_sha256"])[:16]
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise DependencyFreezeError("formal executor module spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    per_run_api = replace(
        FORMAL_EXECUTOR_API,
        dependency_leaf_file_path=MappingProxyType(
            {str(name): str(path) for name, path in leaf_paths.items()}
        ),
        dependency_leaf_file_sha256=MappingProxyType(
            {str(name): str(digest) for name, digest in leaf_hashes.items()}
        ),
        dependency_runtime_module_file_path=MappingProxyType(
            {str(name): str(path) for name, path in runtime_paths.items()}
        ),
        dependency_runtime_module_file_sha256=MappingProxyType(
            {str(name): str(digest) for name, digest in runtime_hashes.items()}
        ),
    )
    try:
        exec(compile(tree, str(module_path), "exec"), module.__dict__)
        factory = getattr(module, FORMAL_EXECUTOR_FACTORY_NAME)
        executor = factory(per_run_api)
        attest_origins = getattr(executor, "attest_dependency_origins")
        if not callable(attest_origins):
            raise TypeError("formal executor lacks dependency-origin attestation")
        attest_origins()
        _capture_observed_runtime_modules(reverified)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not callable(getattr(executor, "run_formal_matrix", None)):
        raise TypeError("formal executor factory returned an invalid executor")
    return executor


def _dependency_evidence(
    execution_mode: str, value: Mapping[str, object] | None
) -> dict[str, object]:
    if execution_mode == SYNTHETIC_EXECUTION_MODE:
        if value is not None:
            raise ValueError(
                "synthetic fixture cannot carry formal dependency evidence"
            )
        return {
            "audit_id": None,
            "audit_scope": None,
            "audit_token_path": None,
            "audit_token_sha256": None,
            "audit_verdict": None,
            "dependency_manifest_path": None,
            "dependency_manifest_sha256": None,
            "leaf_file_path": {},
            "leaf_file_sha256": {},
            "runtime_module_file_path": {},
            "runtime_module_file_sha256": {},
            "observed_runtime_module_file_path": {},
            "observed_runtime_module_file_sha256": {},
            "dependency_set_scope": None,
            "distribution_name_version": {},
            "observation_access": OBSERVATION_ACCESS,
            "status": "not_applicable",
        }
    if execution_mode != FORMAL_EXECUTION_MODE:
        raise ValueError("execution_mode is invalid")
    if value is None:
        return {
            "audit_id": None,
            "audit_scope": None,
            "audit_token_path": None,
            "audit_token_sha256": None,
            "audit_verdict": None,
            "dependency_manifest_path": None,
            "dependency_manifest_sha256": None,
            "leaf_file_path": {},
            "leaf_file_sha256": {},
            "runtime_module_file_path": {},
            "runtime_module_file_sha256": {},
            "observed_runtime_module_file_path": {},
            "observed_runtime_module_file_sha256": {},
            "dependency_set_scope": None,
            "distribution_name_version": {},
            "observation_access": OBSERVATION_ACCESS,
            "status": "unbound",
        }
    return _runtime_reverify_dependency_audit(value)


def _payload_sha256(domain: str, value: object) -> str:
    return sha256(domain.encode("ascii") + b"\0" + _json_bytes(value)).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_json_bytes(list(contiguous.shape)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _v5h10_hash_value(digest: Any, value: object) -> None:
    """Replay the frozen v5h10 row-owner typed hashing grammar."""

    if value is None:
        digest.update(b"N")
    elif type(value) is bool:
        digest.update(b"B1" if value else b"B0")
    elif type(value) is int:
        digest.update(b"I" + str(value).encode("ascii") + b";")
    elif type(value) is float:
        digest.update(b"F" + value.hex().encode("ascii") + b";")
    elif type(value) is str:
        encoded = value.encode("utf-8")
        digest.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif type(value) is tuple:
        digest.update(b"T" + len(value).to_bytes(8, "big"))
        for item in value:
            _v5h10_hash_value(digest, item)
    elif type(value) is np.ndarray:
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _v5h10_hash_value(digest, array.dtype.str)
        _v5h10_hash_value(digest, tuple(array.shape))
        _v5h10_hash_value(digest, tuple(array.strides))
        digest.update(memoryview(array).cast("B"))
    else:
        raise TypeError(f"unsupported v5h10 digest value: {type(value).__name__}")


def _v5h10_digest(domain: str, *values: object) -> str:
    digest = sha256(domain.encode("ascii"))
    for value in values:
        _v5h10_hash_value(digest, value)
    return digest.hexdigest()


def _v5h10_event_digest(
    domain: str,
    value: Mapping[str, object],
    fields: Sequence[str],
    digest_field: str,
    *,
    tuple_fields: Sequence[str] = (),
) -> str:
    payload = tuple(
        (
            field,
            tuple(value[field]) if field in tuple_fields else value[field],
        )
        for field in fields
        if field != digest_field
    )
    return _v5h10_digest(domain, payload)


def encode_array(value: object) -> dict[str, object]:
    """Encode one finite numeric array with exact dtype/shape/raw-byte binding."""

    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise ValueError("artifact arrays must use a real numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("artifact arrays must be finite")
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "dtype": contiguous.dtype.str,
        "order": "C",
        "sha256": _array_sha256(contiguous),
        "shape": list(contiguous.shape),
    }


def decode_array(value: object) -> np.ndarray:
    """Decode and independently revalidate an :func:`encode_array` payload."""

    if not isinstance(value, Mapping) or set(value) != {
        "data_base64",
        "dtype",
        "order",
        "sha256",
        "shape",
    }:
        raise ValueError("encoded array schema is invalid")
    if value["order"] != "C" or type(value["dtype"]) is not str:
        raise ValueError("encoded array dtype/order is invalid")
    shape = value["shape"]
    if type(shape) is not list or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("encoded array shape is invalid")
    try:
        dtype = np.dtype(value["dtype"])
    except TypeError as error:
        raise ValueError("encoded array dtype is invalid") from error
    if dtype.kind not in "iuf" or dtype.kind == "b":
        raise ValueError("encoded array dtype must be real numeric")
    try:
        raw = base64.b64decode(value["data_base64"], validate=True)
    except (binascii.Error, TypeError) as error:
        raise ValueError("encoded array base64 is invalid") from error
    expected_count = math.prod(shape)
    if len(raw) != expected_count * dtype.itemsize:
        raise ValueError("encoded array byte length is invalid")
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("decoded artifact array is non-finite")
    if not _is_sha256(value["sha256"]) or _array_sha256(array) != value["sha256"]:
        raise ValueError("encoded array digest mismatch")
    return array


def _trajectory_record_shell(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "transport_substeps",
        "layer",
        "status",
        "particle_ids",
        "material_tracer_ids",
        "frontier_node_ids",
        "arrays",
        "metadata",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("trajectory callback record schema is invalid")
    arrays = value["arrays"]
    if not isinstance(arrays, Mapping):
        raise TypeError("trajectory callback arrays must be a mapping")
    if any(type(name) is not str for name in arrays):
        raise TypeError("trajectory array channel names must be strings")
    if not set(arrays).issubset(REQUIRED_TRAJECTORY_ARRAYS):
        raise ValueError("trajectory callback contains a foreign array channel")
    return {
        "transport_substeps": value["transport_substeps"],
        "layer": value["layer"],
        "status": value["status"],
        "particle_ids": value["particle_ids"],
        "material_tracer_ids": value["material_tracer_ids"],
        "frontier_node_ids": value["frontier_node_ids"],
        "arrays": arrays,
        "metadata": value["metadata"],
    }


def _validate_preencoded_trajectory_record(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = _trajectory_record_shell(value)
    arrays = result["arrays"]
    assert isinstance(arrays, Mapping)
    completed = result["status"] == "completed"
    if completed and set(arrays) != set(REQUIRED_TRAJECTORY_ARRAYS):
        raise ValueError("completed callback trajectory has an incomplete array set")
    encoded: dict[str, object] = {}
    for name, payload in arrays.items():
        decoded = decode_array(payload)
        if completed and decoded.dtype != np.dtype(np.float64):
            raise ValueError(
                f"completed trajectory channel {name} must have exact float64 dtype"
            )
        assert isinstance(payload, Mapping)
        encoded[name] = dict(payload)
    result["arrays"] = encoded
    return result


def _encode_trajectory_array_record(
    value: Mapping[str, object],
) -> dict[str, object]:
    result = _trajectory_record_shell(value)
    arrays = result["arrays"]
    assert isinstance(arrays, Mapping)
    encoded: dict[str, object] = {}
    for name, array in arrays.items():
        if type(array) is not np.ndarray:
            raise TypeError(
                "trajectory array-record channels must be exact numpy ndarrays"
            )
        encoded[name] = encode_array(array)
    result["arrays"] = encoded
    return result


def _id_sequence(value: object) -> dict[str, object]:
    if type(value) not in (tuple, list) or any(type(item) is not str for item in value):
        raise ValueError("particle_ids must be a string tuple/list")
    items = list(value)
    digest = _payload_sha256("fluxv-v5h13-particle-id-sequence-v1", items)
    return {"items": items, "sha256": digest}


def _decode_id_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, Mapping) or set(value) != {"items", "sha256"}:
        raise ValueError("particle ID sequence schema is invalid")
    items = value["items"]
    if type(items) is not list or any(type(item) is not str for item in items):
        raise ValueError("particle ID sequence contents are invalid")
    expected = _payload_sha256("fluxv-v5h13-particle-id-sequence-v1", items)
    if value["sha256"] != expected:
        raise ValueError("particle ID sequence digest mismatch")
    return tuple(items)


def _format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("CSV payload contains a non-finite float")
        return format(value, ".17g")
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, (tuple, list, dict)):
        return _json_bytes(value).decode("utf-8")
    return value


def _csv_bytes(
    fieldnames: tuple[str, ...], rows: Sequence[Mapping[str, object]]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    expected = set(fieldnames)
    for row in rows:
        if set(row) != expected:
            missing = sorted(expected - set(row))
            foreign = sorted(set(row) - expected)
            raise ValueError(
                f"CSV row schema mismatch: missing={missing}, foreign={foreign}"
            )
        writer.writerow({name: _format_csv_value(row[name]) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _exact_int(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compact_stage_stability_fields(
    compact_evidence: Mapping[str, object],
) -> dict[str, float]:
    """Extract only gates that the coupling compact evidence actually persists."""

    if not isinstance(compact_evidence, Mapping):
        raise TypeError("compact stage evidence must be a mapping")
    result = {
        "invariant_residual_over_slog_max": _finite_float(
            compact_evidence.get("invariant_residual_over_slog_max"),
            "compact normalized invariant residual",
        ),
        "h_jacobian_frobenius": _finite_float(
            compact_evidence.get("h_jacobian_frobenius"),
            "compact h Jacobian Frobenius",
        ),
        "h_convective_over_sigma": _finite_float(
            compact_evidence.get("h_convective_over_sigma"),
            "compact h convective over sigma",
        ),
    }
    if any(value < 0.0 for value in result.values()):
        raise ValueError("compact stage stability evidence must be non-negative")
    return result


def _layer_key(row: Mapping[str, object]) -> tuple[int, int]:
    return (
        _exact_int(row["transport_substeps"], "transport_substeps", 1),
        _exact_int(row["layer"], "layer", 1),
    )


def _source_key(row: Mapping[str, object]) -> tuple[int, int]:
    return (
        _exact_int(row["transport_substeps"], "transport_substeps", 1),
        _exact_int(row["source_step_index"], "source_step_index", 1),
    )


def _load_key(row: Mapping[str, object]) -> tuple[int, int, str, str]:
    return (*_layer_key(row), str(row["scope"]), str(row["panel_id"]))


def _stage_key(row: Mapping[str, object]) -> tuple[int, int, int, int]:
    return (
        *_layer_key(row),
        _exact_int(row["substep"], "substep", 1),
        _exact_int(row["stage"], "stage", 1),
    )


def _assert_prefix(
    name: str,
    observed: Sequence[object],
    expected: Sequence[object],
) -> None:
    if len(observed) > len(expected) or tuple(observed) != tuple(
        expected[: len(observed)]
    ):
        raise ValueError(f"{name} keys are not an exact completed prefix")


def _stage_count_through_layers(layer_count: int) -> int:
    return sum(3 * level for level, _ in EXPECTED_LAYER_KEYS[:layer_count])


def _validate_cross_table_progress(
    layer_keys: Sequence[tuple[int, int]],
    source_keys: Sequence[tuple[int, int]],
    stage_rows: Sequence[Mapping[str, object]],
) -> None:
    """Require source/stage work to cover at most one uncommitted layer."""

    layer_count = len(layer_keys)
    source_count = len(source_keys)
    if source_count not in (layer_count, layer_count + 1):
        raise ValueError(
            "source prefix may contain only the single current uncommitted layer"
        )
    failed = bool(stage_rows and stage_rows[-1].get("status") == "failed")
    completed_count = len(stage_rows) - int(failed)
    committed_boundary = _stage_count_through_layers(layer_count)
    if completed_count < committed_boundary:
        raise ValueError("stage prefix does not cover every completed layer")
    if source_count == layer_count:
        if completed_count != committed_boundary or failed:
            raise ValueError("stage work advanced without its source-event parent")
        return
    if layer_count >= len(EXPECTED_LAYER_KEYS):
        raise ValueError("source prefix advanced beyond the formal layer matrix")
    current_level = EXPECTED_LAYER_KEYS[layer_count][0]
    current_boundary = committed_boundary + 3 * current_level
    if completed_count > current_boundary or (
        completed_count == current_boundary and failed
    ):
        raise ValueError("stage prefix advanced beyond the single current source layer")


def _expected_unbegun_terminal_six(
    *,
    completed_stage_count: int,
    layer_keys: Sequence[tuple[int, int]],
    source_keys: Sequence[tuple[int, int]],
) -> tuple[object, object, object, object, object, object]:
    if not layer_keys and not source_keys and completed_stage_count == 0:
        return (None, None, None, None, None, None)
    layer_count = len(layer_keys)
    source_count = len(source_keys)
    committed_boundary = _stage_count_through_layers(layer_count)
    if source_count == layer_count + 1:
        if completed_stage_count == committed_boundary:
            level, source_step = source_keys[-1]
            return (level, None, source_step, source_step - 1, None, None)
        current_level = EXPECTED_LAYER_KEYS[layer_count][0]
        current_boundary = committed_boundary + 3 * current_level
        stage_key = EXPECTED_STAGE_KEYS[
            completed_stage_count
            if completed_stage_count < current_boundary
            else current_boundary - 1
        ]
        return (
            stage_key[0],
            stage_key[1],
            SOURCE_STEPS[stage_key[1] - 1],
            PTERA_STEPS[stage_key[1] - 1],
            stage_key[2],
            stage_key[3],
        )
    if layer_count < len(EXPECTED_LAYER_KEYS):
        level, source_step = EXPECTED_SOURCE_KEYS[layer_count]
        return (level, None, source_step, source_step - 1, None, None)
    final_stage = EXPECTED_STAGE_KEYS[-1]
    return (
        final_stage[0],
        final_stage[1],
        SOURCE_STEPS[final_stage[1] - 1],
        PTERA_STEPS[final_stage[1] - 1],
        final_stage[2],
        final_stage[3],
    )


def _expected_conversion_terminal_six(
    *,
    completed_stage_count: int,
    layer_keys: Sequence[tuple[int, int]],
    source_keys: Sequence[tuple[int, int]],
) -> tuple[object, object, object, object, object, object] | None:
    """Exact unbegun-next-stage coordinate after a durable source parent."""

    layer_count = len(layer_keys)
    committed_boundary = _stage_count_through_layers(layer_count)
    if (
        len(source_keys) != layer_count + 1
        or layer_count >= len(EXPECTED_LAYER_KEYS)
        or not committed_boundary <= completed_stage_count < len(EXPECTED_STAGE_KEYS)
    ):
        return None
    key = EXPECTED_STAGE_KEYS[completed_stage_count]
    return (
        key[0],
        key[1],
        SOURCE_STEPS[key[1] - 1],
        PTERA_STEPS[key[1] - 1],
        key[2],
        key[3],
    )


def _validate_terminal_coordinate(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(TERMINAL_COORDINATE_FIELDS):
        raise ValueError("STOP terminal coordinate schema is invalid")
    result = dict(value)
    for name in TERMINAL_COORDINATE_FIELDS[:6]:
        item = result[name]
        if item is not None:
            _exact_int(item, f"terminal_coordinate.{name}", 1)
    if type(result["phase"]) is not str or not result["phase"]:
        raise ValueError("STOP terminal phase must be a non-empty string")
    if type(result["stage_began"]) is not bool:
        raise ValueError("STOP stage_began must be an exact bool")
    return result


def _validate_pass_counters(row: Mapping[str, object]) -> None:
    level = _exact_int(row["transport_substeps"], "transport_substeps", 1)
    stage_count = 3 * level
    expected = {
        "direct_field_call_count": 6 * level,
        "ptera_center_call_count": 6 * level,
        "ptera_offset_call_count": 18 * level,
        "fd_physical_evaluation_count": stage_count,
        "fd_tracer_evaluation_count": stage_count,
        "fd_evaluator_call_count": 24 * level,
        "transport_substep_count": level,
        "transport_stage_count": stage_count,
        "physical_field_call_count": stage_count,
        "tracer_field_call_count": stage_count,
        "observer_call_count": stage_count,
        "stage_pre_reconstruction_count": stage_count,
        "stage_post_reconstruction_count": stage_count,
        "physical_rhs_call_count": stage_count,
        "storage_reset_count": level,
        "tracer_storage_reset_count": level,
        "invariant_reference_freeze_count": 1,
        "sigma_storage_update_count": 0,
        "relaxation_call_count": 0,
        "compact_stage_record_count": stage_count,
        "retained_stage_array_count": 0,
        "native_collocation_evaluation_count": 1,
        "native_load_batch_evaluation_count": 4,
        "native_load_call_count": 1,
        "transport_macro_call_count": 1,
        "row_commit_count": 1,
    }
    for name, target in expected.items():
        if row[name] != target:
            raise ValueError(f"raw-step counter mismatch: {name}")
    if _exact_int(row["evidence_byte_count"], "evidence_byte_count") < 0:
        raise ValueError("evidence byte count is invalid")
    if (
        _finite_float(row["no_penetration_max_abs"], "no penetration")
        > NO_PENETRATION_MAX
    ):
        raise ValueError("no-penetration gate failed")
    if (
        _finite_float(row["kelvin_residual_max_abs"], "Kelvin residual")
        > KELVIN_RESIDUAL_MAX
    ):
        raise ValueError("Kelvin gate failed")
    if row["ptera_parent_sha256_before"] != row["ptera_parent_sha256_after"]:
        raise ValueError("Ptera parent digest changed")


def _validate_completed_digest_fields(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    label: str,
) -> None:
    for row in rows:
        if row.get("status") != "completed":
            continue
        for field in fields:
            if not _is_sha256(row[field]):
                raise ValueError(f"{label} completed digest is invalid: {field}")


def _validate_source_semantics(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], float]:
    residual_by_key: dict[tuple[int, int], float] = {}
    reference_by_source_step: dict[int, bytes] = {}
    _validate_completed_digest_fields(
        rows, ("event_sha256", "parent_event_sha256"), "source"
    )
    for row in rows:
        cell_count = _exact_int(row["cell_count"], "source cell_count", 1)
        source_step = _exact_int(row["source_step_index"], "source step", 1)
        ptera_step = _exact_int(row["ptera_step_index"], "Ptera step", 1)
        expected_birth_mode = (
            "first" if source_step == SOURCE_STEPS[0] else "continuous"
        )
        birth_modes = row["birth_modes"]
        if (
            type(birth_modes) not in (list, tuple)
            or len(birth_modes) != cell_count
            or tuple(birth_modes) != (expected_birth_mode,) * cell_count
        ):
            raise ValueError("source birth-mode vector is invalid")
        if (
            _finite_float(row["source_time_s"], "source time")
            != ptera_step * DELTA_TIME_S
        ):
            raise ValueError("source time differs from the frozen Ptera clock")
        vectors: dict[str, np.ndarray] = {}
        for field in SOURCE_VECTOR_FIELDS:
            value = row[field]
            if type(value) not in (list, tuple):
                raise ValueError(f"source vector is not a list/tuple: {field}")
            array = np.asarray(value)
            if (
                array.shape != (cell_count,)
                or array.dtype.kind not in "iuf"
                or array.dtype.kind == "b"
                or not np.all(np.isfinite(array))
            ):
                raise ValueError(f"source vector is non-finite or mis-sized: {field}")
            vectors[field] = np.asarray(array, dtype=np.float64)
        residual_max = float(
            np.max(np.abs(vectors["kelvin_residual_m2_s"]), initial=0.0)
        )
        if residual_max > KELVIN_RESIDUAL_MAX:
            raise ValueError("source Kelvin residual gate failed")
        source_semantics = dict(row)
        del source_semantics["transport_substeps"]
        semantic_bytes = _json_bytes(source_semantics)
        reference = reference_by_source_step.setdefault(source_step, semantic_bytes)
        if semantic_bytes != reference:
            raise ValueError("source generation differs across transport N")
        residual_by_key[_source_key(row)] = residual_max
    return residual_by_key


def _stable_panel_sum_and_tolerance(
    values: np.ndarray, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1:] != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"panel {label} values are non-finite or mis-shaped")
    try:
        summed = np.asarray(
            [math.fsum(float(item) for item in values[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
    except OverflowError as error:
        raise ValueError(f"panel {label} sum is unrepresentable") from error
    tolerances: list[float] = []
    epsilon_factor = 64.0 * np.finfo(np.float64).eps
    for axis in range(3):
        maximum = float(np.max(np.abs(values[:, axis]), initial=0.0))
        if maximum == 0.0:
            tolerance = epsilon_factor
        else:
            scaled_absolute_sum = math.fsum(
                float(abs(item) / maximum) for item in values[:, axis]
            )
            tolerance = max(
                epsilon_factor,
                maximum * (epsilon_factor * scaled_absolute_sum),
            )
        tolerances.append(float(tolerance))
    tolerance_array = np.asarray(tolerances, dtype=np.float64)
    if not np.all(np.isfinite(summed)) or not np.all(np.isfinite(tolerance_array)):
        raise ValueError(f"panel {label} reconciliation is non-finite")
    return summed, tolerance_array


def _require_panel_total_match(
    panel_values: np.ndarray, total_values: np.ndarray, *, label: str
) -> None:
    if total_values.shape != (3,) or not np.all(np.isfinite(total_values)):
        raise ValueError(f"TOTAL {label} values are non-finite or mis-shaped")
    summed, tolerance = _stable_panel_sum_and_tolerance(panel_values, label=label)
    try:
        residuals = tuple(
            math.fsum((float(summed[index]), -float(total_values[index])))
            for index in range(3)
        )
    except OverflowError as error:
        raise ValueError(f"panel {label} residual is unrepresentable") from error
    if any(
        not math.isfinite(value) or abs(value) > tolerance[index]
        for index, value in enumerate(residuals)
    ):
        raise ValueError(f"panel {label} values do not independently sum to TOTAL")


def _validate_load_block(
    raw: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> None:
    if len(rows) != 17:
        raise ValueError("layer load block does not contain exactly 17 rows")
    panels = rows[:16]
    total = rows[16]
    nullable_fields = (
        "force_coefficient_x",
        "force_coefficient_y",
        "force_coefficient_z",
        "raw_cl",
        "raw_cd",
    )
    if any(any(row[field] is not None for field in nullable_fields) for row in panels):
        raise ValueError("panel load rows must leave coefficient/CL/CD fields null")
    vector_fields = (
        "force_x_n",
        "force_y_n",
        "force_z_n",
        "moment_x_nm",
        "moment_y_nm",
        "moment_z_nm",
    )
    panel_values = np.asarray(
        [
            [
                _finite_float(row[field], f"panel load {field}")
                for field in vector_fields
            ]
            for row in panels
        ],
        dtype=np.float64,
    )
    total_values = np.asarray(
        [_finite_float(total[field], f"total load {field}") for field in vector_fields],
        dtype=np.float64,
    )
    _require_panel_total_match(panel_values[:, :3], total_values[:3], label="force")
    _require_panel_total_match(panel_values[:, 3:], total_values[3:], label="moment")
    coefficients = tuple(
        _finite_float(total[field], f"total load {field}")
        for field in (
            "force_coefficient_x",
            "force_coefficient_y",
            "force_coefficient_z",
        )
    )
    total_cl = _finite_float(total["raw_cl"], "total raw_cl")
    total_cd = _finite_float(total["raw_cd"], "total raw_cd")
    if total_cl != -coefficients[2] or total_cd != -coefficients[0]:
        raise ValueError("TOTAL CL/CD does not match the frozen force coefficients")
    if raw["raw_cl"] != total_cl or raw["raw_cd"] != total_cd:
        raise ValueError("raw-step CL/CD differs from the TOTAL load row")


def _validate_owner_semantics(
    owner_rows: Sequence[Mapping[str, object]],
    particle_rows: Sequence[Mapping[str, object]],
    trajectory_rows: Sequence[Mapping[str, object]],
) -> None:
    particle_by_key = {_layer_key(row): row for row in particle_rows}
    trajectory_by_key = {_layer_key(row): row for row in trajectory_rows}
    previous_by_level: dict[int, Mapping[str, object]] = {}
    previous_commit_by_level: dict[int, Mapping[str, object]] = {}
    previous_ids_by_level: dict[int, tuple[str, ...]] = {}
    previous_trajectory_by_level: dict[int, Mapping[str, object]] = {}
    owner_digest_fields = (
        *tuple(field for field in OWNER_EVENT_FIELDS if field.endswith("sha256")),
        "transport_parent_digest",
    )
    _validate_completed_digest_fields(owner_rows, owner_digest_fields, "owner-event")
    for row in owner_rows:
        level, layer = _layer_key(row)
        if row["schema_id"] != "fluxv-v5h13-baik-w2-owner-event-v1":
            raise ValueError("owner-event schema_id is invalid")
        changed = row["changed_particle_ids"]
        appended = row["appended_particle_ids"]
        if (
            type(changed) is not list
            or type(appended) is not list
            or any(
                type(value) is not str or not value for value in (*changed, *appended)
            )
            or len(set(changed)) != len(changed)
            or len(set(appended)) != len(appended)
            or not set(changed).isdisjoint(appended)
        ):
            raise ValueError("owner-event particle ID sequences are invalid")
        counts = particle_by_key[(level, layer)]
        if (
            len(changed) != counts["changed_particle_count"]
            or len(appended) != counts["appended_particle_count"]
        ):
            raise ValueError("owner-event ID sequences differ from particle counts")
        trajectory_id_sequence = tuple(
            trajectory_by_key[(level, layer)]["particle_ids"]
        )
        trajectory = trajectory_by_key[(level, layer)]
        arrays = trajectory["arrays"]
        assert isinstance(arrays, Mapping)
        end_positions = _trajectory_array(arrays["end_positions"])
        end_gamma = _trajectory_array(arrays["end_gamma"])
        end_sigma = _trajectory_array(arrays["end_sigma"])
        frontier_end = _trajectory_array(arrays["frontier_tracer_positions"])
        previous_ids = previous_ids_by_level.get(level)
        trajectory_ids = set(trajectory_id_sequence)
        if not set((*changed, *appended)).issubset(trajectory_ids):
            raise ValueError("owner-event IDs are absent from the trajectory ID set")
        transport = row["transport_event"]
        if not isinstance(transport, Mapping) or set(transport) != set(
            TRANSPORT_EVENT_FIELDS
        ):
            raise ValueError("owner transport-event schema is invalid")
        for field in TRANSPORT_EVENT_FIELDS:
            if (
                field.endswith("sha256") or field.endswith("digest")
            ) and not _is_sha256(transport[field]):
                raise ValueError(f"owner transport-event digest is invalid: {field}")
        transport_release = _exact_int(
            transport["release_index"], "owner transport release", 1
        )
        transport_source = _exact_int(
            transport["source_step_index"], "owner transport source step", 1
        )
        transport_end = _finite_float(
            transport["transport_end_time_s"], "owner transport end time"
        )
        if (
            transport_release != layer
            or transport_source != SOURCE_STEPS[layer - 1]
            or transport_end != SOURCE_STEPS[layer - 1] * DELTA_TIME_S
            or transport["parent_state_sha256"] != row["row_state_before_sha256"]
            or transport["transported_state_sha256"] != row["advanced_state_sha256"]
            or transport["parent_transport_digest"] != row["transport_parent_digest"]
            or transport["common_transport_attestation_sha256"]
            != row["transport_attestation_sha256"]
        ):
            raise ValueError("owner transport-event cross-links are invalid")
        if (
            transport["transported_arrays_sha256"]
            != _v5h10_digest(
                "fluxv-v5h10-transported-arrays-v1",
                end_positions,
                end_gamma,
                end_sigma,
            )
            or transport["live_boundary_nodes_sha256"]
            != _v5h10_digest("fluxv-v5h10-transported-live-nodes-v1", frontier_end)
            or transport["transport_event_sha256"]
            != _v5h10_event_digest(
                "fluxv-v5h10-row-transport-event-v1",
                transport,
                TRANSPORT_EVENT_FIELDS,
                "transport_event_sha256",
            )
        ):
            raise ValueError("owner transport-event replay digest mismatch")
        commit = row["commit_event"]
        previous = previous_by_level.get(level)
        if layer == 1:
            if commit is not None or previous is not None:
                raise ValueError("first owner layer must use the bootstrap commit")
            if changed or tuple(appended) != trajectory_id_sequence:
                raise ValueError("bootstrap owner row must append every particle")
            if transport["previous_transport_event_sha256"] != "0" * 64:
                raise ValueError("bootstrap owner transport genesis is invalid")
        else:
            if not isinstance(commit, Mapping) or set(commit) != set(
                COMMIT_EVENT_FIELDS
            ):
                raise ValueError("owner commit-event schema is invalid")
            for field in COMMIT_EVENT_FIELDS:
                if (
                    field.endswith("sha256") or field.endswith("digest")
                ) and not _is_sha256(commit[field]):
                    raise ValueError(f"owner commit-event digest is invalid: {field}")
            assert previous is not None
            previous_transport = previous["transport_event"]
            assert isinstance(previous_transport, Mapping)
            previous_commit = previous_commit_by_level.get(level)
            changed_indices = commit["changed_indices"]
            appended_indices = commit["appended_indices"]
            zero_counter_fields = (
                "clone_count",
                "counter_particle_count",
                "fresh_upstream_particle_count",
                "remesh_count",
                "ptera_call_count",
                "load_call_count",
                "feedback_call_count",
                "transport_call_count",
            )
            if (
                type(commit["proposal_id"]) is not str
                or not commit["proposal_id"]
                or type(changed_indices) is not list
                or type(appended_indices) is not list
                or any(
                    type(index) is not int
                    or index < 0
                    or index >= counts["particle_count"]
                    for index in (*changed_indices, *appended_indices)
                )
                or len(set(changed_indices)) != len(changed_indices)
                or len(set(appended_indices)) != len(appended_indices)
                or not set(changed_indices).isdisjoint(appended_indices)
                or len(changed_indices) != len(changed)
                or len(appended_indices) != len(appended)
                or tuple(changed)
                != tuple(trajectory_id_sequence[index] for index in changed_indices)
                or tuple(appended)
                != tuple(trajectory_id_sequence[index] for index in appended_indices)
                or commit["operator_order"]
                != "global_row_commit_before_ptera_before_lsrk3"
                or _exact_int(
                    commit["global_graph_build_count"],
                    "owner global_graph_build_count",
                )
                != 1
                or any(
                    _exact_int(commit[name], f"owner {name}") != 0
                    for name in zero_counter_fields
                )
            ):
                raise ValueError("owner commit-event index mapping is invalid")
            if (
                _exact_int(commit["release_index"], "owner commit release", 1) != layer
                or commit["parent_owner_sha256"] != previous["advanced_owner_sha256"]
                or commit["parent_state_sha256"] != previous["advanced_state_sha256"]
                or commit["committed_state_sha256"] != row["row_state_before_sha256"]
                or commit["parent_transport_digest"]
                != previous["transport_parent_digest"]
                or commit["parent_transport_event_sha256"]
                != previous_transport["transport_event_sha256"]
                or commit["previous_event_sha256"]
                != (
                    "0" * 64
                    if previous_commit is None
                    else previous_commit["event_sha256"]
                )
                or transport["previous_transport_event_sha256"]
                != previous_transport["transport_event_sha256"]
            ):
                raise ValueError("owner commit-event parent chain is invalid")
            previous_commit_by_level[level] = commit
            assert previous_ids is not None
            previous_trajectory = previous_trajectory_by_level[level]
            previous_arrays = previous_trajectory["arrays"]
            assert isinstance(previous_arrays, Mapping)
            previous_positions = _trajectory_array(previous_arrays["end_positions"])
            previous_gamma = _trajectory_array(previous_arrays["end_gamma"])
            previous_sigma = _trajectory_array(previous_arrays["end_sigma"])
            previous_frontier = _trajectory_array(
                previous_arrays["frontier_tracer_positions"]
            )
            start_positions = _trajectory_array(arrays["start_positions"])
            start_gamma = _trajectory_array(arrays["start_gamma"])
            start_sigma = _trajectory_array(arrays["start_sigma"])
            if (
                trajectory_id_sequence != (*previous_ids, *tuple(appended))
                or not set(changed).issubset(previous_ids)
                or not set(appended).isdisjoint(previous_ids)
            ):
                raise ValueError("owner particle-ID append ancestry is invalid")
            old_count = len(previous_ids)
            if _array_sha256(start_positions[:old_count]) != _array_sha256(
                previous_positions
            ) or _array_sha256(start_sigma[:old_count]) != _array_sha256(
                previous_sigma
            ):
                raise ValueError("row commit reset cumulative position/sigma state")
            changed_array = np.asarray(changed_indices, dtype=np.int64)
            unchanged_indices = np.asarray(
                [index for index in range(old_count) if index not in changed_indices],
                dtype=np.int64,
            )
            if _array_sha256(start_gamma[unchanged_indices]) != _array_sha256(
                previous_gamma[unchanged_indices]
            ):
                raise ValueError("row commit changed an unattested Gamma entry")
            before_gamma = previous_gamma[changed_array]
            after_gamma = start_gamma[changed_array]
            with np.errstate(over="ignore", invalid="ignore"):
                added_gamma = after_gamma - before_gamma
            if (
                not np.all(np.isfinite(before_gamma))
                or not np.all(np.isfinite(after_gamma))
                or not np.all(np.isfinite(added_gamma))
            ):
                raise ValueError("owner commit Gamma delta is non-finite")
            if (
                commit["before_gamma_sha256"]
                != _v5h10_digest("v5h10-before-gamma", before_gamma)
                or commit["after_gamma_sha256"]
                != _v5h10_digest("v5h10-after-gamma", after_gamma)
                or commit["added_gamma_sha256"]
                != _v5h10_digest("v5h10-added-gamma", added_gamma)
                or commit["upstream_nodes_sha256"]
                != _v5h10_digest(
                    "fluxv-v5h10-transported-live-nodes-v1", previous_frontier
                )
                or commit["event_sha256"]
                != _v5h10_event_digest(
                    "fluxv-v5h10-row-event-v1",
                    commit,
                    COMMIT_EVENT_FIELDS,
                    "event_sha256",
                    tuple_fields=("changed_indices", "appended_indices"),
                )
            ):
                raise ValueError("owner commit-event replay digest mismatch")
        previous_by_level[level] = row
        previous_ids_by_level[level] = trajectory_id_sequence
        previous_trajectory_by_level[level] = trajectory


def _validate_trajectory_metadata(row: Mapping[str, object]) -> None:
    metadata = row["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != set(
        TRAJECTORY_METADATA_FIELDS
    ):
        raise ValueError("trajectory metadata schema is invalid")
    manifests = metadata["source_cell_manifests"]
    if type(manifests) is not list or len(manifests) != 8:
        raise ValueError("trajectory source-cell manifest ledger is invalid")
    if metadata["source_cell_manifest_sha256"] != _payload_sha256(
        "fluxv-v5h13-source-cell-manifests-v1", manifests
    ):
        raise ValueError("trajectory source-cell manifest digest mismatch")
    prehistory = metadata["source_prehistory_manifests"]
    if type(prehistory) is not list or metadata[
        "source_prehistory_manifest_sha256"
    ] != _payload_sha256("fluxv-v5h13-source-prehistory-manifests-v1", prehistory):
        raise ValueError("trajectory source-prehistory manifest digest mismatch")
    layer = _exact_int(row["layer"], "trajectory metadata layer", 1)
    if (
        layer == 1
        and (
            len(prehistory) != 3
            or any(type(step) is not list or len(step) != 8 for step in prehistory)
        )
    ) or (layer != 1 and prehistory != []):
        raise ValueError("trajectory source-prehistory scope is invalid")
    for field in (
        "source_kelvin_ledger_sha256",
        "source_kelvin_evidence_sha256",
        "frontier_start_positions_sha256",
    ):
        if not _is_sha256(metadata[field]):
            raise ValueError(f"trajectory attested digest is invalid: {field}")
    if metadata["particle_id_sequence_sha256"] != _payload_sha256(
        "fluxv-v5h13-particle-id-sequence-v1", list(row["particle_ids"])
    ):
        raise ValueError("trajectory particle-ID sequence digest mismatch")
    if metadata["material_tracer_id_sequence_sha256"] != _payload_sha256(
        "fluxv-v5h13-material-id-sequence-v1",
        list(row["material_tracer_ids"]),
    ):
        raise ValueError("trajectory material-ID sequence digest mismatch")
    if metadata["fixed_probe_contract"] != [
        list(point) for point in FIXED_PROBES_GP1_M
    ]:
        raise ValueError("trajectory fixed-probe contract drifted")


def _source_pair(value: object, name: str) -> tuple[float, float]:
    if type(value) not in (list, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an exact two-component pair")
    return (
        _finite_float(value[0], f"{name}[0]"),
        _finite_float(value[1], f"{name}[1]"),
    )


def _validate_source_provenance_v3(
    value: object, *, lesp_critical: object
) -> tuple[Mapping[str, object], str]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_PROVENANCE_FIELDS):
        raise ValueError("source-cell provenance schema is invalid")
    exact = {
        "interface_id": SOURCE_INTERFACE_ID,
        "backend_id": SOURCE_BACKEND_ID,
        "circulation_units": "Gamma/(U_ref*c_ref)",
        "position_units": "(x/c_ref,z/c_ref)",
        "position_frame": SOURCE_POSITION_FRAME,
        "circulation_sign": SOURCE_CIRCULATION_SIGN,
        "tev_birth_law": SOURCE_TEV_BIRTH_LAW,
        "lev_birth_law": SOURCE_LEV_BIRTH_LAW,
        "birth_time_layer": SOURCE_BIRTH_TIME_LAYER,
        "dimensionalization_limitations": SOURCE_DIMENSIONALIZATION_LIMITATIONS,
        "source_solver": "clean_linear",
        "canonical_blocker": SOURCE_CANONICAL_BLOCKER,
        "bottom_model_parity": SOURCE_BOTTOM_MODEL_PARITY,
        "ownership_scope": SOURCE_OWNERSHIP_SCOPE,
        "observation_access": OBSERVATION_ACCESS,
        "target_case_branch": "none",
    }
    if any(value[name] != expected for name, expected in exact.items()):
        raise ValueError("source-cell provenance identity/domain drifted")
    if value["source_parity"] is not True or value["canonical"] is not False:
        raise ValueError("source-cell provenance trust boundary drifted")
    text_fields = (
        "physical_section_id",
        "physical_strip_id",
        "section_family",
        "threshold_source",
        "threshold_source_role",
        "geometry_identity",
        "geometry_role",
        "position_frame",
        "circulation_sign",
        "tev_birth_law",
        "lev_birth_law",
        "birth_time_layer",
        "dimensionalization_limitations",
        "canonical_blocker",
        "bottom_model_parity",
    )
    if any(type(value[name]) is not str or not value[name] for name in text_fields):
        raise ValueError("source-cell provenance text identity is invalid")
    if value["physical_section_id"] == value["physical_strip_id"]:
        raise ValueError("source-cell section and strip identities must differ")
    if value["threshold_source_role"] not in {
        "published_model_parameter",
        "published_source_input",
        "independent_non_target_calibration",
    } or value["geometry_role"] not in {
        "explicit_zero_camber_surrogate",
        "explicit_paired_camber_ordinate_and_slope",
    }:
        raise ValueError("source-cell provenance role is invalid")
    if not _is_sha256(value["geometry_hash_sha256"]):
        raise ValueError("source-cell geometry digest is invalid")
    positive_fields = (
        "reynolds",
        "lesp_critical",
        "delta_time_convective_nominal",
        "resolved_core_radius_chord",
        "circulation_scale_u_times_c_m2_per_s",
        "position_scale_chord_m",
    )
    if any(
        _finite_float(value[name], f"source-cell provenance {name}") <= 0.0
        for name in positive_fields
    ):
        raise ValueError("source-cell provenance positive scalar is invalid")
    if value["lesp_critical"] != lesp_critical:
        raise ValueError("source-cell provenance LESP threshold drifted")
    pivot = _finite_float(value["pivot_fraction_chord"], "source-cell provenance pivot")
    ndiv = _exact_int(value["ndiv"], "source-cell provenance ndiv", 12)
    naterm = _exact_int(value["naterm"], "source-cell provenance naterm", 3)
    _exact_int(value["max_wake_steps"], "source-cell provenance max_wake_steps", 8)
    station_count = _exact_int(
        value["geometry_station_count"],
        "source-cell provenance geometry_station_count",
        12,
    )
    if not 0.0 <= pivot <= 1.0 or naterm >= ndiv or station_count != ndiv:
        raise ValueError("source-cell provenance geometry/count domain is invalid")
    threshold = {
        "value": value["lesp_critical"],
        "section_family": value["section_family"],
        "reynolds": value["reynolds"],
        "source": value["threshold_source"],
        "source_role": value["threshold_source_role"],
    }
    lineage_id = (
        "dvm-section-"
        + sha256(
            _json_bytes({"provenance": dict(value), "threshold": threshold})
        ).hexdigest()[:20]
    )
    return value, lineage_id


def _validate_formal_w2_source_identity(trajectory: Mapping[str, object]) -> None:
    """Pin every replayed source cell to the frozen Baik W2 source contract."""

    metadata = trajectory["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("formal W2 source metadata is invalid")
    active = metadata["source_cell_manifests"]
    prehistory = metadata["source_prehistory_manifests"]
    if type(active) is not list or type(prehistory) is not list:
        raise ValueError("formal W2 source manifest collections are invalid")
    manifest_groups = [*prehistory, active]
    for manifests in manifest_groups:
        if type(manifests) is not list or len(manifests) != 8:
            raise ValueError("formal W2 source manifest width is invalid")
        for cell, manifest in enumerate(manifests):
            if not isinstance(manifest, Mapping):
                raise ValueError("formal W2 source manifest is invalid")
            provenance = manifest.get("provenance")
            expected = {
                "interface_id": SOURCE_INTERFACE_ID,
                "backend_id": SOURCE_BACKEND_ID,
                "physical_section_id": f"baik-w2:cell:{cell}:section",
                "physical_strip_id": f"baik-w2:cell:{cell}:strip",
                "section_family": "rounded flat plate",
                "reynolds": 5000.0,
                "lesp_critical": 0.11,
                "threshold_source": FORMAL_W2_THRESHOLD_SOURCE,
                "threshold_source_role": "published_source_input",
                "delta_time_convective_nominal": FORMAL_W2_CONVECTIVE_DT,
                "pivot_fraction_chord": 0.25,
                "ndiv": 32,
                "naterm": 14,
                "resolved_core_radius_chord": 0.02,
                "max_wake_steps": 64,
                "geometry_identity": FORMAL_W2_GEOMETRY_IDENTITY,
                "geometry_hash_sha256": FORMAL_W2_GEOMETRY_SHA256,
                "geometry_role": "explicit_zero_camber_surrogate",
                "geometry_station_count": 32,
                "circulation_units": "Gamma/(U_ref*c_ref)",
                "circulation_scale_u_times_c_m2_per_s": (FORMAL_W2_CIRCULATION_SCALE),
                "position_units": "(x/c_ref,z/c_ref)",
                "position_scale_chord_m": 0.076,
                "position_frame": SOURCE_POSITION_FRAME,
                "circulation_sign": SOURCE_CIRCULATION_SIGN,
                "tev_birth_law": SOURCE_TEV_BIRTH_LAW,
                "lev_birth_law": SOURCE_LEV_BIRTH_LAW,
                "birth_time_layer": SOURCE_BIRTH_TIME_LAYER,
                "dimensionalization_limitations": (
                    SOURCE_DIMENSIONALIZATION_LIMITATIONS
                ),
                "source_parity": True,
                "source_solver": "clean_linear",
                "canonical": False,
                "canonical_blocker": SOURCE_CANONICAL_BLOCKER,
                "bottom_model_parity": SOURCE_BOTTOM_MODEL_PARITY,
                "ownership_scope": SOURCE_OWNERSHIP_SCOPE,
                "observation_access": OBSERVATION_ACCESS,
                "target_case_branch": "none",
            }
            if (
                not isinstance(provenance, Mapping)
                or set(expected) != set(SOURCE_PROVENANCE_FIELDS)
                or dict(provenance) != expected
                or manifest.get("delta_time_convective") != FORMAL_W2_CONVECTIVE_DT
            ):
                raise ValueError("formal W2 source provenance/event identity drifted")


def _validate_source_lineage_v3(
    value: object,
    *,
    provenance: Mapping[str, object],
    lineage_id: str,
    source_step: int,
    expected_mode: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_LINEAGE_FIELDS):
        raise ValueError("source-cell lineage schema is invalid")
    active = expected_mode != "none"
    max_wake = _exact_int(provenance["max_wake_steps"], "source-cell max_wake_steps", 8)
    lev_before = max(0, source_step - SOURCE_STEPS[0])
    expected_tev_role = (
        "constraint_column_solved_then_zeroed_before_persistence"
        if source_step == 1
        else "coupled_newest_persisted_step_source"
    )
    expected_prefix = f"{lineage_id}:step:{source_step}"
    if (
        value["physical_section_id"] != provenance["physical_section_id"]
        or value["physical_strip_id"] != provenance["physical_strip_id"]
        or value["section_lineage_id"] != lineage_id
        or value["source_step_index"] != source_step
        or value["parent_state_step_index"] != source_step - 1
        or value["persistent_history_exported"] is not False
        or value["persistent_tev_history_role"]
        != "backend-owned convected TE history; not re-exported as newborn"
        or value["persistent_lev_history_role"]
        != "backend-owned convected LE history; not re-exported as newborn"
        or value["persistent_tev_count_before"] != min(source_step - 1, max_wake)
        or value["persistent_tev_count_after"] != min(source_step, max_wake)
        or value["persistent_lev_count_before"] != lev_before
        or value["persistent_lev_count_after"]
        != min(lev_before + int(active), max_wake)
        or value["newborn_tev_source_id"] != f"{expected_prefix}:tev-newborn"
        or value["newborn_tev_role"] != expected_tev_role
        or value["newborn_lev_source_id"]
        != (f"{expected_prefix}:lev-newborn" if active else None)
        or value["newborn_lev_role"]
        != (
            "lesp_gated_newborn_persisted_step_source"
            if active
            else "inactive_no_newborn_source"
        )
    ):
        raise ValueError("source-cell lineage identity/count transition is invalid")


def _validate_source_kelvin_ledger_v3(
    value: object,
    *,
    manifest: Mapping[str, object],
    source_step: int,
    active: bool,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_KELVIN_LEDGER_FIELDS):
        raise ValueError("source-cell Kelvin-ledger schema is invalid")
    if (
        value["circulation_units"] != "Gamma/(U_ref*c_ref)"
        or type(value["first_tev_zeroed"]) is not bool
    ):
        raise ValueError("source-cell Kelvin-ledger domain is invalid")
    if value["first_tev_zeroed"] != (source_step == 1):
        raise ValueError("source-cell first-TEV persistence flag is invalid")
    numeric_fields = tuple(
        name
        for name in SOURCE_KELVIN_LEDGER_FIELDS
        if name not in {"circulation_units", "first_tev_zeroed"}
    )
    ledger = {
        name: _finite_float(value[name], f"source-cell ledger {name}")
        for name in numeric_fields
    }
    if (
        manifest["gamma_tev_new_solved_over_u_c"] != ledger["gamma_tev_new_solved"]
        or manifest["gamma_tev_new_persisted_over_u_c"]
        != ledger["gamma_tev_new_persisted"]
        or manifest["gamma_lev_new_over_u_c"] != ledger["gamma_lev_new_persisted"]
        or manifest["kelvin_residual_over_u_c"] != ledger["kelvin_solve_residual"]
        or ledger["gamma_lev_new_solved"] != ledger["gamma_lev_new_persisted"]
        or (
            not active
            and ledger["gamma_tev_new_te_only_provisional"]
            != ledger["gamma_tev_new_solved"]
        )
        or (source_step == 1 and ledger["gamma_tev_new_persisted"] != 0.0)
        or (
            source_step != 1
            and ledger["gamma_tev_new_persisted"] != ledger["gamma_tev_new_solved"]
        )
    ):
        raise ValueError("source-cell event/Kelvin-ledger fields disagree")
    try:
        solve = math.fsum(
            (
                -ledger["gamma_bound_post"],
                ledger["gamma_old_tev_persisted"],
                ledger["gamma_old_lev_persisted"],
                ledger["gamma_deleted_before"],
                ledger["gamma_tev_new_solved"],
                ledger["gamma_lev_new_solved"],
            )
        )
        persistence = math.fsum(
            (
                ledger["gamma_tev_persisted_after"],
                ledger["gamma_lev_persisted_after"],
                ledger["gamma_deleted_after"],
                -ledger["gamma_old_tev_persisted"],
                -ledger["gamma_old_lev_persisted"],
                -ledger["gamma_deleted_before"],
                -ledger["gamma_tev_new_persisted"],
                -ledger["gamma_lev_new_persisted"],
            )
        )
        deleted_delta = math.fsum(
            (ledger["gamma_deleted_after"], -ledger["gamma_deleted_before"])
        )
        tev_delta = math.fsum(
            (
                ledger["gamma_tev_new_persisted"],
                -ledger["gamma_tev_new_solved"],
            )
        )
    except OverflowError as error:
        raise ValueError("source-cell Kelvin ledger overflowed replay") from error
    scale = max(1.0, *(abs(item) for item in ledger.values()))
    tolerance = 1024.0 * np.finfo(np.float64).eps * scale
    identities = (
        (solve, ledger["kelvin_solve_residual"]),
        (persistence, ledger["persistence_residual"]),
        (
            deleted_delta,
            ledger["gamma_deleted_delta"],
        ),
        (
            tev_delta,
            ledger["tev_solved_to_persisted_delta"],
        ),
    )
    if not all(
        math.isfinite(item) for pair in identities for item in pair
    ) or not math.isfinite(tolerance):
        raise ValueError("source-cell Kelvin ledger replay is non-finite")
    if any(abs(recomputed - saved) > tolerance for recomputed, saved in identities) or (
        abs(ledger["kelvin_solve_residual"]) > tolerance
        or abs(ledger["persistence_residual"]) > tolerance
    ):
        raise ValueError("source-cell Kelvin ledger is not independently replayable")
    return ledger


def _direct_vatistas2_single_velocity(
    *,
    target: tuple[float, float],
    source: tuple[float, float],
    gamma: float,
    core: float,
) -> tuple[float, float]:
    rx = target[0] - source[0]
    ry = target[1] - source[1]
    r2 = rx * rx + ry * ry
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factor = gamma / (2.0 * np.pi) / np.sqrt(r2 * r2 + core**4)
        result = (float(factor * (-ry)), float(factor * rx))
    if not all(math.isfinite(item) for item in result):
        raise ValueError("source-cell provisional-TEV velocity replay is non-finite")
    return result


def _validate_source_placement_v3(
    value: object,
    *,
    family: str,
    expected_mode: str,
    legacy_birth: object,
    delta_time_convective: object,
    section_lineage_id: str,
    source_step: int,
    provisional_tev_gamma: float,
    provisional_tev_birth: object,
    resolved_core_radius: float,
) -> None:
    """Replay the v3 provisional-TEV velocity decomposition without importing it."""

    if not isinstance(value, Mapping) or set(value) != set(SOURCE_PLACEMENT_FIELDS):
        raise ValueError(f"source-cell {family} placement schema is invalid")
    if (
        value["schema_id"] != SOURCE_PLACEMENT_SCHEMA_ID
        or value["vortex_family"] != family
        or value["placement_mode"] != expected_mode
        or type(value["used_for_topology_eligible"]) is not bool
        or value["used_for_topology_eligible"]
        != (family == "lev" and expected_mode in {"first", "restart"})
    ):
        raise ValueError(f"source-cell {family} placement identity drifted")
    edge = _source_pair(
        value["edge_anchor_position_over_chord_backend_world"],
        f"source-cell {family} edge",
    )
    if expected_mode == "inactive":
        nullable = tuple(name for name in SOURCE_PLACEMENT_FIELDS[4:-1])
        if (
            family != "lev"
            or legacy_birth is not None
            or any(value[name] is not None for name in nullable)
        ):
            raise ValueError("source-cell inactive LEV placement is invalid")
        return
    birth = _source_pair(
        value["birth_position_over_chord_backend_world"],
        f"source-cell {family} birth",
    )
    displacement = _source_pair(
        value["birth_displacement_from_edge_over_chord_backend_world"],
        f"source-cell {family} displacement",
    )
    if birth != _source_pair(legacy_birth, f"legacy source-cell {family} birth") or (
        displacement[0] != float(birth[0] - edge[0])
        or displacement[1] != float(birth[1] - edge[1])
    ):
        raise ValueError(f"source-cell {family} placement is not replayable")
    q_fields = (
        "q_birth_over_u_backend_world",
        "q_kinematic_over_u_backend_world",
        "q_old_wake_over_u_backend_world",
        "q_provisional_tev_over_u_backend_world",
    )
    if expected_mode in {"first", "restart"}:
        q_birth, q_kinematic, q_old_wake, q_provisional_tev = (
            _source_pair(value[name], f"source-cell {family} {name}")
            for name in q_fields
        )
        expected_q = (
            float(q_kinematic[0] + q_old_wake[0] + q_provisional_tev[0]),
            float(q_kinematic[1] + q_old_wake[1] + q_provisional_tev[1]),
        )
        delta_time = _finite_float(
            delta_time_convective, "source-cell delta_time_convective"
        )
        reconstructed = (
            float(edge[0] + 0.5 * q_birth[0] * delta_time),
            float(edge[1] + 0.5 * q_birth[1] * delta_time),
        )
        if (
            q_birth != expected_q
            or reconstructed != birth
            or value["continuous_parent_source_id"] is not None
            or value["continuous_parent_position_over_chord_backend_world"] is not None
        ):
            raise ValueError(
                f"source-cell {family} provisional-TEV decomposition is invalid"
            )
        if family == "lev" and q_provisional_tev != _direct_vatistas2_single_velocity(
            target=edge,
            source=_source_pair(
                provisional_tev_birth, "source-cell provisional TEV birth"
            ),
            gamma=provisional_tev_gamma,
            core=resolved_core_radius,
        ):
            raise ValueError(
                "source-cell LEV provisional-TEV velocity is not replayable"
            )
        return
    if expected_mode != "continuous" or any(
        value[name] is not None for name in q_fields
    ):
        raise ValueError(f"source-cell {family} placement mode is unsupported")
    if (
        type(value["continuous_parent_source_id"]) is not str
        or not value["continuous_parent_source_id"]
    ):
        raise ValueError(f"source-cell {family} continuous parent ID is invalid")
    parent = _source_pair(
        value["continuous_parent_position_over_chord_backend_world"],
        f"source-cell {family} continuous parent position",
    )
    if value["continuous_parent_source_id"] != (
        f"{section_lineage_id}:step:{source_step - 1}:{family}-newborn"
    ) or birth != (
        float(edge[0] + (parent[0] - edge[0]) / 3.0),
        float(edge[1] + (parent[1] - edge[1]) / 3.0),
    ):
        raise ValueError(f"source-cell {family} continuous placement is not replayable")


def _validated_source_cell_manifest_v3(
    manifest: object, *, source_step: int, expected_mode: str
) -> dict[str, object]:
    if not isinstance(manifest, Mapping) or set(manifest) != set(
        SOURCE_CELL_MANIFEST_FIELDS
    ):
        raise ValueError("source-cell manifest schema is invalid")
    producer = manifest["producer_manifest_sha256"]
    parent = manifest["parent_event_manifest_sha256"]
    if not _is_sha256(producer) or not _is_sha256(parent):
        raise ValueError("source-cell manifest lineage digest is invalid")
    unsigned = dict(manifest)
    del unsigned["producer_manifest_sha256"]
    if (
        producer
        != sha256(SOURCE_EVENT_DIGEST_PREFIX + _json_bytes(unsigned)).hexdigest()
    ):
        raise ValueError("source-cell producer manifest digest mismatch")
    active = expected_mode != "none"
    if (
        manifest["enabled"] is not True
        or manifest["status"] != "evaluated_source_only_noncanonical_d0_unconsumed"
        or manifest["lesp_active"] is not active
        or manifest["restart"] is not False
        or manifest["lev_birth_mode"] != expected_mode
    ):
        raise ValueError("source-cell fixed source schedule is invalid")
    delta_time = _finite_float(
        manifest["delta_time_convective"], "source-cell delta_time_convective"
    )
    if delta_time <= 0.0:
        raise ValueError("source-cell delta_time_convective must be positive")
    critical = _finite_float(manifest["lesp_critical"], "source-cell LESP critical")
    a0_pre = _finite_float(manifest["a0_pre"], "source-cell a0_pre")
    a0_post = _finite_float(manifest["a0_post"], "source-cell a0_post")
    target = _finite_float(
        manifest["lesp_signed_target"], "source-cell LESP signed target"
    )
    residual = _finite_float(
        manifest["lesp_constraint_residual"], "source-cell LESP residual"
    )
    if critical <= 0.0 or active != (abs(a0_pre) > critical):
        raise ValueError("source-cell LESP activity is not replayable")
    tolerance = (
        512.0 * np.finfo(np.float64).eps * max(1.0, abs(a0_pre), abs(a0_post), critical)
    )
    if active:
        expected_target = float(np.clip(a0_pre, -critical, critical))
        if (
            target != expected_target
            or abs(residual - (a0_post - target)) > tolerance
            or abs(residual) > tolerance
        ):
            raise ValueError("source-cell active LESP reduction is not replayable")
    elif not (
        a0_pre == a0_post
        and target == a0_pre
        and residual == 0.0
        and manifest["gamma_lev_new_over_u_c"] == 0.0
        and manifest["lev_birth_position_over_chord_backend_world"] is None
    ):
        raise ValueError("source-cell inactive LESP reduction is not exact")
    provenance, lineage_id = _validate_source_provenance_v3(
        manifest["provenance"], lesp_critical=manifest["lesp_critical"]
    )
    _validate_source_lineage_v3(
        manifest["lineage"],
        provenance=provenance,
        lineage_id=lineage_id,
        source_step=source_step,
        expected_mode=expected_mode,
    )
    ledger = _validate_source_kelvin_ledger_v3(
        manifest["kelvin_ledger"],
        manifest=manifest,
        source_step=source_step,
        active=active,
    )
    scale = _finite_float(
        provenance["circulation_scale_u_times_c_m2_per_s"],
        "source-cell circulation scale",
    )
    core = _finite_float(
        provenance["resolved_core_radius_chord"], "source-cell core radius"
    )
    _validate_source_placement_v3(
        manifest["lev_placement"],
        family="lev",
        expected_mode=("inactive" if expected_mode == "none" else expected_mode),
        legacy_birth=manifest["lev_birth_position_over_chord_backend_world"],
        delta_time_convective=delta_time,
        section_lineage_id=lineage_id,
        source_step=source_step,
        provisional_tev_gamma=ledger["gamma_tev_new_te_only_provisional"],
        provisional_tev_birth=manifest["tev_birth_position_over_chord_backend_world"],
        resolved_core_radius=core,
    )
    _validate_source_placement_v3(
        manifest["tev_placement"],
        family="tev",
        expected_mode=("first" if source_step == 1 else "continuous"),
        legacy_birth=manifest["tev_birth_position_over_chord_backend_world"],
        delta_time_convective=delta_time,
        section_lineage_id=lineage_id,
        source_step=source_step,
        provisional_tev_gamma=ledger["gamma_tev_new_te_only_provisional"],
        provisional_tev_birth=manifest["tev_birth_position_over_chord_backend_world"],
        resolved_core_radius=core,
    )
    return {
        "a0_post": a0_post,
        "a0_pre": a0_pre,
        "birth_mode": expected_mode,
        "gamma_lev_new_m2_s": _finite_float(
            manifest["gamma_lev_new_over_u_c"], "source-cell gamma_lev_new"
        )
        * scale,
        "gamma_lev_persisted_m2_s": ledger["gamma_lev_persisted_after"] * scale,
        "gamma_tev_new_m2_s": _finite_float(
            manifest["gamma_tev_new_solved_over_u_c"],
            "source-cell gamma_tev_new",
        )
        * scale,
        "kelvin_residual_m2_s": _finite_float(
            manifest["kelvin_residual_over_u_c"], "source-cell Kelvin residual"
        )
        * scale,
        "ledger": dict(manifest["kelvin_ledger"]),
        "lineage_id": lineage_id,
        "parent": parent,
        "producer": producer,
    }


def _validate_source_manifest_binding(
    source: Mapping[str, object],
    trajectory: Mapping[str, object],
    previous_trajectory: Mapping[str, object] | None,
) -> None:
    metadata = trajectory["metadata"]
    assert isinstance(metadata, Mapping)
    manifests = metadata["source_cell_manifests"]
    assert type(manifests) is list
    producers: list[str] = []
    parents: list[str] = []
    ledgers: list[object] = []
    reconstructed: dict[str, list[object]] = {
        field: [] for field in (*SOURCE_VECTOR_FIELDS, "birth_modes")
    }
    lineage_ids: list[str] = []
    expected_mode = (
        "first" if source["source_step_index"] == SOURCE_STEPS[0] else "continuous"
    )
    for manifest in manifests:
        validated = _validated_source_cell_manifest_v3(
            manifest,
            source_step=int(source["source_step_index"]),
            expected_mode=expected_mode,
        )
        lineage_ids.append(str(validated["lineage_id"]))
        producers.append(str(validated["producer"]))
        parents.append(str(validated["parent"]))
        ledgers.append(validated["ledger"])
        reconstructed["birth_modes"].append(validated["birth_mode"])
        for field in SOURCE_VECTOR_FIELDS:
            reconstructed[field].append(validated[field])
    if len(set(lineage_ids)) != len(lineage_ids):
        raise ValueError("source-cell lineage IDs are duplicated")
    if source["event_sha256"] != _payload_sha256(
        "fluxv-v5h13-source-event-aggregate-v1", producers
    ) or source["parent_event_sha256"] != _payload_sha256(
        "fluxv-v5h13-source-parent-aggregate-v1", parents
    ):
        raise ValueError("source aggregate event/parent digest mismatch")
    if metadata["source_kelvin_ledger_sha256"] != _payload_sha256(
        "fluxv-v5h13-source-kelvin-ledgers-v1", ledgers
    ):
        raise ValueError("source aggregate Kelvin-ledger digest mismatch")
    for field, expected in reconstructed.items():
        if source[field] != expected:
            raise ValueError(f"source aggregate vector differs from manifests: {field}")
    if previous_trajectory is not None:
        previous_metadata = previous_trajectory["metadata"]
        assert isinstance(previous_metadata, Mapping)
        previous_manifests = previous_metadata["source_cell_manifests"]
        assert type(previous_manifests) is list
        if len(previous_manifests) != len(manifests):
            raise ValueError("source-cell lineage width changed between layers")
        for previous, current in zip(previous_manifests, manifests, strict=True):
            assert isinstance(previous, Mapping) and isinstance(current, Mapping)
            if (
                previous["lineage"]["section_lineage_id"]
                != current["lineage"]["section_lineage_id"]
                or current["parent_event_manifest_sha256"]
                != previous["producer_manifest_sha256"]
            ):
                raise ValueError("source-cell parent chain is discontinuous")
    else:
        prehistory = metadata["source_prehistory_manifests"]
        assert type(prehistory) is list and len(prehistory) == 3
        validated_history: list[list[dict[str, object]]] = []
        for source_step, step_manifests in enumerate(prehistory, start=1):
            assert type(step_manifests) is list and len(step_manifests) == 8
            validated_history.append(
                [
                    _validated_source_cell_manifest_v3(
                        item, source_step=source_step, expected_mode="none"
                    )
                    for item in step_manifests
                ]
            )
        for cell_index, current in enumerate(manifests):
            assert isinstance(current, Mapping)
            expected_parent = sha256(
                _json_bytes([SOURCE_EVENT_CHAIN_DOMAIN, lineage_ids[cell_index]])
            ).hexdigest()
            for step in validated_history:
                item = step[cell_index]
                if (
                    item["lineage_id"] != lineage_ids[cell_index]
                    or item["parent"] != expected_parent
                ):
                    raise ValueError("source-cell prehistory chain is discontinuous")
                expected_parent = str(item["producer"])
            if current["parent_event_manifest_sha256"] != expected_parent:
                raise ValueError("source-cell active row lacks exact prehistory parent")


def _validate_stage_state_continuity(
    stages: Sequence[Mapping[str, object]], trajectory: Mapping[str, object]
) -> None:
    arrays = trajectory["arrays"]
    assert isinstance(arrays, Mapping)
    if not stages:
        raise ValueError("completed layer has no durable transport stages")
    for previous, current in zip(stages, stages[1:]):
        if (
            previous["post_state_sha256"] != current["pre_state_sha256"]
            or previous["tracer_post_sha256"] != current["tracer_pre_sha256"]
        ):
            raise ValueError("transport-stage state/tracer chain is discontinuous")
    expected_start = _stream_state_sha256_from_arrays(
        _trajectory_array(arrays["start_positions"]),
        _trajectory_array(arrays["start_gamma"]),
        _trajectory_array(arrays["start_sigma"]),
    )
    expected_end = _stream_state_sha256_from_arrays(
        _trajectory_array(arrays["end_positions"]),
        _trajectory_array(arrays["end_gamma"]),
        _trajectory_array(arrays["end_sigma"]),
    )
    final_tracers = np.concatenate(
        (
            _trajectory_array(arrays["material_tracer_positions"]),
            _trajectory_array(arrays["frontier_tracer_positions"]),
        ),
        axis=0,
    )
    if (
        stages[0]["pre_state_sha256"] != expected_start
        or stages[-1]["post_state_sha256"] != expected_end
        or stages[-1]["tracer_post_sha256"] != _array_sha256(final_tracers)
    ):
        raise ValueError("transport-stage terminal state/tracer digest mismatch")


def _validate_completed_crosslinks(
    *,
    execution_mode: str,
    raw_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
    owner_rows: Sequence[Mapping[str, object]],
    particle_rows: Sequence[Mapping[str, object]],
    load_rows: Sequence[Mapping[str, object]],
    stage_rows: Sequence[Mapping[str, object]],
    trajectory_rows: Sequence[Mapping[str, object]],
) -> None:
    _validate_completed_digest_fields(
        raw_rows,
        tuple(field for field in RAW_STEP_FIELDS if "sha256" in field),
        "raw-step",
    )
    _validate_completed_digest_fields(
        stage_rows,
        tuple(field for field in STAGE_FIELDS if "sha256" in field),
        "transport-stage",
    )
    source_residual = _validate_source_semantics(source_rows)
    _validate_owner_semantics(owner_rows, particle_rows, trajectory_rows)
    source_by_key = {_source_key(row): row for row in source_rows}
    owner_by_key = {_layer_key(row): row for row in owner_rows}
    particle_by_key = {_layer_key(row): row for row in particle_rows}
    trajectory_by_key = {_layer_key(row): row for row in trajectory_rows}
    stages_by_key: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for stage in stage_rows:
        if stage["status"] == "completed":
            stages_by_key.setdefault(_layer_key(stage), []).append(stage)
    loads_by_key: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for load in load_rows:
        loads_by_key.setdefault(_layer_key(load), []).append(load)
    previous_trajectory_by_level: dict[int, Mapping[str, object]] = {}
    for raw in raw_rows:
        key = _layer_key(raw)
        owner = owner_by_key[key]
        particle = particle_by_key[key]
        common_count_fields = (
            "particle_count",
            "material_tracer_count",
            "material_support_tracer_count",
            "frontier_node_tracer_count",
        )
        if any(raw[field] != particle[field] for field in common_count_fields):
            raise ValueError("raw-step counts differ from particle_counts")
        if (
            raw["material_tracer_count"]
            != raw["material_support_tracer_count"] + raw["frontier_node_tracer_count"]
        ):
            raise ValueError("material tracer count is not support plus frontier")
        if (
            raw["row_owner_before_sha256"] != owner["row_owner_before_sha256"]
            or raw["advanced_owner_sha256"] != owner["advanced_owner_sha256"]
        ):
            raise ValueError("raw-step owner digests differ from owner_events")
        stages = stages_by_key[key]
        if raw["stream_stage_chain_sha256"] != stages[-1]["chain_sha256"]:
            raise ValueError("raw-step stage-chain digest differs from its final stage")
        frozen_parent = raw["ptera_parent_sha256_before"]
        if any(
            stage["ptera_parent_sha256_before"] != frozen_parent
            or stage["ptera_parent_sha256_after"] != frozen_parent
            for stage in stages
        ):
            raise ValueError(
                "stage Ptera parent differs from the raw-step frozen parent"
            )
        source_key = (key[0], SOURCE_STEPS[key[1] - 1])
        if raw["kelvin_residual_max_abs"] != source_residual[source_key]:
            raise ValueError("raw-step Kelvin residual differs from source_events")
        trajectory = trajectory_by_key[key]
        _validate_trajectory_metadata(trajectory)
        _validate_source_manifest_binding(
            source_by_key[source_key],
            trajectory,
            previous_trajectory_by_level.get(key[0]),
        )
        if execution_mode == FORMAL_EXECUTION_MODE:
            _validate_formal_w2_source_identity(trajectory)
        previous_trajectory_by_level[key[0]] = trajectory
        metadata = trajectory["metadata"]
        assert isinstance(metadata, Mapping)
        expected_kelvin_evidence = _payload_sha256(
            "fluxv-v5h13-source-kelvin-v1",
            {
                "atol_m2_s": float(KELVIN_RESIDUAL_MAX).hex(),
                "kelvin_ledger_sha256": metadata["source_kelvin_ledger_sha256"],
                "residual_m2_s": float(source_residual[source_key]).hex(),
                "row_owner_sha256": raw["row_owner_before_sha256"],
                "source_event_sha256": source_by_key[source_key]["event_sha256"],
                "source_step_index": source_key[1],
            },
        )
        if metadata["source_kelvin_evidence_sha256"] != expected_kelvin_evidence:
            raise ValueError("trajectory source-Kelvin evidence digest mismatch")
        arrays = trajectory["arrays"]
        assert isinstance(arrays, Mapping)
        decoded_arrays = {
            name: _trajectory_array(value) for name, value in arrays.items()
        }
        _validate_probe_oracle(decoded_arrays)
        _validate_stage_state_continuity(stages, trajectory)
        no_penetration = _trajectory_array(arrays["no_penetration_residual"])
        residual_max = float(np.max(np.abs(no_penetration), initial=0.0))
        if raw["no_penetration_max_abs"] != residual_max:
            raise ValueError(
                "raw-step no-penetration maximum differs from trajectory evidence"
            )
        layer_loads = loads_by_key[key]
        _validate_load_block(raw, layer_loads)
        total = layer_loads[-1]
        expected_force = np.asarray(
            [total[f"force_{axis}_n"] for axis in "xyz"], dtype=np.float64
        )
        expected_moment = np.asarray(
            [total[f"moment_{axis}_nm"] for axis in "xyz"], dtype=np.float64
        )
        if not np.array_equal(
            decoded_arrays["force"], expected_force
        ) or not np.array_equal(decoded_arrays["moment"], expected_moment):
            raise ValueError("trajectory force/moment differs from the TOTAL load row")


def _validate_stage_sequence(
    rows: Sequence[Mapping[str, object]],
    *,
    status: str,
    terminal: Mapping[str, object] | None,
    durable_source_keys: Sequence[tuple[int, int]] = (),
    durable_layer_keys: Sequence[tuple[int, int]] = (),
    stop_code: str | None = None,
) -> None:
    _validate_completed_digest_fields(
        rows,
        tuple(field for field in STAGE_FIELDS if "sha256" in field),
        "transport-stage",
    )
    keys = tuple(_stage_key(row) for row in rows)
    if len(set(keys)) != len(keys):
        raise ValueError("transport stage keys are duplicated")
    if status == "PASS":
        if keys != EXPECTED_STAGE_KEYS or any(
            row["status"] != "completed" for row in rows
        ):
            raise ValueError(
                "PASS transport stages are not the exact graded full-row matrix"
            )
    elif status == "PREFIX":
        if terminal is not None or any(row["status"] != "completed" for row in rows):
            raise ValueError("live stage prefix contains a non-completed row")
        _assert_prefix("live transport stage", keys, EXPECTED_STAGE_KEYS)
    else:
        failed = bool(rows and rows[-1]["status"] == "failed")
        completed_rows = rows[:-1] if failed else rows
        if any(row["status"] != "completed" for row in completed_rows):
            raise ValueError("STOP stage prefix contains a non-completed row")
        _assert_prefix(
            "transport stage",
            tuple(_stage_key(row) for row in completed_rows),
            EXPECTED_STAGE_KEYS,
        )
        assert terminal is not None
        if bool(terminal["stage_began"]) != failed:
            raise ValueError("terminal stage_began does not match failed-stage row")
        if failed:
            if len(completed_rows) >= len(EXPECTED_STAGE_KEYS):
                raise ValueError("STOP cannot fail after a complete matrix")
            expected_failed = EXPECTED_STAGE_KEYS[len(completed_rows)]
            if _stage_key(rows[-1]) != expected_failed:
                raise ValueError(
                    "terminal failed stage is not the next prefix coordinate"
                )
            expected_source_step = SOURCE_STEPS[expected_failed[1] - 1]
            expected_ptera_step = PTERA_STEPS[expected_failed[1] - 1]
            coordinate_key = (
                terminal["transport_substeps"],
                terminal["layer"],
                terminal["source_step_index"],
                terminal["ptera_step_index"],
                terminal["substep"],
                terminal["stage"],
            )
            if coordinate_key != (
                expected_failed[0],
                expected_failed[1],
                expected_source_step,
                expected_ptera_step,
                expected_failed[2],
                expected_failed[3],
            ):
                raise ValueError("STOP summary and failed stage coordinates disagree")
        else:
            coordinate_key = tuple(
                terminal[name] for name in TERMINAL_COORDINATE_FIELDS[:6]
            )
            expected_coordinate = _expected_unbegun_terminal_six(
                completed_stage_count=len(completed_rows),
                layer_keys=durable_layer_keys,
                source_keys=durable_source_keys,
            )
            if coordinate_key != expected_coordinate:
                conversion_expected = None
                if (
                    stop_code == CONVERSION_STOP_CODE
                    and terminal["phase"] == CONVERSION_PHASE
                    and terminal["stage_began"] is False
                ):
                    conversion_expected = _expected_conversion_terminal_six(
                        completed_stage_count=len(completed_rows),
                        layer_keys=durable_layer_keys,
                        source_keys=durable_source_keys,
                    )
                if coordinate_key != conversion_expected:
                    raise ValueError(
                        "unbegun STOP terminal is not the exact durable-prefix coordinate"
                    )

    coefficients_a = (0.0, -5.0 / 9.0, -153.0 / 128.0)
    coefficients_b = (1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0)
    previous_by_layer: dict[tuple[int, int], str] = {}
    post_state_by_layer: dict[tuple[int, int], str] = {}
    tracer_post_by_layer: dict[tuple[int, int], str] = {}
    ptera_parent_by_layer: dict[tuple[int, int], str] = {}
    for row in rows:
        level, layer, substep, stage = _stage_key(row)
        if row["source_step_index"] != SOURCE_STEPS[layer - 1]:
            raise ValueError("stage source-step coordinate drift")
        if row["ptera_step_index"] != PTERA_STEPS[layer - 1]:
            raise ValueError("stage Ptera-step coordinate drift")
        if row["status"] == "completed":
            if row["substep_delta_time"] != graded_substep_delta_time(level, substep):
                raise ValueError("stage substep delta time drift")
            if (
                row["rk_a"] != coefficients_a[stage - 1]
                or row["rk_b"] != coefficients_b[stage - 1]
            ):
                raise ValueError("stage RK coefficient drift")
            if (
                row["direct_field_call_count"] != 2
                or row["ptera_center_call_count"] != 2
                or row["ptera_offset_call_count"] != 6
            ):
                raise ValueError("per-stage direct/FD call ledger mismatch")
            invariant_residual = _finite_float(
                row["invariant_residual_over_slog_max"],
                "stage normalized invariant residual",
            )
            h_jacobian = _finite_float(
                row["h_jacobian_frobenius"], "stage h Jacobian Frobenius"
            )
            h_convective = _finite_float(
                row["h_convective_over_sigma"], "stage h convective over sigma"
            )
            if invariant_residual < 0.0 or h_jacobian < 0.0 or h_convective < 0.0:
                raise ValueError("stage stability operands are outside their domains")
            if row["failure_type"] != "" or row["failure_message"] != "":
                raise ValueError("completed stage carries failure metadata")
            if invariant_residual > STAGE_INVARIANT_RESIDUAL_OVER_SLOG_MAX:
                raise ValueError(
                    "stage normalized invariant-residual stability gate failed"
                )
            if h_jacobian > STAGE_H_JACOBIAN_FROBENIUS_MAX:
                raise ValueError("stage h||J_total||F stability gate failed")
            if h_convective > STAGE_H_GALILEAN_OVER_SIGMA_MAX:
                raise ValueError("stage Galilean/sigma stability gate failed")
            if row["ptera_parent_sha256_before"] != row["ptera_parent_sha256_after"]:
                raise ValueError("completed stage changed its Ptera parent")
            parent_key = (level, layer)
            frozen_ptera = ptera_parent_by_layer.setdefault(
                parent_key, str(row["ptera_parent_sha256_before"])
            )
            if row["ptera_parent_sha256_before"] != frozen_ptera:
                raise ValueError("completed stage layer changed frozen Ptera parent")
            if parent_key in post_state_by_layer and (
                row["pre_state_sha256"] != post_state_by_layer[parent_key]
                or row["tracer_pre_sha256"] != tracer_post_by_layer[parent_key]
            ):
                raise ValueError("transport-stage state/tracer prefix is discontinuous")
            post_state_by_layer[parent_key] = str(row["post_state_sha256"])
            tracer_post_by_layer[parent_key] = str(row["tracer_post_sha256"])
        elif row["status"] == "failed":
            if (
                type(row["failure_type"]) is not str
                or not row["failure_type"]
                or type(row["failure_message"]) is not str
                or not row["failure_message"]
            ):
                raise ValueError("failed stage lacks failure metadata")
            for name in (
                "substep_delta_time",
                "rk_a",
                "rk_b",
                "invariant_residual_over_slog_max",
                "h_jacobian_frobenius",
                "h_convective_over_sigma",
            ):
                value = row[name]
                if value is not None:
                    observed = _finite_float(value, f"failed stage {name}")
                    if (
                        name
                        in {
                            "invariant_residual_over_slog_max",
                            "h_jacobian_frobenius",
                            "h_convective_over_sigma",
                        }
                        and observed < 0.0
                    ):
                        raise ValueError(
                            "failed stage partial stability evidence is invalid"
                        )
            for name in (
                "direct_field_call_count",
                "ptera_center_call_count",
                "ptera_offset_call_count",
            ):
                if row[name] is not None:
                    _exact_int(row[name], f"failed stage {name}")
            for name in (
                item
                for item in STAGE_FIELDS
                if "sha256" in item
                and item not in {"previous_chain_sha256", "chain_sha256"}
            ):
                if row[name] not in (None, "") and not _is_sha256(row[name]):
                    raise ValueError(f"failed stage partial digest is invalid: {name}")
            parent_key = (level, layer)
            before = row["ptera_parent_sha256_before"]
            after = row["ptera_parent_sha256_after"]
            if (before is None) != (after is None) or (
                before is not None and before != after
            ):
                raise ValueError("failed-stage partial Ptera parent is inconsistent")
            if before is not None and parent_key in ptera_parent_by_layer:
                if before != ptera_parent_by_layer[parent_key]:
                    raise ValueError("failed stage changed the frozen Ptera parent")
            if parent_key in post_state_by_layer:
                if row["pre_state_sha256"] not in (
                    None,
                    "",
                    post_state_by_layer[parent_key],
                ) or row["tracer_pre_sha256"] not in (
                    None,
                    "",
                    tracer_post_by_layer[parent_key],
                ):
                    raise ValueError(
                        "failed stage partial prefix state is inconsistent"
                    )
        else:
            raise ValueError("transport stage status is invalid")
        parent_key = (level, layer)
        expected_previous = previous_by_layer.get(
            parent_key, STREAM_STAGE_CHAIN_GENESIS
        )
        if row["previous_chain_sha256"] != expected_previous:
            raise ValueError("stage chain predecessor mismatch")
        if row["status"] == "completed":
            if not _is_sha256(row["stream_record_sha256"]):
                raise ValueError("completed stage record digest is invalid")
            expected_chain = sha256(
                (
                    "fluxv-ir-wrk3-stream-stage-link-v1\0"
                    + expected_previous
                    + str(row["stream_record_sha256"])
                ).encode("ascii")
            ).hexdigest()
            if row["chain_sha256"] != expected_chain:
                raise ValueError("stage chain link mismatch")
            previous_by_layer[parent_key] = expected_chain
        elif row["chain_sha256"] != expected_previous:
            raise ValueError("failed stage must preserve the completed chain")


def _trajectory_array(value: object) -> np.ndarray:
    return decode_array(value) if isinstance(value, Mapping) else np.asarray(value)


def _require_invariant_replay(
    actual: np.ndarray, expected: np.ndarray, *, name: str
) -> None:
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        raise ValueError(f"trajectory {name} is not replayable")
    epsilon_factor = 64.0 * np.finfo(np.float64).eps
    for observed, reference in zip(
        actual.reshape(-1), expected.reshape(-1), strict=True
    ):
        try:
            residual = math.fsum((float(observed), -float(reference)))
        except OverflowError as error:
            raise ValueError(f"trajectory {name} is not replayable") from error
        tolerance = epsilon_factor * max(
            1.0, abs(float(observed)), abs(float(reference))
        )
        if not math.isfinite(residual) or abs(residual) > tolerance:
            raise ValueError(f"trajectory {name} is not replayable")


def _validate_macro_invariant_domain(
    *,
    start_gamma: np.ndarray,
    start_sigma: np.ndarray,
    end_gamma: np.ndarray,
    end_sigma: np.ndarray,
) -> None:
    """Replay the B1 exact-zero/active split and macro invariant gate."""

    start_max_abs = np.max(np.abs(start_gamma), axis=1)
    end_max_abs = np.max(np.abs(end_gamma), axis=1)
    start_zero = start_max_abs == 0.0
    end_zero = end_max_abs == 0.0
    if not np.array_equal(start_zero, end_zero):
        raise ValueError("trajectory Gamma exact-zero classification changed")
    active = ~start_zero
    if np.any(start_max_abs[active] <= ACTIVE_GAMMA_MAXABS_MIN) or np.any(
        end_max_abs[active] <= ACTIVE_GAMMA_MAXABS_MIN
    ):
        raise ValueError("trajectory active Gamma is at the frozen B1 threshold")
    if np.any(start_zero):
        start_zero_gamma = np.ascontiguousarray(start_gamma[start_zero])
        end_zero_gamma = np.ascontiguousarray(end_gamma[start_zero])
        start_zero_sigma = np.ascontiguousarray(start_sigma[start_zero])
        end_zero_sigma = np.ascontiguousarray(end_sigma[start_zero])
        if start_zero_gamma.tobytes(order="C") != end_zero_gamma.tobytes(
            order="C"
        ) or start_zero_sigma.tobytes(order="C") != end_zero_sigma.tobytes(order="C"):
            raise ValueError(
                "trajectory exact-zero Gamma/sigma bits changed during transport"
            )
    start_norm = _stable_row_norms(start_gamma)
    end_norm = _stable_row_norms(end_gamma)
    if (
        not np.all(np.isfinite(start_norm))
        or not np.all(np.isfinite(end_norm))
        or np.any(start_norm[active] <= 0.0)
        or np.any(end_norm[active] <= 0.0)
    ):
        raise ValueError("trajectory active Gamma norm is not finite and positive")
    if np.any(active):
        log_norm_delta = np.log(end_norm[active]) - np.log(start_norm[active])
        log_sigma_delta = np.log(end_sigma[active]) - np.log(start_sigma[active])
        residual = np.abs(log_norm_delta + 2.0 * log_sigma_delta)
        slog = np.maximum.reduce(
            (
                np.ones(np.count_nonzero(active), dtype=np.float64),
                np.abs(log_norm_delta),
                2.0 * np.abs(log_sigma_delta),
            )
        )
        _require_invariant_log_gate(residual, slog)


def _require_invariant_log_gate(residual: np.ndarray, slog: np.ndarray) -> None:
    """Apply the frozen IR-WRK3 residual <= 512*eps*Slog comparison."""

    if (
        residual.shape != slog.shape
        or not np.all(np.isfinite(residual))
        or not np.all(np.isfinite(slog))
        or np.any(residual < 0.0)
        or np.any(slog < 1.0)
    ):
        raise ValueError("trajectory macro invariant log evidence is invalid")
    tolerance = STAGE_INVARIANT_RESIDUAL_OVER_SLOG_MAX * slog
    if not np.all(np.isfinite(tolerance)) or np.any(residual > tolerance):
        raise ValueError("trajectory macro invariant log residual exceeds frozen gate")


def _validate_completed_trajectory_semantics(
    row: Mapping[str, object],
) -> dict[str, np.ndarray]:
    if row.get("status") != "completed":
        return {}
    arrays_value = row["arrays"]
    if not isinstance(arrays_value, Mapping) or set(arrays_value) != set(
        REQUIRED_TRAJECTORY_ARRAYS
    ):
        raise ValueError("completed trajectory array schema is invalid")
    arrays = {name: _trajectory_array(value) for name, value in arrays_value.items()}
    if any(
        array.dtype != np.dtype(np.float64) or not np.all(np.isfinite(array))
        for array in arrays.values()
    ):
        raise ValueError("completed trajectory arrays must be finite exact float64")
    particle_ids = tuple(row["particle_ids"])
    material_ids = tuple(row["material_tracer_ids"])
    frontier_ids = tuple(row["frontier_node_ids"])
    positions = arrays["end_positions"]
    gamma = arrays["end_gamma"]
    sigma = arrays["end_sigma"]
    start_positions = arrays["start_positions"]
    start_gamma = arrays["start_gamma"]
    start_sigma = arrays["start_sigma"]
    if (
        positions.ndim != 2
        or positions.shape != (len(particle_ids), 3)
        or gamma.shape != positions.shape
        or sigma.shape != (len(particle_ids),)
        or start_positions.shape != positions.shape
        or start_gamma.shape != gamma.shape
        or start_sigma.shape != sigma.shape
        or arrays["material_tracer_positions"].shape != (len(material_ids), 3)
        or arrays["frontier_tracer_positions"].shape != (len(frontier_ids), 3)
        or arrays["probe_velocity"].shape != (3, 3)
        or arrays["probe_jacobian"].shape != (3, 3, 3)
        or arrays["force"].shape != (3,)
        or arrays["moment"].shape != (3,)
        or arrays["invariant_start"].shape != (len(particle_ids),)
        or arrays["invariant_end"].shape != (len(particle_ids),)
        or arrays["no_penetration_residual"].shape != (16,)
        or np.any(sigma <= 0.0)
        or np.any(start_sigma <= 0.0)
    ):
        raise ValueError("completed trajectory array shapes/domains are invalid")
    id_to_index = {particle_id: index for index, particle_id in enumerate(particle_ids)}
    expected_material = positions[
        [id_to_index[particle_id] for particle_id in material_ids]
    ]
    if not np.array_equal(arrays["material_tracer_positions"], expected_material):
        raise ValueError(
            "material tracer positions differ from ID-aligned particle support"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        invariant_start = _stable_row_norms(start_gamma) * start_sigma**2
        invariant_end = _stable_row_norms(gamma) * sigma**2
    _require_invariant_replay(
        arrays["invariant_start"], invariant_start, name="invariant_start"
    )
    _require_invariant_replay(
        arrays["invariant_end"], invariant_end, name="invariant_end"
    )
    _validate_macro_invariant_domain(
        start_gamma=start_gamma,
        start_sigma=start_sigma,
        end_gamma=gamma,
        end_sigma=sigma,
    )
    if (
        float(np.max(np.abs(arrays["no_penetration_residual"]), initial=0.0))
        > NO_PENETRATION_MAX
    ):
        raise ValueError("trajectory no-penetration residual failed")
    _validate_probe_oracle(arrays)
    _validate_trajectory_metadata(row)
    return arrays


def _validate_trajectory_ids_and_counts(
    records: ArtifactRecords,
) -> None:
    raw_by_key = {_layer_key(row): row for row in records.raw_steps}
    count_by_key = {_layer_key(row): row for row in records.particle_counts}
    for trajectory in records.trajectories:
        shell = _trajectory_record_shell(trajectory)
        key = _layer_key(trajectory)
        particle_ids = tuple(shell["particle_ids"])
        material_ids = tuple(shell["material_tracer_ids"])
        frontier_ids = tuple(shell["frontier_node_ids"])
        if (
            any(
                type(item) is not str
                for item in (*particle_ids, *material_ids, *frontier_ids)
            )
            or len(set(particle_ids)) != len(particle_ids)
            or len(set(material_ids)) != len(material_ids)
            or len(set(frontier_ids)) != len(frontier_ids)
            or not set(material_ids).issubset(particle_ids)
            or frontier_ids != FRONTIER_NODE_IDS
        ):
            raise ValueError("trajectory ID schema/order/binding is invalid")
        decoded_arrays = _validate_completed_trajectory_semantics(shell)
        arrays = shell["arrays"]
        assert isinstance(arrays, Mapping)
        if shell["status"] == "completed":
            particle_positions = decoded_arrays["end_positions"]
            material_positions = decoded_arrays["material_tracer_positions"]
            frontier_positions = decoded_arrays["frontier_tracer_positions"]
            probe_velocity = decoded_arrays["probe_velocity"]
            probe_jacobian = decoded_arrays["probe_jacobian"]
            if (
                particle_positions.shape != (len(particle_ids), 3)
                or material_positions.shape != (len(material_ids), 3)
                or frontier_positions.shape != (len(frontier_ids), 3)
                or probe_velocity.shape != (3, 3)
                or probe_jacobian.shape != (3, 3, 3)
            ):
                raise ValueError("trajectory ID/probe array counts or shapes drifted")
            id_to_index = {
                particle_id: index for index, particle_id in enumerate(particle_ids)
            }
            expected_material = particle_positions[
                [id_to_index[particle_id] for particle_id in material_ids]
            ]
            if not np.array_equal(material_positions, expected_material):
                raise ValueError(
                    "material tracer positions differ from ID-aligned particle support"
                )
            raw = raw_by_key[key]
            counts = count_by_key[key]
            for row in (raw, counts):
                if (
                    row["particle_count"] != len(particle_ids)
                    or row["material_tracer_count"]
                    != len(material_ids) + len(frontier_ids)
                    or row["material_support_tracer_count"] != len(material_ids)
                    or row["frontier_node_tracer_count"] != len(frontier_ids)
                ):
                    raise ValueError("trajectory IDs differ from durable count rows")


def _validate_records(records: ArtifactRecords) -> None:
    if records.execution_mode not in EXECUTION_MODES:
        raise ValueError("artifact execution_mode is invalid")
    if records.status not in ("PASS", "STOP"):
        raise ValueError("artifact status must be PASS or STOP")
    terminal = None
    if records.status == "PASS":
        if (
            records.terminal_coordinate is not None
            or records.stop_code is not None
            or records.stop_message is not None
        ):
            raise ValueError("PASS artifact cannot have STOP metadata")
    else:
        terminal = _validate_terminal_coordinate(records.terminal_coordinate)
        if type(records.stop_code) is not str or not records.stop_code:
            raise ValueError("STOP artifact requires a stop_code")
        if type(records.stop_message) is not str:
            raise ValueError("STOP artifact requires a stop_message")

    for rows, fields, name in (
        (records.raw_steps, RAW_STEP_FIELDS, "raw_steps"),
        (records.source_events, SOURCE_EVENT_FIELDS, "source_events"),
        (records.owner_events, OWNER_EVENT_FIELDS, "owner_events"),
        (records.particle_counts, PARTICLE_COUNT_FIELDS, "particle_counts"),
        (records.raw_loads, RAW_LOAD_FIELDS, "raw_loads"),
        (records.transport_stages, STAGE_FIELDS, "transport_stages"),
    ):
        expected_fields = set(fields)
        for row in rows:
            if set(row) != expected_fields:
                raise ValueError(f"{name} row schema mismatch")

    layer_key_sets = (
        tuple(_layer_key(row) for row in records.raw_steps),
        tuple(_layer_key(row) for row in records.owner_events),
        tuple(_layer_key(row) for row in records.particle_counts),
        tuple(_layer_key(row) for row in records.trajectories),
    )
    if len(set(layer_key_sets)) != 1:
        raise ValueError("layer-keyed artifact collections disagree")
    layer_keys = layer_key_sets[0]
    source_keys = tuple(_source_key(row) for row in records.source_events)
    load_keys = tuple(_load_key(row) for row in records.raw_loads)
    if len(set(layer_keys)) != len(layer_keys):
        raise ValueError("layer keys are duplicated")
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("source keys are duplicated")
    if len(set(load_keys)) != len(load_keys):
        raise ValueError("load keys are duplicated")

    for row in records.raw_steps:
        if row["status"] != "completed":
            raise ValueError("layer row status is not completed")
        for field in (
            "particle_count",
            "material_tracer_count",
            "material_support_tracer_count",
            "frontier_node_tracer_count",
        ):
            _exact_int(row[field], f"raw-step {field}")
        layer = int(row["layer"])
        if (
            row["source_step_index"] != SOURCE_STEPS[layer - 1]
            or row["ptera_step_index"] != PTERA_STEPS[layer - 1]
        ):
            raise ValueError("raw-step source/Ptera coordinates drifted")
        _validate_pass_counters(row)
    for row in records.source_events:
        if (
            row["status"] != "completed"
            or row["cell_count"] != 8
            or row["ptera_step_index"] != row["source_step_index"] - 1
        ):
            raise ValueError("source-event aggregate/status is invalid")
    for row in records.owner_events:
        layer = int(row["layer"])
        if (
            row["status"] != "completed"
            or row["source_step_index"] != SOURCE_STEPS[layer - 1]
            or row["ptera_step_index"] != PTERA_STEPS[layer - 1]
        ):
            raise ValueError("owner-event status/coordinates are invalid")
    for row in records.particle_counts:
        if row["status"] != "completed":
            raise ValueError("particle-count status is not completed")
        for field in PARTICLE_COUNT_FIELDS:
            if field not in {"transport_substeps", "layer", "status"}:
                _exact_int(row[field], f"particle-count {field}")
    if any(row["status"] != "completed" for row in records.trajectories):
        raise ValueError("trajectory status is not completed")
    _validate_trajectory_ids_and_counts(records)

    completed_stage_keys = {
        _stage_key(row)
        for row in records.transport_stages
        if row["status"] == "completed"
    }
    if any(
        (level, SOURCE_STEPS[layer - 1]) not in set(source_keys)
        for level, layer, _, _ in (_stage_key(row) for row in records.transport_stages)
    ):
        raise ValueError("transport stage lacks its durable source-event parent")
    for level, layer in layer_keys:
        required_stages = {
            (level, layer, substep, stage)
            for substep in range(1, level + 1)
            for stage in (1, 2, 3)
        }
        if not required_stages.issubset(completed_stage_keys):
            raise ValueError("layer row committed before all 3N stages completed")
    required_load_keys = EXPECTED_LOAD_KEYS[: 17 * len(layer_keys)]
    if load_keys != required_load_keys:
        raise ValueError("layer rows do not have exact complete 17-row load blocks")

    if records.status == "PASS":
        if layer_keys != EXPECTED_LAYER_KEYS:
            raise ValueError("PASS layer collections do not have exactly nine keys")
        if source_keys != EXPECTED_SOURCE_KEYS:
            raise ValueError("PASS source collection does not have exactly nine keys")
        if load_keys != EXPECTED_LOAD_KEYS:
            raise ValueError("PASS load collection does not have exactly 153 keys")
    else:
        _assert_prefix("layer", layer_keys, EXPECTED_LAYER_KEYS)
        _assert_prefix("source", source_keys, EXPECTED_SOURCE_KEYS)
        _assert_prefix("load", load_keys, EXPECTED_LOAD_KEYS)

    _validate_cross_table_progress(layer_keys, source_keys, records.transport_stages)
    _validate_stage_sequence(
        records.transport_stages,
        status=records.status,
        terminal=terminal,
        durable_source_keys=source_keys,
        durable_layer_keys=layer_keys,
        stop_code=records.stop_code,
    )
    _validate_completed_crosslinks(
        execution_mode=records.execution_mode,
        raw_rows=records.raw_steps,
        source_rows=records.source_events,
        owner_rows=records.owner_events,
        particle_rows=records.particle_counts,
        load_rows=records.raw_loads,
        stage_rows=records.transport_stages,
        trajectory_rows=records.trajectories,
    )


def _trajectory_document(records: ArtifactRecords) -> dict[str, object]:
    encoded_records: list[dict[str, object]] = []
    for raw in records.trajectories:
        if set(raw) != {
            "transport_substeps",
            "layer",
            "status",
            "particle_ids",
            "material_tracer_ids",
            "frontier_node_ids",
            "arrays",
            "metadata",
        }:
            raise ValueError("trajectory record schema mismatch")
        arrays = raw["arrays"]
        if not isinstance(arrays, Mapping):
            raise ValueError("trajectory arrays must be a mapping")
        if raw["status"] == "completed" and set(arrays) != set(
            REQUIRED_TRAJECTORY_ARRAYS
        ):
            raise ValueError("completed trajectory does not have the exact array set")
        if not set(arrays).issubset(REQUIRED_TRAJECTORY_ARRAYS):
            raise ValueError("trajectory contains a foreign array channel")
        encoded_arrays: dict[str, object] = {}
        for name in sorted(arrays):
            value = arrays[name]
            if isinstance(value, Mapping):
                decoded = decode_array(value)
                encoded_arrays[name] = dict(value)
            else:
                decoded = np.asarray(value)
                encoded_arrays[name] = encode_array(value)
            if raw["status"] == "completed" and decoded.dtype != np.dtype(np.float64):
                raise ValueError(
                    f"completed trajectory channel {name} must have exact float64 dtype"
                )
        encoded_records.append(
            {
                "arrays": encoded_arrays,
                "frontier_node_ids": _id_sequence(raw["frontier_node_ids"]),
                "layer": raw["layer"],
                "material_tracer_ids": _id_sequence(raw["material_tracer_ids"]),
                "metadata": dict(raw["metadata"]),
                "particle_ids": _id_sequence(raw["particle_ids"]),
                "status": raw["status"],
                "transport_substeps": raw["transport_substeps"],
            }
        )
    return {
        "case_id": CASE_ID,
        "execution_mode": records.execution_mode,
        "fixed_probe_source_sha256": dict(FIXED_PROBE_SOURCE_SHA256),
        "fixed_probes_gp1_m": [list(point) for point in FIXED_PROBES_GP1_M],
        "observation_access": OBSERVATION_ACCESS,
        "records": encoded_records,
        "schema_id": TRAJECTORY_SCHEMA_BY_MODE[records.execution_mode],
        "status": records.status,
    }


def _stable_row_norms(gamma: np.ndarray) -> np.ndarray:
    maximum = np.max(np.abs(gamma), axis=1)
    result = np.zeros(gamma.shape[0], dtype=np.float64)
    active = maximum != 0.0
    if np.any(active):
        scaled = gamma[active] / maximum[active, None]
        result[active] = maximum[active] * np.sqrt(
            np.einsum("ni,ni->n", scaled, scaled)
        )
    return result


def _stream_state_sha256_from_arrays(
    positions: np.ndarray, gamma: np.ndarray, sigma: np.ndarray
) -> str:
    """Replay the public IR-WRK3 particle-state digest from durable arrays."""

    return sha256(
        (
            "fluxv-ir-wrk3-stream-state-v1\0"
            + _array_sha256(positions)
            + _array_sha256(gamma)
            + _array_sha256(sigma)
        ).encode("ascii")
    ).hexdigest()


def _direct_gaussian_erf_probe_oracle(
    positions: np.ndarray, gamma: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pure replay of the frozen Gaussian-erf U/J formula at fixed probes.

    This intentionally uses only Python/NumPy and the scalar :func:`math.erf`;
    it does not import the FluxVortex package or any solver implementation.
    The source-outer/probe-inner accumulation order matches the frozen direct
    reference.
    """

    positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
    gamma = np.ascontiguousarray(np.asarray(gamma, dtype=np.float64))
    sigma = np.ascontiguousarray(np.asarray(sigma, dtype=np.float64))
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or gamma.shape != positions.shape
        or sigma.shape != (positions.shape[0],)
        or np.any(sigma <= 0.0)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(gamma))
        or not np.all(np.isfinite(sigma))
    ):
        raise ValueError("probe oracle particle state is invalid")
    targets = np.asarray(FIXED_PROBES_GP1_M, dtype=np.float64)
    velocity = np.zeros((3, 3), dtype=np.float64)
    jacobian = np.zeros((3, 3, 3), dtype=np.float64)
    minus_one_over_four_pi = -1.0 / (4.0 * math.pi)
    sqrt_two_over_pi = math.sqrt(2.0 / math.pi)
    sqrt_two = math.sqrt(2.0)
    for source_index in range(positions.shape[0]):
        for target_index in range(targets.shape[0]):
            dx = targets[target_index] - positions[source_index]
            r2 = float(np.dot(dx, dx))
            if r2 == 0.0:
                continue
            radius = math.sqrt(r2)
            radius_over_sigma = radius / float(sigma[source_index])
            exponential = math.exp(-0.5 * radius_over_sigma**2)
            auxiliary = sqrt_two_over_pi * radius_over_sigma * exponential
            regularization = math.erf(radius_over_sigma / sqrt_two) - auxiliary
            regularization_derivative = radius_over_sigma * auxiliary
            radius_cubed_inverse = 1.0 / (r2 * radius)
            cross = np.cross(dx, gamma[source_index])
            kernel_cross_gamma = minus_one_over_four_pi * radius_cubed_inverse * cross
            velocity[target_index] += regularization * kernel_cross_gamma
            gradient_radial = (
                regularization_derivative / (float(sigma[source_index]) * radius)
                - 3.0 * regularization / r2
            )
            kronecker = minus_one_over_four_pi * regularization * radius_cubed_inverse
            source_strength = gamma[source_index]
            contribution = np.empty((3, 3), dtype=np.float64)
            contribution[0, 0] = gradient_radial * kernel_cross_gamma[0] * dx[0]
            contribution[1, 0] = (
                gradient_radial * kernel_cross_gamma[1] * dx[0]
                - kronecker * source_strength[2]
            )
            contribution[2, 0] = (
                gradient_radial * kernel_cross_gamma[2] * dx[0]
                + kronecker * source_strength[1]
            )
            contribution[0, 1] = (
                gradient_radial * kernel_cross_gamma[0] * dx[1]
                + kronecker * source_strength[2]
            )
            contribution[1, 1] = gradient_radial * kernel_cross_gamma[1] * dx[1]
            contribution[2, 1] = (
                gradient_radial * kernel_cross_gamma[2] * dx[1]
                - kronecker * source_strength[0]
            )
            contribution[0, 2] = (
                gradient_radial * kernel_cross_gamma[0] * dx[2]
                - kronecker * source_strength[1]
            )
            contribution[1, 2] = (
                gradient_radial * kernel_cross_gamma[1] * dx[2]
                + kronecker * source_strength[0]
            )
            contribution[2, 2] = gradient_radial * kernel_cross_gamma[2] * dx[2]
            jacobian[target_index] += contribution
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(jacobian)):
        raise ValueError("probe oracle produced non-finite evidence")
    return velocity, jacobian


def _validate_probe_oracle(arrays: Mapping[str, np.ndarray]) -> None:
    expected_velocity, expected_jacobian = _direct_gaussian_erf_probe_oracle(
        arrays["end_positions"], arrays["end_gamma"], arrays["end_sigma"]
    )
    for name, expected in (
        ("probe_velocity", expected_velocity),
        ("probe_jacobian", expected_jacobian),
    ):
        actual = arrays[name]
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=PROBE_DIRECT_ORACLE_RTOL,
            atol=PROBE_DIRECT_ORACLE_ATOL,
        ):
            raise ValueError(f"trajectory {name} differs from direct replay oracle")


def _decoded_trajectory_records(path: Path) -> list[dict[str, object]]:
    document = _load_json(path)
    if not isinstance(document, Mapping) or set(document) != {
        "case_id",
        "execution_mode",
        "fixed_probe_source_sha256",
        "fixed_probes_gp1_m",
        "observation_access",
        "records",
        "schema_id",
        "status",
    }:
        raise ValueError("trajectory document schema is invalid")
    execution_mode = document["execution_mode"]
    if (
        execution_mode not in EXECUTION_MODES
        or document["schema_id"] != TRAJECTORY_SCHEMA_BY_MODE[execution_mode]
        or document["case_id"] != CASE_ID
        or document["fixed_probe_source_sha256"] != dict(FIXED_PROBE_SOURCE_SHA256)
        or document["fixed_probes_gp1_m"]
        != [list(point) for point in FIXED_PROBES_GP1_M]
        or document["observation_access"] != OBSERVATION_ACCESS
        or document["status"] not in ("PASS", "STOP")
        or type(document["records"]) is not list
    ):
        raise ValueError("trajectory document contract drift")
    decoded: list[dict[str, object]] = []
    for raw in document["records"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "arrays",
            "frontier_node_ids",
            "layer",
            "material_tracer_ids",
            "metadata",
            "particle_ids",
            "status",
            "transport_substeps",
        }:
            raise ValueError("serialized trajectory record schema is invalid")
        if not isinstance(raw["arrays"], Mapping) or not isinstance(
            raw["metadata"], Mapping
        ):
            raise ValueError("serialized trajectory mappings are invalid")
        arrays = {name: decode_array(value) for name, value in raw["arrays"].items()}
        ids = _decode_id_sequence(raw["particle_ids"])
        material_ids = _decode_id_sequence(raw["material_tracer_ids"])
        frontier_ids = _decode_id_sequence(raw["frontier_node_ids"])
        if (
            len(set(ids)) != len(ids)
            or len(set(material_ids)) != len(material_ids)
            or len(set(frontier_ids)) != len(frontier_ids)
        ):
            raise ValueError("trajectory ID sequences contain duplicates")
        status = raw["status"]
        if status == "completed":
            if set(arrays) != set(REQUIRED_TRAJECTORY_ARRAYS):
                raise ValueError("completed serialized trajectory is incomplete")
            if any(array.dtype != np.dtype(np.float64) for array in arrays.values()):
                raise ValueError(
                    "completed trajectory channels must have exact float64 dtype"
                )
            positions = arrays["end_positions"]
            gamma = arrays["end_gamma"]
            sigma = arrays["end_sigma"]
            start_positions = arrays["start_positions"]
            start_gamma = arrays["start_gamma"]
            start_sigma = arrays["start_sigma"]
            if (
                positions.ndim != 2
                or positions.shape[1:] != (3,)
                or gamma.shape != positions.shape
                or sigma.shape != (positions.shape[0],)
                or start_positions.shape != positions.shape
                or start_gamma.shape != gamma.shape
                or start_sigma.shape != sigma.shape
                or len(ids) != positions.shape[0]
                or not set(material_ids).issubset(ids)
                or arrays["material_tracer_positions"].shape != (len(material_ids), 3)
                or frontier_ids != FRONTIER_NODE_IDS
                or arrays["frontier_tracer_positions"].shape != (len(frontier_ids), 3)
                or arrays["probe_velocity"].shape != (3, 3)
                or arrays["probe_jacobian"].shape != (3, 3, 3)
                or np.any(sigma <= 0.0)
                or np.any(start_sigma <= 0.0)
            ):
                raise ValueError("trajectory particle arrays are inconsistent")
            id_to_index = {particle_id: index for index, particle_id in enumerate(ids)}
            expected_material = positions[
                [id_to_index[particle_id] for particle_id in material_ids]
            ]
            if not np.array_equal(
                arrays["material_tracer_positions"], expected_material
            ):
                raise ValueError(
                    "material tracer positions differ from ID-aligned particle support"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                invariant_start = _stable_row_norms(start_gamma) * start_sigma**2
                invariant_end = _stable_row_norms(gamma) * sigma**2
            for name, actual, expected in (
                ("invariant_start", arrays["invariant_start"], invariant_start),
                ("invariant_end", arrays["invariant_end"], invariant_end),
            ):
                _require_invariant_replay(actual, expected, name=name)
            residual = arrays["no_penetration_residual"]
            if (
                residual.shape != (16,)
                or float(np.max(np.abs(residual), initial=0.0)) > NO_PENETRATION_MAX
            ):
                raise ValueError("trajectory no-penetration residual failed")
            _validate_probe_oracle(arrays)
            _validate_trajectory_metadata(
                {
                    "layer": raw["layer"],
                    "metadata": raw["metadata"],
                    "particle_ids": ids,
                    "material_tracer_ids": material_ids,
                }
            )
        decoded_row: dict[str, object] = {
            "arrays": arrays,
            "frontier_node_ids": frontier_ids,
            "layer": _exact_int(raw["layer"], "trajectory.layer", 1),
            "material_tracer_ids": material_ids,
            "metadata": dict(raw["metadata"]),
            "particle_ids": ids,
            "status": status,
            "transport_substeps": _exact_int(
                raw["transport_substeps"], "trajectory.transport_substeps", 1
            ),
        }
        # Use the same scientific replay gate at the producer boundary and
        # after reopening durable bytes.  The checks above retain targeted
        # serialized-format diagnostics; this call prevents either path from
        # silently acquiring a weaker semantic contract.
        _validate_completed_trajectory_semantics(decoded_row)
        decoded.append(decoded_row)
    return decoded


def _l2_metrics(
    coarse: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    relative_max: float,
) -> dict[str, object]:
    if coarse.shape != candidate.shape or candidate.shape != reference.shape:
        raise ValueError("convergence arrays are not shape-aligned")

    def stable_norm(value: np.ndarray) -> float | None:
        flat = value.reshape(-1)
        scale = float(np.max(np.abs(flat), initial=0.0))
        if scale == 0.0:
            return 0.0
        scaled_norm = float(np.linalg.norm(flat / scale))
        if not math.isfinite(scaled_norm) or (
            scaled_norm > 1.0 and scale > np.finfo(np.float64).max / scaled_norm
        ):
            return None
        return float(scale * scaled_norm)

    def stable_difference(left: np.ndarray, right: np.ndarray) -> float | None:
        scale = max(
            float(np.max(np.abs(left), initial=0.0)),
            float(np.max(np.abs(right), initial=0.0)),
        )
        if scale == 0.0:
            return 0.0
        normalized = left / scale - right / scale
        scaled_norm = float(np.linalg.norm(normalized.reshape(-1)))
        if not math.isfinite(scaled_norm) or (
            scaled_norm > 1.0 and scale > np.finfo(np.float64).max / scaled_norm
        ):
            return None
        return float(scale * scaled_norm)

    d32_64 = stable_difference(coarse, candidate)
    d64_128 = stable_difference(candidate, reference)
    reference_norm = stable_norm(reference)
    if d32_64 is None or d64_128 is None or reference_norm is None:
        return {
            "d32_64": d32_64,
            "d64_128": d64_128,
            "failure_reason": "unrepresentable_l2_magnitude",
            "passed": False,
            "ratio": None,
            "ratio_exempt_roundoff": False,
            "ratio_min": MIN_DIFFERENCE_REDUCTION_RATIO,
            "relative_64_128": None,
            "relative_max": relative_max,
        }
    denominator = max(1.0e-15, reference_norm)
    relative = d64_128 / denominator
    if not math.isfinite(relative):
        return {
            "d32_64": d32_64,
            "d64_128": d64_128,
            "failure_reason": "unrepresentable_relative_magnitude",
            "passed": False,
            "ratio": None,
            "ratio_exempt_roundoff": False,
            "ratio_min": MIN_DIFFERENCE_REDUCTION_RATIO,
            "relative_64_128": None,
            "relative_max": relative_max,
        }
    exempt = d32_64 <= ROUNDOFF_DIFFERENCE_MAX and d64_128 <= ROUNDOFF_DIFFERENCE_MAX
    if d64_128 == 0.0:
        ratio: float | str = "inf"
        ratio_passed = True
    else:
        ratio = d32_64 / d64_128
        if math.isfinite(ratio):
            ratio_passed = ratio >= MIN_DIFFERENCE_REDUCTION_RATIO
        else:
            ratio = "inf"
            ratio_passed = True
    passed = relative <= relative_max and (exempt or ratio_passed)
    return {
        "d32_64": d32_64,
        "d64_128": d64_128,
        "failure_reason": None,
        "passed": passed,
        "ratio": ratio,
        "ratio_exempt_roundoff": exempt,
        "ratio_min": MIN_DIFFERENCE_REDUCTION_RATIO,
        "relative_64_128": relative,
        "relative_max": relative_max,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    expected_by_name = {
        "raw_steps.csv": RAW_STEP_FIELDS,
        "source_events.csv": SOURCE_EVENT_FIELDS,
        "particle_counts.csv": PARTICLE_COUNT_FIELDS,
        "raw_loads.csv": RAW_LOAD_FIELDS,
        "transport_stages.csv": STAGE_FIELDS,
    }
    expected = expected_by_name.get(path.name)
    if expected is None:
        raise ValueError(f"no frozen CSV schema for {path.name}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows or tuple(rows[0]) != expected or len(set(rows[0])) != len(rows[0]):
        raise ValueError(f"{path.name} header/schema is invalid")
    decoded: list[dict[str, str]] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(expected):
            raise ValueError(f"{path.name} row {row_number} width is invalid")
        decoded.append(dict(zip(expected, row, strict=True)))
    return decoded


def _validate_and_index_loads(
    directory: Path,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    rows = _read_csv(directory / "raw_loads.csv")
    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (int(row["transport_substeps"]), int(row["layer"]))
        groups.setdefault(key, []).append(row)
    totals: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, group in groups.items():
        if len(group) != 17:
            # An incomplete STOP may have a true prefix of a load block.
            continue
        panel = group[:16]
        total = group[16]
        if (
            tuple(row["panel_id"] for row in panel) != PANEL_IDS
            or any(row["scope"] != "panel" for row in panel)
            or total["scope"] != "total"
            or total["panel_id"] != TOTAL_PANEL_ID
        ):
            raise ValueError("raw load panel topology/order is invalid")
        panel_forces = np.asarray(
            [[float(row[f"force_{axis}_n"]) for axis in "xyz"] for row in panel],
            dtype=np.float64,
        )
        panel_moments = np.asarray(
            [[float(row[f"moment_{axis}_nm"]) for axis in "xyz"] for row in panel],
            dtype=np.float64,
        )
        total_force = np.asarray(
            [float(total[f"force_{axis}_n"]) for axis in "xyz"], dtype=np.float64
        )
        total_moment = np.asarray(
            [float(total[f"moment_{axis}_nm"]) for axis in "xyz"], dtype=np.float64
        )
        _require_panel_total_match(panel_forces, total_force, label="force")
        _require_panel_total_match(panel_moments, total_moment, label="moment")
        totals[key] = (total_force, total_moment)
    return totals


def recompute_convergence_from_artifacts(directory: Path) -> dict[str, object]:
    """Reopen durable bytes and independently recompute every B3 comparison."""

    directory = Path(directory)
    trajectory_document = _load_json(directory / "trajectory_arrays.json")
    if not isinstance(trajectory_document, Mapping):
        raise ValueError("trajectory document must be an object")
    execution_mode = trajectory_document.get("execution_mode")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("trajectory execution_mode is invalid")
    artifact_status = trajectory_document.get("status")
    if artifact_status not in ("PASS", "STOP"):
        raise ValueError("trajectory artifact status is invalid")
    records = _decoded_trajectory_records(directory / "trajectory_arrays.json")
    load_totals = _validate_and_index_loads(directory)
    completed = {
        (record["transport_substeps"], record["layer"]): record
        for record in records
        if record["status"] == "completed"
    }
    metrics: list[dict[str, object]] = []
    state_channels = (
        ("positions", "end_positions", STATE_RELATIVE_L2_MAX),
        ("gamma", "end_gamma", STATE_RELATIVE_L2_MAX),
        ("sigma", "end_sigma", STATE_RELATIVE_L2_MAX),
        ("material_tracer", "material_tracer_positions", STATE_RELATIVE_L2_MAX),
        ("frontier_tracer", "frontier_tracer_positions", STATE_RELATIVE_L2_MAX),
        ("probe_velocity", "probe_velocity", PROBE_RELATIVE_L2_MAX),
        ("probe_jacobian", "probe_jacobian", PROBE_RELATIVE_L2_MAX),
    )
    for layer in LAYERS:
        triplet = [completed.get((level, layer)) for level in FORMAL_LEVELS]
        if any(record is None for record in triplet):
            continue
        coarse, candidate, reference = triplet
        assert coarse is not None and candidate is not None and reference is not None
        if not (
            coarse["particle_ids"]
            == candidate["particle_ids"]
            == reference["particle_ids"]
            and coarse["material_tracer_ids"]
            == candidate["material_tracer_ids"]
            == reference["material_tracer_ids"]
            and coarse["frontier_node_ids"]
            == candidate["frontier_node_ids"]
            == reference["frontier_node_ids"]
        ):
            raise ValueError("trajectory IDs are not exactly aligned across levels")
        for channel, array_name, threshold in state_channels:
            values = [record["arrays"][array_name] for record in triplet]
            metric = _l2_metrics(*values, relative_max=threshold)
            metrics.append({"channel": channel, "scope": f"layer_{layer}", **metric})
        for record in triplet:
            key = (record["transport_substeps"], layer)
            if key not in load_totals:
                raise ValueError("trajectory lacks a complete durable load block")
            force, moment = load_totals[key]
            if not np.array_equal(record["arrays"]["force"], force):
                raise ValueError("trajectory force differs from durable total load")
            if not np.array_equal(record["arrays"]["moment"], moment):
                raise ValueError("trajectory moment differs from durable total load")

    for layer in (2, 3):
        triplet = [completed.get((level, layer)) for level in FORMAL_LEVELS]
        if any(record is None for record in triplet):
            continue
        for channel in ("force", "moment"):
            values = [record["arrays"][channel] for record in triplet]  # type: ignore[index]
            metrics.append(
                {
                    "channel": channel,
                    "scope": f"layer_{layer}",
                    **_l2_metrics(*values, relative_max=LOAD_RELATIVE_L2_MAX),
                }
            )

    if all(key in completed for key in EXPECTED_LAYER_KEYS):
        for channel in ("force", "moment"):
            values = [
                np.concatenate(
                    [completed[(level, layer)]["arrays"][channel] for layer in LAYERS]
                )
                for level in FORMAL_LEVELS
            ]
            metrics.append(
                {
                    "channel": channel,
                    "scope": "stacked_layers_1_2_3",
                    **_l2_metrics(*values, relative_max=LOAD_RELATIVE_L2_MAX),
                }
            )

    complete = len(completed) == len(EXPECTED_LAYER_KEYS)
    expected_metric_count = 27
    passed = (
        complete
        and len(metrics) == expected_metric_count
        and all(bool(metric["passed"]) for metric in metrics)
    )
    input_hashes = {
        name: _sha256_file(directory / name) for name in RAW_REPLAY_INPUT_FILES
    }
    return {
        "case_id": CASE_ID,
        "complete_matrix": complete,
        "execution_mode": execution_mode,
        "input_file_sha256": input_hashes,
        "metric_count": len(metrics),
        "metrics": metrics,
        "observation_access": OBSERVATION_ACCESS,
        "passed": passed,
        "schema_id": CONVERGENCE_SCHEMA_BY_MODE[execution_mode],
        "status": artifact_status,
    }


def _passing_convergence_stop_is_allowed(
    *,
    status: object,
    stop_code: object,
    terminal_coordinate: object,
    row_counts: object,
) -> bool:
    if (
        status != "STOP"
        or type(stop_code) is not str
        or not isinstance(terminal_coordinate, Mapping)
        or not isinstance(row_counts, Mapping)
    ):
        return False
    expected_counts = {
        "owner_events": 9,
        "particle_counts": 9,
        "raw_loads": 153,
        "raw_steps": 9,
        "source_events": 9,
        "trajectories": 9,
        "transport_stages": len(EXPECTED_STAGE_KEYS),
    }
    expected_coordinate = {
        "transport_substeps": FORMAL_LEVELS[-1],
        "layer": 3,
        "source_step_index": 6,
        "ptera_step_index": 5,
        "substep": FORMAL_LEVELS[-1],
        "stage": 3,
        "stage_began": False,
    }
    phase = terminal_coordinate.get("phase")
    observed_coordinate = {
        name: terminal_coordinate.get(name) for name in expected_coordinate
    }
    return (
        dict(row_counts) == expected_counts
        and observed_coordinate == expected_coordinate
        and (stop_code, phase) in POST_MATRIX_PASSING_CONVERGENCE_STOP_PAIRS
    )


def _semantic_root(file_sha256: Mapping[str, str]) -> str:
    if tuple(file_sha256) != SEMANTIC_FILES:
        raise ValueError("semantic file map order/scope is invalid")
    return _payload_sha256(
        "fluxv-v5h13-baik-w2-semantic-result-v1",
        [{"name": name, "sha256": file_sha256[name]} for name in SEMANTIC_FILES],
    )


def _run_provenance(
    output_dir: Path,
    replicate: str,
    invocation_argv: Sequence[str] | None,
) -> dict[str, object]:
    return {
        "argv": list(sys.argv if invocation_argv is None else invocation_argv),
        "cwd": os.getcwd(),
        "output_path": str(output_dir),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "replicate": replicate,
        "run_uuid": str(uuid4()),
        "start_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_file_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_staging_file_fsync(path: Path, payload: bytes) -> None:
    """Atomically replace one private staging file and fsync its directory."""

    temporary = path.with_name(f".{path.name}.replacement-{uuid4().hex}")
    _write_file_fsync(temporary, payload)
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _convergence_failure_stop_records(records: ArtifactRecords) -> ArtifactRecords:
    """Downgrade a complete producer matrix after independent replay fails."""

    if (
        tuple(_stage_key(row) for row in records.transport_stages)
        != EXPECTED_STAGE_KEYS
    ):
        raise ValueError("convergence replay failed before a complete stage matrix")
    final_stage = records.transport_stages[-1]
    terminal = {
        "transport_substeps": final_stage["transport_substeps"],
        "layer": final_stage["layer"],
        "source_step_index": final_stage["source_step_index"],
        "ptera_step_index": final_stage["ptera_step_index"],
        "substep": final_stage["substep"],
        "stage": final_stage["stage"],
        "phase": "convergence_replay",
        "stage_began": False,
    }
    return replace(
        records,
        status="STOP",
        terminal_coordinate=terminal,
        stop_code="convergence_gate_failed",
        stop_message="independent durable replay rejected the completed matrix",
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a directory using Linux RENAME_NOREPLACE."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publish unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(staging),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            f"refusing to overwrite existing output: {destination}",
            destination,
        )
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("atomic no-replace directory publish unavailable")
    raise OSError(error_number, os.strerror(error_number), destination)


def _parse_csv_keys(
    path: Path, names: tuple[str, ...]
) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(row[name] for name in names) for row in _read_csv(path))


def _csv_int(value: str, name: str, minimum: int = 0) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} is not an integer") from error
    if str(result) != value or result < minimum:
        raise ValueError(f"{name} is not a canonical integer")
    return result


def _csv_float(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{name} is not a float") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _typed_stage_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    integer_fields = {
        "transport_substeps",
        "layer",
        "source_step_index",
        "ptera_step_index",
        "substep",
        "stage",
        "direct_field_call_count",
        "ptera_center_call_count",
        "ptera_offset_call_count",
    }
    float_fields = {
        "substep_delta_time",
        "rk_a",
        "rk_b",
        "invariant_residual_over_slog_max",
        "h_jacobian_frobenius",
        "h_convective_over_sigma",
    }
    typed: list[dict[str, object]] = []
    for row in rows:
        result: dict[str, object] = dict(row)
        failed = row.get("status") == "failed"
        for name in integer_fields:
            if (
                failed
                and name
                in {
                    "direct_field_call_count",
                    "ptera_center_call_count",
                    "ptera_offset_call_count",
                }
                and row[name] == ""
            ):
                result[name] = None
            else:
                result[name] = _csv_int(row[name], f"stage.{name}")
        for name in float_fields:
            if failed and row[name] == "":
                result[name] = None
            else:
                result[name] = _csv_float(row[name], f"stage.{name}")
        if failed:
            for name in STAGE_FIELDS:
                if "sha256" in name and row[name] == "":
                    result[name] = None
        typed.append(result)
    return typed


def _typed_raw_step_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    float_fields = {
        "no_penetration_max_abs",
        "kelvin_residual_max_abs",
        "raw_cl",
        "raw_cd",
    }
    string_fields = {"status"} | {name for name in RAW_STEP_FIELDS if "sha256" in name}
    typed: list[dict[str, object]] = []
    for row in rows:
        result: dict[str, object] = dict(row)
        for name in RAW_STEP_FIELDS:
            if name in float_fields:
                result[name] = _csv_float(row[name], f"raw_steps.{name}")
            elif name not in string_fields:
                result[name] = _csv_int(row[name], f"raw_steps.{name}")
        typed.append(result)
    return typed


def _typed_source_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    typed: list[dict[str, object]] = []
    integer_fields = {
        "transport_substeps",
        "source_step_index",
        "ptera_step_index",
        "cell_count",
    }
    json_fields = {"birth_modes", *SOURCE_VECTOR_FIELDS}
    for row in rows:
        result: dict[str, object] = dict(row)
        for name in integer_fields:
            result[name] = _csv_int(row[name], f"source.{name}")
        result["source_time_s"] = _csv_float(
            row["source_time_s"], "source.source_time_s"
        )
        for name in json_fields:
            result[name] = _loads_json_bytes(row[name].encode("utf-8"))
        typed.append(result)
    return typed


def _typed_particle_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    typed: list[dict[str, object]] = []
    for row in rows:
        result: dict[str, object] = dict(row)
        for name in PARTICLE_COUNT_FIELDS:
            if name != "status":
                result[name] = _csv_int(row[name], f"particle.{name}")
        typed.append(result)
    return typed


def _typed_load_rows(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    typed: list[dict[str, object]] = []
    coordinate_fields = {"transport_substeps", "layer"}
    optional_fields = {
        "force_coefficient_x",
        "force_coefficient_y",
        "force_coefficient_z",
        "raw_cl",
        "raw_cd",
    }
    for row in rows:
        result: dict[str, object] = dict(row)
        for name in coordinate_fields:
            result[name] = _csv_int(row[name], f"loads.{name}")
        for name in RAW_LOAD_FIELDS:
            if name in coordinate_fields or name in {"scope", "panel_id"}:
                continue
            if name in optional_fields and row[name] == "":
                result[name] = None
            else:
                result[name] = _csv_float(row[name], f"loads.{name}")
        typed.append(result)
    return typed


def _validate_serialized_counts(directory: Path, summary: Mapping[str, object]) -> None:
    status = summary["status"]
    raw_rows = _read_csv(directory / "raw_steps.csv")
    typed_raw_rows = _typed_raw_step_rows(raw_rows)
    layer_keys = tuple(
        (
            _csv_int(row["transport_substeps"], "raw_steps.transport_substeps", 1),
            _csv_int(row["layer"], "raw_steps.layer", 1),
        )
        for row in raw_rows
    )
    source_rows = _read_csv(directory / "source_events.csv")
    typed_source_rows = _typed_source_rows(source_rows)
    source_keys = tuple(
        (
            _csv_int(row["transport_substeps"], "source.transport_substeps", 1),
            _csv_int(row["source_step_index"], "source.source_step_index", 1),
        )
        for row in source_rows
    )
    load_rows = _read_csv(directory / "raw_loads.csv")
    typed_load_rows = _typed_load_rows(load_rows)
    load_keys = tuple(
        (
            _csv_int(row["transport_substeps"], "loads.transport_substeps", 1),
            _csv_int(row["layer"], "loads.layer", 1),
            row["scope"],
            row["panel_id"],
        )
        for row in load_rows
    )
    stage_rows = _read_csv(directory / "transport_stages.csv")
    typed_stage_rows = _typed_stage_rows(stage_rows)
    stage_keys = tuple(_stage_key(row) for row in typed_stage_rows)
    trajectory = _load_json(directory / "trajectory_arrays.json")
    if not isinstance(trajectory, Mapping):
        raise ValueError("trajectory document is not an object")
    trajectory_keys = tuple(
        (
            _exact_int(row["transport_substeps"], "trajectory.transport_substeps", 1),
            _exact_int(row["layer"], "trajectory.layer", 1),
        )
        for row in trajectory["records"]  # type: ignore[index]
    )
    owner_rows = _load_jsonl(directory / "owner_events.jsonl")
    if any(set(row) != set(OWNER_EVENT_FIELDS) for row in owner_rows):
        raise ValueError("owner_events JSONL row schema is invalid")
    owner_keys = tuple(
        (
            _exact_int(row["transport_substeps"], "owner.transport_substeps", 1),
            _exact_int(row["layer"], "owner.layer", 1),
        )
        for row in owner_rows
    )
    particle_rows = _read_csv(directory / "particle_counts.csv")
    typed_particle_rows = _typed_particle_rows(particle_rows)
    particle_keys = tuple(
        (
            _csv_int(row["transport_substeps"], "particle.transport_substeps", 1),
            _csv_int(row["layer"], "particle.layer", 1),
        )
        for row in particle_rows
    )
    if not (layer_keys == trajectory_keys == owner_keys == particle_keys):
        raise ValueError("serialized layer-keyed collections disagree")
    if any(row["status"] != "completed" for row in raw_rows):
        raise ValueError("serialized raw-step status is not completed")
    for row in typed_raw_rows:
        _validate_pass_counters(row)
        layer = int(row["layer"])
        if (
            row["source_step_index"] != SOURCE_STEPS[layer - 1]
            or row["ptera_step_index"] != PTERA_STEPS[layer - 1]
        ):
            raise ValueError("serialized raw-step source/Ptera coordinates drifted")
    if any(row["status"] != "completed" for row in source_rows):
        raise ValueError("serialized source status is not completed")
    if any(row["status"] != "completed" for row in owner_rows):
        raise ValueError("serialized owner status is not completed")
    if any(row["status"] != "completed" for row in particle_rows):
        raise ValueError("serialized particle status is not completed")
    trajectory_rows = trajectory["records"]
    if type(trajectory_rows) is not list or any(
        not isinstance(row, Mapping) or row.get("status") != "completed"
        for row in trajectory_rows
    ):
        raise ValueError("serialized trajectory status is not completed")
    decoded_trajectories = _decoded_trajectory_records(
        directory / "trajectory_arrays.json"
    )
    typed_raw_by_key = {_layer_key(row): row for row in typed_raw_rows}
    particle_by_key = {_layer_key(row): row for row in typed_particle_rows}
    for trajectory_row in decoded_trajectories:
        key = (
            int(trajectory_row["transport_substeps"]),
            int(trajectory_row["layer"]),
        )
        raw = typed_raw_by_key[key]
        counts = particle_by_key[key]
        expected_counts = (
            len(trajectory_row["particle_ids"]),
            len(trajectory_row["material_tracer_ids"]),
            len(trajectory_row["frontier_node_ids"]),
        )
        observed_raw = (
            raw["particle_count"],
            raw["material_support_tracer_count"],
            raw["frontier_node_tracer_count"],
        )
        observed_counts = (
            counts["particle_count"],
            counts["material_support_tracer_count"],
            counts["frontier_node_tracer_count"],
        )
        if expected_counts != observed_raw or expected_counts != observed_counts:
            raise ValueError("serialized trajectory IDs differ from durable counts")
    for row in source_rows:
        if (
            _csv_int(row["cell_count"], "source.cell_count") != 8
            or _csv_int(row["ptera_step_index"], "source.ptera_step_index", 1)
            != _csv_int(row["source_step_index"], "source.source_step_index", 1) - 1
        ):
            raise ValueError("serialized source aggregate/coordinates are invalid")

    terminal = (
        None
        if status == "PASS"
        else _validate_terminal_coordinate(summary["terminal_coordinate"])
    )
    _validate_cross_table_progress(layer_keys, source_keys, typed_stage_rows)
    _validate_stage_sequence(
        typed_stage_rows,
        status=str(status),
        terminal=terminal,
        durable_source_keys=source_keys,
        durable_layer_keys=layer_keys,
        stop_code=(None if status == "PASS" else str(summary["stop_code"])),
    )
    _validate_completed_crosslinks(
        execution_mode=str(summary["execution_mode"]),
        raw_rows=typed_raw_rows,
        source_rows=typed_source_rows,
        owner_rows=owner_rows,
        particle_rows=typed_particle_rows,
        load_rows=typed_load_rows,
        stage_rows=typed_stage_rows,
        trajectory_rows=decoded_trajectories,
    )
    completed_stage_keys = {
        _stage_key(row) for row in typed_stage_rows if row["status"] == "completed"
    }
    if any(
        (level, SOURCE_STEPS[layer - 1]) not in set(source_keys)
        for level, layer, _, _ in stage_keys
    ):
        raise ValueError("serialized transport stage lacks its source-event parent")
    for level, layer in layer_keys:
        required = {
            (level, layer, substep, stage)
            for substep in range(1, level + 1)
            for stage in (1, 2, 3)
        }
        if not required.issubset(completed_stage_keys):
            raise ValueError("serialized layer precedes completion of all 3N stages")
    if load_keys != EXPECTED_LOAD_KEYS[: 17 * len(layer_keys)]:
        raise ValueError("serialized layer lacks an exact complete 17-row load block")
    if status == "PASS":
        if (
            layer_keys != EXPECTED_LAYER_KEYS
            or source_keys != EXPECTED_SOURCE_KEYS
            or load_keys != EXPECTED_LOAD_KEYS
            or stage_keys != EXPECTED_STAGE_KEYS
        ):
            raise ValueError("serialized PASS keys/counts are not exact")
    else:
        _assert_prefix("serialized layer", layer_keys, EXPECTED_LAYER_KEYS)
        _assert_prefix("serialized source", source_keys, EXPECTED_SOURCE_KEYS)
        _assert_prefix("serialized load", load_keys, EXPECTED_LOAD_KEYS)
        failed = bool(stage_rows and stage_rows[-1]["status"] == "failed")
        completed_keys = stage_keys[:-1] if failed else stage_keys
        _assert_prefix("serialized stages", completed_keys, EXPECTED_STAGE_KEYS)
        if bool(terminal["stage_began"]) != failed:
            raise ValueError("serialized STOP stage-began contract mismatch")
        if failed and stage_keys[-1] != EXPECTED_STAGE_KEYS[len(completed_keys)]:
            raise ValueError("serialized failed stage is not next in prefix")
    counts = summary["row_counts"]
    if counts != {
        "owner_events": len(owner_keys),
        "particle_counts": len(particle_keys),
        "raw_loads": len(load_keys),
        "raw_steps": len(layer_keys),
        "source_events": len(source_keys),
        "trajectories": len(trajectory_keys),
        "transport_stages": len(stage_keys),
    }:
        raise ValueError("summary row counts do not match durable files")


def verify_artifact(directory: Path) -> dict[str, object]:
    """Verify exact files, checksums, semantic DAG, and replayed convergence."""

    directory = Path(directory)
    if set(path.name for path in directory.iterdir()) != set(ARTIFACT_FILES):
        raise ValueError("artifact does not contain exactly the frozen 12 files")
    checksum_lines = (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected_checksum_names = sorted(
        name for name in ARTIFACT_FILES if name != "SHA256SUMS"
    )
    if len(checksum_lines) != 11:
        raise ValueError("SHA256SUMS must contain exactly 11 lines")
    observed_names: list[str] = []
    for line in checksum_lines:
        digest, separator, name = line.partition("  ")
        if not separator or not _is_sha256(digest):
            raise ValueError("SHA256SUMS line is invalid")
        observed_names.append(name)
        if _sha256_file(directory / name) != digest:
            raise ValueError(f"checksum mismatch: {name}")
    if observed_names != expected_checksum_names:
        raise ValueError("SHA256SUMS names/order are invalid")

    recomputed = recompute_convergence_from_artifacts(directory)
    if (directory / "convergence.json").read_bytes() != _json_bytes(recomputed) + b"\n":
        raise ValueError("durable convergence differs from independent replay")
    summary = _load_json(directory / "summary.json")
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    expected_summary_fields = {
        "artifact_file_count",
        "candidate_score_authorized",
        "case_id",
        "convergence_passed",
        "dependency_audit_payload_sha256",
        "execution_mode",
        "fixed_probe_source_sha256",
        "fixed_probes_gp1_m",
        "force_scoring_status",
        "observation_access",
        "paper_accuracy_claim",
        "row_counts",
        "schema_id",
        "semantic_input_file_sha256",
        "status",
        "stop_code",
        "stop_message",
        "summary_payload_sha256",
        "target_read_count",
        "terminal_coordinate",
    }
    if set(summary) != expected_summary_fields:
        raise ValueError("summary schema is invalid")
    summary_without_hash = dict(summary)
    observed_summary_hash = summary_without_hash.pop("summary_payload_sha256", None)
    expected_summary_hash = _payload_sha256(
        "fluxv-v5h13-baik-w2-summary-v1", summary_without_hash
    )
    if observed_summary_hash != expected_summary_hash:
        raise ValueError("summary payload digest mismatch")
    execution_mode = summary.get("execution_mode")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("summary execution_mode is invalid")
    if (
        summary.get("schema_id") != SUMMARY_SCHEMA_BY_MODE[execution_mode]
        or summary.get("case_id") != CASE_ID
        or summary.get("observation_access") != OBSERVATION_ACCESS
        or summary.get("target_read_count") != 0
        or summary.get("candidate_score_authorized") is not False
        or summary.get("paper_accuracy_claim") is not False
        or summary.get("force_scoring_status") != FORCE_SCORING_STATUS
        or summary.get("fixed_probe_source_sha256") != dict(FIXED_PROBE_SOURCE_SHA256)
        or summary.get("fixed_probes_gp1_m")
        != [list(point) for point in FIXED_PROBES_GP1_M]
        or summary.get("artifact_file_count") != 12
        or summary.get("convergence_passed") != bool(recomputed["passed"])
        or summary.get("status") not in ("PASS", "STOP")
    ):
        raise ValueError("summary no-observation contract drift")
    expected_summary_inputs = {
        name: _sha256_file(directory / name) for name in ARTIFACT_FILES[:8]
    }
    if summary.get("semantic_input_file_sha256") != expected_summary_inputs:
        raise ValueError("summary semantic-input file map differs from durable bytes")
    trajectory_document = _load_json(directory / "trajectory_arrays.json")
    if not isinstance(trajectory_document, Mapping):
        raise ValueError("trajectory document must be an object")
    if (
        trajectory_document.get("execution_mode") != execution_mode
        or trajectory_document.get("status") != summary["status"]
        or trajectory_document.get("observation_access") != OBSERVATION_ACCESS
        or trajectory_document.get("fixed_probe_source_sha256")
        != summary["fixed_probe_source_sha256"]
        or trajectory_document.get("fixed_probes_gp1_m")
        != summary["fixed_probes_gp1_m"]
        or recomputed.get("execution_mode") != execution_mode
        or recomputed.get("status") != summary["status"]
        or recomputed.get("observation_access") != OBSERVATION_ACCESS
    ):
        raise ValueError(
            "trajectory/convergence/summary status or mode cross-link drift"
        )
    if summary["status"] == "PASS":
        if (
            summary["stop_code"] is not None
            or summary["stop_message"] is not None
            or summary["terminal_coordinate"] is not None
            or not recomputed["passed"]
        ):
            raise ValueError("PASS summary carries STOP metadata")
    elif (
        type(summary["stop_code"]) is not str
        or type(summary["stop_message"]) is not str
        or not isinstance(summary["terminal_coordinate"], Mapping)
        or (
            recomputed["passed"]
            and not _passing_convergence_stop_is_allowed(
                status=summary["status"],
                stop_code=summary["stop_code"],
                terminal_coordinate=summary["terminal_coordinate"],
                row_counts=summary["row_counts"],
            )
        )
    ):
        raise ValueError("STOP summary metadata is invalid")
    _validate_serialized_counts(directory, summary)

    manifest = _load_json(directory / "run_manifest.json")
    expected_manifest_fields = {
        "artifact_files",
        "candidate_score_authorized",
        "case_id",
        "dependency_audit",
        "dependency_audit_payload_sha256",
        "execution_mode",
        "fixed_probe_source_sha256",
        "fixed_probes_gp1_m",
        "observation_access",
        "provenance",
        "schema_id",
        "semantic_file_sha256",
        "semantic_result_sha256",
        "status",
        "target_read_count",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_manifest_fields:
        raise ValueError("run manifest schema is invalid")
    if (
        manifest.get("schema_id") != RUN_MANIFEST_SCHEMA_BY_MODE[execution_mode]
        or manifest.get("execution_mode") != execution_mode
        or manifest.get("status") != summary["status"]
        or manifest.get("case_id") != CASE_ID
        or manifest.get("artifact_files") != list(ARTIFACT_FILES)
        or manifest.get("observation_access") != OBSERVATION_ACCESS
        or manifest.get("target_read_count") != 0
        or manifest.get("candidate_score_authorized") is not False
        or manifest.get("fixed_probe_source_sha256")
        != summary["fixed_probe_source_sha256"]
        or manifest.get("fixed_probes_gp1_m") != summary["fixed_probes_gp1_m"]
    ):
        raise ValueError("run manifest status/mode/no-observation cross-link drift")
    dependency = manifest["dependency_audit"]
    if not isinstance(dependency, Mapping):
        raise ValueError("run manifest dependency evidence is invalid")
    if execution_mode == SYNTHETIC_EXECUTION_MODE:
        expected_dependency = _dependency_evidence(execution_mode, None)
    elif dependency.get("status") == "verified":
        expected_dependency = _runtime_reverify_dependency_audit(dependency)
    else:
        expected_dependency = _dependency_evidence(execution_mode, None)
    if dict(dependency) != expected_dependency:
        raise ValueError("run manifest dependency evidence is not canonical")
    if summary["status"] == "PASS" and execution_mode == FORMAL_EXECUTION_MODE:
        if dependency.get("status") != "verified":
            raise DependencyFreezeError(
                "formal PASS lacks verified dependency evidence"
            )
    dependency_digest = _payload_sha256(
        "fluxv-v5h13-dependency-evidence-v1", expected_dependency
    )
    if (
        manifest.get("dependency_audit_payload_sha256") != dependency_digest
        or summary.get("dependency_audit_payload_sha256") != dependency_digest
    ):
        raise ValueError("dependency audit digest cross-link mismatch")
    semantic_hashes = {name: _sha256_file(directory / name) for name in SEMANTIC_FILES}
    if manifest.get("semantic_file_sha256") != semantic_hashes:
        raise ValueError("run manifest semantic file map mismatch")
    semantic_root = _semantic_root(semantic_hashes)
    if manifest.get("semantic_result_sha256") != semantic_root:
        raise ValueError("run manifest semantic root mismatch")
    provenance = manifest["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "argv",
        "cwd",
        "output_path",
        "python_executable",
        "python_version",
        "replicate",
        "run_uuid",
        "start_utc",
    }:
        raise ValueError("run manifest provenance schema is invalid")
    expected_log = (
        f"status={summary['status']}\n"
        f"replicate={provenance['replicate']}\n"
        f"run_uuid={provenance['run_uuid']}\n"
        f"start_utc={provenance['start_utc']}\n"
        f"output_path={provenance['output_path']}\n"
        f"execution_mode={execution_mode}\n"
        f"dependency_audit_payload_sha256={dependency_digest}\n"
        f"semantic_result_sha256={semantic_root}\n"
        "observation_access=none\n"
        "target_read_count=0\n"
        "candidate_score_authorized=false\n"
    ).encode("utf-8")
    if (directory / "run.log").read_bytes() != expected_log:
        raise ValueError("run.log differs from manifest/summary cross-links")
    return {
        "passed": bool(summary["status"] == "PASS" and recomputed["passed"]),
        "semantic_result_sha256": semantic_root,
        "status": summary["status"],
        "summary_payload_sha256": observed_summary_hash,
    }


def _publish_artifact_core(
    records: ArtifactRecords,
    output_dir: Path,
    *,
    replicate: str,
    run_provenance: Mapping[str, object] | None = None,
    dependency_audit: Mapping[str, object] | None = None,
    formal_authorized: bool,
) -> dict[str, object]:
    """Assemble, verify, fsync, and atomically publish one immutable attempt."""

    if replicate not in ("A", "B"):
        raise ValueError("replicate must be exactly A or B")
    _validate_records(records)
    if records.execution_mode == FORMAL_EXECUTION_MODE and not formal_authorized:
        raise PermissionError(
            "formal artifacts may only be published by the audited formal entry path"
        )
    if records.execution_mode == SYNTHETIC_EXECUTION_MODE and formal_authorized:
        raise PermissionError("synthetic publication cannot use formal authorization")
    dependency_evidence = _dependency_evidence(records.execution_mode, dependency_audit)
    if (
        records.execution_mode == FORMAL_EXECUTION_MODE
        and records.status == "PASS"
        and dependency_evidence["status"] != "verified"
    ):
        raise DependencyFreezeError(
            "formal PASS requires a runtime-rehashed external dependency audit"
        )
    dependency_audit_sha256 = _payload_sha256(
        "fluxv-v5h13-dependency-evidence-v1", dependency_evidence
    )
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    provenance = _run_provenance(destination, replicate, None)
    if run_provenance is not None:
        provenance.update(run_provenance)
    if set(provenance) != {
        "argv",
        "cwd",
        "output_path",
        "python_executable",
        "python_version",
        "replicate",
        "run_uuid",
        "start_utc",
    }:
        raise ValueError("run provenance schema is invalid")
    if provenance["replicate"] != replicate or provenance["output_path"] != str(
        destination
    ):
        raise ValueError("run provenance identity does not match publication target")

    first_payloads: dict[str, bytes] = {
        "raw_steps.csv": _csv_bytes(RAW_STEP_FIELDS, records.raw_steps),
        "source_events.csv": _csv_bytes(SOURCE_EVENT_FIELDS, records.source_events),
        "owner_events.jsonl": b"".join(
            _json_bytes(dict(row)) + b"\n" for row in records.owner_events
        ),
        "particle_counts.csv": _csv_bytes(
            PARTICLE_COUNT_FIELDS, records.particle_counts
        ),
        "raw_loads.csv": _csv_bytes(RAW_LOAD_FIELDS, records.raw_loads),
        "transport_stages.csv": _csv_bytes(STAGE_FIELDS, records.transport_stages),
        "trajectory_arrays.json": _json_bytes(_trajectory_document(records)) + b"\n",
    }
    for name, payload in first_payloads.items():
        _write_file_fsync(staging / name, payload)

    convergence = recompute_convergence_from_artifacts(staging)
    if records.status == "PASS" and not convergence["passed"]:
        records = _convergence_failure_stop_records(records)
        _validate_records(records)
        _replace_staging_file_fsync(
            staging / "trajectory_arrays.json",
            _json_bytes(_trajectory_document(records)) + b"\n",
        )
        convergence = recompute_convergence_from_artifacts(staging)
        if convergence["status"] != "STOP" or convergence["passed"]:
            raise RuntimeError("convergence STOP downgrade did not replay exactly")
    if records.status == "STOP" and convergence["status"] != "STOP":
        raise ValueError("STOP producer records unexpectedly replay as PASS")
    producer_row_counts = {
        "owner_events": len(records.owner_events),
        "particle_counts": len(records.particle_counts),
        "raw_loads": len(records.raw_loads),
        "raw_steps": len(records.raw_steps),
        "source_events": len(records.source_events),
        "trajectories": len(records.trajectories),
        "transport_stages": len(records.transport_stages),
    }
    if (
        records.status == "STOP"
        and convergence["passed"]
        and not (
            _passing_convergence_stop_is_allowed(
                status=records.status,
                stop_code=records.stop_code,
                terminal_coordinate=records.terminal_coordinate,
                row_counts=producer_row_counts,
            )
        )
    ):
        raise ValueError(
            "STOP with passing convergence is not a frozen post-matrix failure"
        )
    _write_file_fsync(staging / "convergence.json", _json_bytes(convergence) + b"\n")

    semantic_inputs = {
        name: _sha256_file(staging / name) for name in ARTIFACT_FILES[:8]
    }
    summary_without_hash: dict[str, object] = {
        "artifact_file_count": 12,
        "candidate_score_authorized": False,
        "case_id": CASE_ID,
        "convergence_passed": bool(convergence["passed"]),
        "dependency_audit_payload_sha256": dependency_audit_sha256,
        "execution_mode": records.execution_mode,
        "fixed_probe_source_sha256": dict(FIXED_PROBE_SOURCE_SHA256),
        "fixed_probes_gp1_m": [list(point) for point in FIXED_PROBES_GP1_M],
        "force_scoring_status": FORCE_SCORING_STATUS,
        "observation_access": OBSERVATION_ACCESS,
        "paper_accuracy_claim": False,
        "row_counts": producer_row_counts,
        "schema_id": SUMMARY_SCHEMA_BY_MODE[records.execution_mode],
        "semantic_input_file_sha256": semantic_inputs,
        "status": records.status,
        "stop_code": records.stop_code,
        "stop_message": records.stop_message,
        "target_read_count": 0,
        "terminal_coordinate": records.terminal_coordinate,
    }
    summary = {
        **summary_without_hash,
        "summary_payload_sha256": _payload_sha256(
            "fluxv-v5h13-baik-w2-summary-v1", summary_without_hash
        ),
    }
    _write_file_fsync(staging / "summary.json", _json_bytes(summary) + b"\n")

    semantic_hashes = {name: _sha256_file(staging / name) for name in SEMANTIC_FILES}
    semantic_root = _semantic_root(semantic_hashes)
    manifest = {
        "artifact_files": list(ARTIFACT_FILES),
        "candidate_score_authorized": False,
        "case_id": CASE_ID,
        "dependency_audit": dependency_evidence,
        "dependency_audit_payload_sha256": dependency_audit_sha256,
        "execution_mode": records.execution_mode,
        "fixed_probe_source_sha256": dict(FIXED_PROBE_SOURCE_SHA256),
        "fixed_probes_gp1_m": [list(point) for point in FIXED_PROBES_GP1_M],
        "observation_access": OBSERVATION_ACCESS,
        "provenance": provenance,
        "schema_id": RUN_MANIFEST_SCHEMA_BY_MODE[records.execution_mode],
        "semantic_file_sha256": semantic_hashes,
        "semantic_result_sha256": semantic_root,
        "status": records.status,
        "target_read_count": 0,
    }
    _write_file_fsync(staging / "run_manifest.json", _json_bytes(manifest) + b"\n")
    log = (
        f"status={records.status}\n"
        f"replicate={replicate}\n"
        f"run_uuid={provenance['run_uuid']}\n"
        f"start_utc={provenance['start_utc']}\n"
        f"output_path={provenance['output_path']}\n"
        f"execution_mode={records.execution_mode}\n"
        f"dependency_audit_payload_sha256={dependency_audit_sha256}\n"
        f"semantic_result_sha256={semantic_root}\n"
        "observation_access=none\n"
        "target_read_count=0\n"
        "candidate_score_authorized=false\n"
    ).encode("utf-8")
    _write_file_fsync(staging / "run.log", log)

    checksum_names = sorted(name for name in ARTIFACT_FILES if name != "SHA256SUMS")
    checksum_payload = "".join(
        f"{_sha256_file(staging / name)}  {name}\n" for name in checksum_names
    ).encode("ascii")
    _write_file_fsync(staging / "SHA256SUMS", checksum_payload)
    if set(path.name for path in staging.iterdir()) != set(ARTIFACT_FILES):
        raise RuntimeError("staging artifact does not contain exactly 12 files")
    verification = verify_artifact(staging)
    _fsync_directory(staging)
    _publish_directory_noreplace(staging, destination)
    _fsync_directory(destination.parent)
    return verification


def compare_semantic_artifacts(left: Path, right: Path) -> str:
    """Require byte-identical semantic payloads and distinct run identities."""

    left = Path(left)
    right = Path(right)
    left_verification = verify_artifact(left)
    right_verification = verify_artifact(right)
    for name in SEMANTIC_FILES:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise ValueError(f"A/B semantic byte mismatch: {name}")
    if (
        left_verification["semantic_result_sha256"]
        != right_verification["semantic_result_sha256"]
    ):
        raise ValueError("A/B semantic root mismatch")
    left_manifest = _load_json(left / "run_manifest.json")
    right_manifest = _load_json(right / "run_manifest.json")
    assert isinstance(left_manifest, Mapping) and isinstance(right_manifest, Mapping)
    left_provenance = left_manifest["provenance"]
    right_provenance = right_manifest["provenance"]
    assert isinstance(left_provenance, Mapping) and isinstance(
        right_provenance, Mapping
    )
    for field in ("run_uuid", "start_utc", "output_path", "replicate"):
        if left_provenance.get(field) == right_provenance.get(field):
            raise ValueError(f"A/B run provenance is not independent: {field}")
    return str(left_verification["semantic_result_sha256"])


def _empty_stop_records(
    *, stop_code: str, stop_message: str, phase: str
) -> ArtifactRecords:
    return ArtifactRecords(
        execution_mode=FORMAL_EXECUTION_MODE,
        status="STOP",
        raw_steps=(),
        source_events=(),
        owner_events=(),
        particle_counts=(),
        raw_loads=(),
        transport_stages=(),
        trajectories=(),
        terminal_coordinate={
            "transport_substeps": None,
            "layer": None,
            "source_step_index": None,
            "ptera_step_index": None,
            "substep": None,
            "stage": None,
            "phase": phase,
            "stage_began": False,
        },
        stop_code=stop_code,
        stop_message=stop_message,
    )


def _terminal_coordinate_from_sink(
    sink: ArtifactSink, *, phase: str
) -> dict[str, object]:
    """Describe the exact durable prefix without inventing failed-stage evidence."""

    stages = sink.transport_stages
    sources = sink.source_events
    if stages and stages[-1].get("status") == "failed":
        last = stages[-1]
        return {
            "transport_substeps": last["transport_substeps"],
            "layer": last["layer"],
            "source_step_index": last["source_step_index"],
            "ptera_step_index": last["ptera_step_index"],
            "substep": last["substep"],
            "stage": last["stage"],
            "phase": phase,
            "stage_began": True,
        }
    terminal_six = _expected_unbegun_terminal_six(
        completed_stage_count=len(stages),
        layer_keys=tuple(_layer_key(row) for row in sink.raw_steps),
        source_keys=tuple(_source_key(row) for row in sources),
    )
    return {
        **dict(zip(TERMINAL_COORDINATE_FIELDS[:6], terminal_six)),
        "phase": phase,
        "stage_began": False,
    }


def _sink_has_complete_matrix_shape(sink: ArtifactSink) -> bool:
    return (
        len(sink.raw_steps) == len(EXPECTED_LAYER_KEYS)
        and len(sink.source_events) == len(EXPECTED_SOURCE_KEYS)
        and len(sink.owner_events) == len(EXPECTED_LAYER_KEYS)
        and len(sink.particle_counts) == len(EXPECTED_LAYER_KEYS)
        and len(sink.raw_loads) == len(EXPECTED_LOAD_KEYS)
        and len(sink.transport_stages) == len(EXPECTED_STAGE_KEYS)
        and len(sink.trajectories) == len(EXPECTED_LAYER_KEYS)
        and all(row.get("status") == "completed" for row in sink.transport_stages)
    )


def _validated_stop_from_sink(
    sink: ArtifactSink,
    *,
    stop_code: str,
    stop_message: str,
    phase: str,
    terminal_coordinate: Mapping[str, object] | None = None,
) -> ArtifactRecords:
    """Return a publishable last-good STOP even when executor STOP data is bad."""

    terminal = (
        _terminal_coordinate_from_sink(sink, phase=phase)
        if terminal_coordinate is None
        else dict(terminal_coordinate)
    )
    candidate = sink.freeze(
        execution_mode=FORMAL_EXECUTION_MODE,
        status="STOP",
        terminal_coordinate=terminal,
        stop_code=stop_code,
        stop_message=stop_message,
    )
    try:
        _validate_records(candidate)
        return candidate
    except Exception as first_error:
        derived = sink.freeze(
            execution_mode=FORMAL_EXECUTION_MODE,
            status="STOP",
            terminal_coordinate=_terminal_coordinate_from_sink(
                sink, phase="coupling_callback"
            ),
            stop_code="coupling_stop_contract_error",
            stop_message=(
                f"{type(first_error).__name__}: {first_error}; "
                "executor STOP evidence was replaced by the durable sink coordinate"
            ),
        )
        try:
            _validate_records(derived)
            return derived
        except Exception as second_error:
            return _empty_stop_records(
                stop_code="coupling_stop_contract_error",
                stop_message=(
                    f"{type(second_error).__name__}: {second_error}; "
                    "invalid sink prefix was excluded from publication"
                ),
                phase="coupling_callback",
            )


def _publish_formal_after_dependency_rehash_impl(
    records: ArtifactRecords,
    destination: Path,
    *,
    replicate: str,
    provenance: Mapping[str, object],
    preflight_dependency_audit: Mapping[str, object],
    formal_publish: object,
) -> dict[str, object]:
    if not callable(formal_publish):
        raise TypeError("formal publisher closure is unavailable")
    try:
        postflight = _capture_observed_runtime_modules(preflight_dependency_audit)
        return formal_publish(
            records,
            destination,
            replicate=replicate,
            run_provenance=provenance,
            dependency_audit=postflight,
        )
    except (DependencyFreezeError, FileNotFoundError) as error:
        return formal_publish(
            _empty_stop_records(
                stop_code="dependency_drift",
                stop_message=f"{type(error).__name__}: {error}",
                phase="dependency_postflight",
            ),
            destination,
            replicate=replicate,
            run_provenance=provenance,
            dependency_audit=None,
        )


def _run_formal_attempt_impl(
    output_dir: Path,
    *,
    replicate: str,
    dependency_audit_token: Path | None = DEFAULT_DEPENDENCY_AUDIT_TOKEN,
    invocation_argv: Sequence[str] | None = None,
    formal_publish: object,
    formal_postflight_publish: object,
) -> dict[str, object]:
    """Preflight external hashes before constructing or calling the executor."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    provenance = _run_provenance(destination, replicate, invocation_argv)
    try:
        if dependency_audit_token is None:
            raise DependencyFreezeError("external dependency audit token is unbound")
        dependency_audit = _verified_dependency_audit(dependency_audit_token)
    except (DependencyFreezeError, FileNotFoundError, ValueError) as error:
        if not callable(formal_publish):
            raise TypeError("formal publisher closure is unavailable")
        return formal_publish(
            _empty_stop_records(
                stop_code="dependencies_unbound",
                stop_message=str(error),
                phase="dependency_preflight",
            ),
            destination,
            replicate=replicate,
            run_provenance=provenance,
            dependency_audit=None,
        )
    sink = ArtifactSink()
    try:
        executor = _load_formal_executor(dependency_audit)
    except Exception as error:
        if not callable(formal_postflight_publish):
            raise TypeError("formal postflight publisher closure is unavailable")
        return formal_postflight_publish(
            _empty_stop_records(
                stop_code="coupling_factory_error",
                stop_message=f"{type(error).__name__}: {error}",
                phase="coupling_factory",
            ),
            destination,
            replicate=replicate,
            provenance=provenance,
            preflight_dependency_audit=dependency_audit,
        )
    try:
        executor.run_formal_matrix(levels=FORMAL_LEVELS, sink=sink)
    except FormalRunStopped as error:
        records = _validated_stop_from_sink(
            sink,
            stop_code=error.stop_code,
            stop_message=str(error),
            phase="coupling_callback",
            terminal_coordinate=error.terminal_coordinate,
        )
    except Exception as error:
        records = _validated_stop_from_sink(
            sink,
            stop_code="coupling_callback_error",
            stop_message=f"{type(error).__name__}: {error}",
            phase="coupling_callback",
        )
    else:
        records = sink.freeze(execution_mode=FORMAL_EXECUTION_MODE, status="PASS")
        try:
            _validate_records(records)
        except Exception as error:
            records = _validated_stop_from_sink(
                sink,
                stop_code=(
                    "coupling_callback_contract_error"
                    if _sink_has_complete_matrix_shape(sink)
                    else "coupling_incomplete"
                ),
                stop_message=f"{type(error).__name__}: {error}",
                phase="coupling_completion",
            )
    if not callable(formal_postflight_publish):
        raise TypeError("formal postflight publisher closure is unavailable")
    return formal_postflight_publish(
        records,
        destination,
        replicate=replicate,
        provenance=provenance,
        preflight_dependency_audit=dependency_audit,
    )


def _build_publication_surfaces() -> tuple[object, object]:
    """Close formal publication authority over the one audited entry route."""

    core = _publish_artifact_core
    postflight_impl = _publish_formal_after_dependency_rehash_impl
    formal_impl = _run_formal_attempt_impl

    def public_publish(
        records: ArtifactRecords,
        output_dir: Path,
        *,
        replicate: str,
        run_provenance: Mapping[str, object] | None = None,
        dependency_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if records.execution_mode == FORMAL_EXECUTION_MODE:
            raise PermissionError(
                "formal artifacts may only be published by the audited formal entry path"
            )
        return core(
            records,
            output_dir,
            replicate=replicate,
            run_provenance=run_provenance,
            dependency_audit=dependency_audit,
            formal_authorized=False,
        )

    def formal_publish(
        records: ArtifactRecords,
        output_dir: Path,
        *,
        replicate: str,
        run_provenance: Mapping[str, object] | None = None,
        dependency_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return core(
            records,
            output_dir,
            replicate=replicate,
            run_provenance=run_provenance,
            dependency_audit=dependency_audit,
            formal_authorized=True,
        )

    def formal_postflight_publish(
        records: ArtifactRecords,
        destination: Path,
        *,
        replicate: str,
        provenance: Mapping[str, object],
        preflight_dependency_audit: Mapping[str, object],
    ) -> dict[str, object]:
        return postflight_impl(
            records,
            destination,
            replicate=replicate,
            provenance=provenance,
            preflight_dependency_audit=preflight_dependency_audit,
            formal_publish=formal_publish,
        )

    def formal_attempt(
        output_dir: Path,
        *,
        replicate: str,
        dependency_audit_token: Path | None = DEFAULT_DEPENDENCY_AUDIT_TOKEN,
        invocation_argv: Sequence[str] | None = None,
    ) -> dict[str, object]:
        return formal_impl(
            output_dir,
            replicate=replicate,
            dependency_audit_token=dependency_audit_token,
            invocation_argv=invocation_argv,
            formal_publish=formal_publish,
            formal_postflight_publish=formal_postflight_publish,
        )

    return public_publish, formal_attempt


publish_artifact, run_formal_attempt = _build_publication_surfaces()
for _hidden_publication_name in (
    "_publish_artifact_core",
    "_publish_formal_after_dependency_rehash_impl",
    "_run_formal_attempt_impl",
    "_build_publication_surfaces",
):
    globals().pop(_hidden_publication_name, None)
del _hidden_publication_name


def main(argv: Sequence[str] | None = None) -> int:
    if __package__:
        print(
            "STOP formal invocation must execute this audited runner file directly; "
            "package -m startup occurs before dependency preflight"
        )
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate", choices=("A", "B"), required=True)
    parser.add_argument("--dependency-audit-token", type=Path)
    args = parser.parse_args(argv)
    try:
        verification = run_formal_attempt(
            args.output_dir,
            replicate=args.replicate,
            dependency_audit_token=args.dependency_audit_token,
            invocation_argv=tuple(sys.argv if argv is None else argv),
        )
    except FileExistsError as error:
        print(f"STOP {error}; no files written")
        return 2
    print(
        f"{verification['status']} v5h13 no-GT inner artifact; "
        f"semantic_result_sha256={verification['semantic_result_sha256']}"
    )
    return 0 if verification["status"] == "PASS" else 2


__all__ = (
    "ARTIFACT_FILES",
    "ArtifactRecords",
    "ArtifactSink",
    "EXPECTED_LAYER_KEYS",
    "EXPECTED_LOAD_KEYS",
    "EXPECTED_SOURCE_KEYS",
    "EXPECTED_STAGE_KEYS",
    "FIXED_PROBES_GP1_M",
    "FIXED_PROBE_SOURCE_SHA256",
    "FORMAL_LEVELS",
    "FormalCouplingExecutor",
    "FormalExecutorAPI",
    "FormalRunStopped",
    "FRONTIER_NODE_IDS",
    "PANEL_IDS",
    "RAW_LOAD_FIELDS",
    "RAW_STEP_FIELDS",
    "SEMANTIC_FILES",
    "SOURCE_EVENT_FIELDS",
    "STAGE_FIELDS",
    "compare_semantic_artifacts",
    "compact_stage_stability_fields",
    "decode_array",
    "encode_array",
    "publish_artifact",
    "recompute_convergence_from_artifacts",
    "run_formal_attempt",
    "verify_artifact",
)


if __name__ == "__main__":
    raise SystemExit(main())
