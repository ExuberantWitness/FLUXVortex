"""Run the preregistered read-only N1/N2/N3 G0c AoA ladder.

The campaign adds no aerodynamic formula or force.  It reuses the audited
last-cycle recorder and artifact/runtime machinery from the N1/N2 G0 runner,
while owning a separate eight-case contract and content-addressed source
closure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import run_n1_n2_ledger_phase_witnesses as base


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
SRC = ROOT / "src"
DOCS = PLATFORM / "docs"
DIAG = DOCS / "diag"
PREREG = DIAG / "n1_n2_n3_aoa_ladder_prereg_20260730.md"
N5_YAML = PLATFORM / "claim_nodes" / "n5_twist_coupling.yaml"

SCHEMA_VERSION = "n1-n2-n3-aoa-ladder-witness-v1"
RAW_SCHEMA_VERSION = base.RAW_SCHEMA_VERSION
RAW_STAGE = "G0_exploratory_quick_identity"
CAMPAIGN_STAGE = "G0c_supplementary_aoa_attribution"
CAMPAIGN_SCOPE = (
    "read-only Fig19(a,b) frequency-endpoint AoA-gradient attribution "
    "across N1 leading-edge suction, N2 separation direction, and N3 "
    "booked direct vortex force"
)

EXPERIMENT_PHASE_DEG = -90.0
NOMINAL_TWIST_DEG = 22.5
SOLVER_TWIST_AMPLITUDE_DEG = NOMINAL_TWIST_DEG / 2.0
G0_NC = 4
G0_NS = 8
G0_N_CYCLE = 2
G0_STEPS_PER_CYCLE = 240
G0_WAKE_ROWS = 240
EXPECTED_RAW_FIELD_COUNT = 92
PRECONDITIONER_FORCE_TOLERANCE_N = 0.15
MATERIAL_CONTRAST_TOLERANCE_N = 0.30
COLLINEAR_CONDITION_LIMIT = 20.0
NUMERIC_ZERO = 1.0e-12

CaseContract = base.CaseContract
WitnessContractError = base.WitnessContractError

AOA_VALUES_DEG = (0.0, 5.0, 10.0, 15.0)
FREQUENCY_CONTEXTS_HZ = (1.4, 2.6)

PRECONDITIONER_CASE = CaseContract(
    case_id="excluded_current_source_preconditioner",
    family="excluded_preconditioner",
    physical_id="P0",
    U_m_s=8.0,
    frequency_Hz=2.6,
    nominal_twist_deg=0.0,
    aoa_deg=5.0,
    twist_phase_deg=EXPERIMENT_PHASE_DEG,
    roles=(
        "excluded_numeric_runtime_preconditioner",
        "zero_twist_phase_equivalent",
    ),
    coverage="none",
)

NEW_CAMPAIGN_SOURCE_FILES = (
    "platform/run_n1_n2_n3_aoa_ladder_witnesses.py",
    "platform/score_n1_n2_n3_aoa_ladder_witnesses.py",
    "platform/tests/test_run_n1_n2_n3_aoa_ladder_witnesses.py",
    "platform/tests/test_score_n1_n2_n3_aoa_ladder_witnesses.py",
    "platform/docs/diag/n1_n2_n3_aoa_ladder_prereg_20260730.md",
)

# The old runner/scorer and their complete governed solver/evidence closure are
# dependencies because this runner intentionally reuses the audited recorder,
# raw guards, runtime fingerprint, artifact writer, and graph identity logic.
DIRECT_SOURCE_FILES = tuple(
    dict.fromkeys((*NEW_CAMPAIGN_SOURCE_FILES, *base.DIRECT_SOURCE_FILES))
)
CONDITIONAL_SOURCE_FILES = base.CONDITIONAL_SOURCE_FILES
SOURCE_GLOBS = base.SOURCE_GLOBS


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _jsonable(value: Any) -> Any:
    return base._jsonable(value)


def _canonical_hash(value: Any) -> str:
    return base._canonical_hash(value)


def _sha256_bytes(value: bytes) -> str:
    return base._sha256_bytes(value)


def _sha256_file(path: Path) -> str:
    return base._sha256_file(path)


def _shared_recorder_contract_gate() -> None:
    expected = {
        "nc": G0_NC,
        "ns": G0_NS,
        "n_cycle": G0_N_CYCLE,
        "steps_per_cycle": G0_STEPS_PER_CYCLE,
        "wake_rows": G0_WAKE_ROWS,
        "raw_schema": RAW_SCHEMA_VERSION,
    }
    actual = {
        "nc": base.G0_NC,
        "ns": base.G0_NS,
        "n_cycle": base.G0_N_CYCLE,
        "steps_per_cycle": base.G0_STEPS_PER_CYCLE,
        "wake_rows": base.G0_STEPS_PER_CYCLE,
        "raw_schema": base.RAW_SCHEMA_VERSION,
    }
    if actual != expected:
        raise WitnessContractError(
            "audited base recorder contract drifted: "
            f"expected={expected}, actual={actual}"
        )
    if (
        base.PRECONDITIONER_FORCE_TOLERANCE_N
        != PRECONDITIONER_FORCE_TOLERANCE_N
    ):
        raise WitnessContractError(
            "base preconditioner force tolerance drifted"
        )


def _frequency_tag(frequency_hz: float) -> str:
    if frequency_hz == 1.4:
        return "1p4"
    if frequency_hz == 2.6:
        return "2p6"
    raise WitnessContractError(
        f"unregistered frequency context {frequency_hz}"
    )


def _aoa_tag(aoa_deg: float) -> str:
    if aoa_deg not in AOA_VALUES_DEG:
        raise WitnessContractError(f"unregistered AoA {aoa_deg}")
    return str(int(aoa_deg))


def _case_contracts() -> tuple[CaseContract, ...]:
    cases = tuple(
        CaseContract(
            case_id=(
                f"aoa_f{_frequency_tag(frequency)}_"
                f"A{_aoa_tag(aoa)}"
            ),
            family="fig19_aoa_ladder",
            physical_id=(
                f"F19_f{_frequency_tag(frequency)}_"
                f"A{_aoa_tag(aoa)}"
            ),
            U_m_s=8.0,
            frequency_Hz=frequency,
            nominal_twist_deg=NOMINAL_TWIST_DEG,
            aoa_deg=aoa,
            twist_phase_deg=EXPERIMENT_PHASE_DEG,
            roles=(
                "experiment_identity_corrected_minus90",
                "fig19_raw_frequency_endpoint",
                "g0c_supplementary_aoa_attribution",
            ),
            coverage=(
                f"fig19_ab_f{_frequency_tag(frequency)}_"
                f"aoa{_aoa_tag(aoa)}"
            ),
        )
        for frequency in FREQUENCY_CONTEXTS_HZ
        for aoa in AOA_VALUES_DEG
    )
    identities = [case.case_id for case in cases]
    if len(cases) != 8 or len(identities) != len(set(identities)):
        raise WitnessContractError(
            "G0c must contain exactly eight unique case identities"
        )
    calls = [case.condition_tuple for case in cases]
    if len(calls) != len(set(calls)):
        raise WitnessContractError("duplicate G0c solver call")
    for case in cases:
        if (
            case.U_m_s != 8.0
            or case.nominal_twist_deg != NOMINAL_TWIST_DEG
            or case.solver_twist_amplitude_deg
            != SOLVER_TWIST_AMPLITUDE_DEG
            or case.twist_phase_deg != EXPERIMENT_PHASE_DEG
        ):
            raise WitnessContractError(
                f"{case.case_id}: science identity drift"
            )
    return cases


def _source_paths() -> tuple[Path, ...]:
    paths = {ROOT / relative for relative in DIRECT_SOURCE_FILES}
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted(paths)
        if not path.is_file()
    ]
    if missing:
        raise WitnessContractError(
            f"mandatory G0c source-closure files are missing: {missing}"
        )
    for relative in CONDITIONAL_SOURCE_FILES:
        path = ROOT / relative
        if path.exists() and not path.is_file():
            raise WitnessContractError(
                f"conditional source-closure path is not a file: {relative}"
            )
        if path.is_file():
            paths.add(path)
    for pattern in SOURCE_GLOBS:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return tuple(sorted(path.resolve() for path in paths))


def _source_closure_from_paths(
    paths: Sequence[Path],
    *,
    root: Path,
) -> dict[str, Any]:
    return base._source_closure_from_paths(paths, root=root)


def _source_closure_snapshot() -> tuple[dict[str, Any], bytes]:
    """Read each governed file once and bind N5 parsing to captured bytes."""
    root = ROOT.resolve()
    n5_path = N5_YAML.resolve()
    members: dict[str, str] = {}
    n5_yaml_bytes: bytes | None = None
    for path in _source_paths():
        content = path.read_bytes()
        relative = str(path.relative_to(root))
        members[relative] = _sha256_bytes(content)
        if path == n5_path:
            n5_yaml_bytes = content
    if n5_yaml_bytes is None:
        raise WitnessContractError("N5 YAML is absent from G0c closure")
    closure = {
        "schema": "content-addressed-source-closure-v1",
        "members": members,
        "members_sha256": _canonical_hash(members),
        "governed_globs": list(SOURCE_GLOBS),
        "conditional_files": list(CONDITIONAL_SOURCE_FILES),
    }
    return closure, n5_yaml_bytes


def _source_closure() -> dict[str, Any]:
    closure, _ = _source_closure_snapshot()
    return closure


def _kinematic_identity_gate(
    cases: Sequence[CaseContract],
    *,
    n5_yaml_bytes: bytes,
    expected_n5_yaml_sha256: str,
    source_closure_sha256: str,
) -> dict[str, Any]:
    gate = base._kinematic_identity_gate(
        cases,
        n5_yaml=N5_YAML,
        n5_yaml_bytes=n5_yaml_bytes,
        expected_n5_yaml_sha256=expected_n5_yaml_sha256,
        source_closure_sha256=source_closure_sha256,
    )
    if any(
        case.twist_phase_deg != EXPERIMENT_PHASE_DEG for case in cases
    ):
        raise WitnessContractError(
            "all G0c science cases must use N5.1c -90 identity"
        )
    gate = dict(gate)
    gate.update(
        {
            "campaign_stage": CAMPAIGN_STAGE,
            "all_science_cases_use_experiment_identity_minus90": True,
            "production_plus90_science_calls": 0,
        }
    )
    return gate


def _campaign_inputs(
    cases: Sequence[CaseContract],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _shared_recorder_contract_gate()
    source_closure, n5_yaml_bytes = _source_closure_snapshot()
    n5_relative = str(N5_YAML.resolve().relative_to(ROOT.resolve()))
    n5_sha256 = source_closure["members"].get(n5_relative)
    if not isinstance(n5_sha256, str):
        raise WitnessContractError("N5 hash is missing from G0c closure")
    gate = _kinematic_identity_gate(
        cases,
        n5_yaml_bytes=n5_yaml_bytes,
        expected_n5_yaml_sha256=n5_sha256,
        source_closure_sha256=source_closure["members_sha256"],
    )
    if gate.get("source_snapshot_bound") is not True:
        raise WitnessContractError(
            "G0c N5 gate is not bound to its source snapshot"
        )
    return source_closure, gate


def _assert_source_closure(expected: Mapping[str, Any]) -> None:
    current = _source_closure()
    if current != expected:
        expected_members = expected.get("members", {})
        current_members = current.get("members", {})
        changed = sorted(
            key
            for key in set(expected_members) | set(current_members)
            if expected_members.get(key) != current_members.get(key)
        )
        raise WitnessContractError(
            "current G0c source closure drifted: "
            + ", ".join(changed[:20])
        )


def _governed_module_sources(
    source_closure: Mapping[str, Any],
) -> dict[str, str]:
    members = source_closure.get("members")
    if not isinstance(members, Mapping):
        raise WitnessContractError("G0c source-closure members are missing")
    module_sources: dict[str, str] = {}
    for relative in members:
        if not isinstance(relative, str) or not relative.endswith(".py"):
            continue
        parts = list(Path(relative).parts)
        if parts[:1] == ["platform"]:
            parts = parts[1:]
        elif parts[:2] == ["src", "fluxvortex"]:
            parts = parts[1:]
        else:
            continue
        parts[-1] = Path(parts[-1]).stem
        if parts[-1] == "__init__":
            parts.pop()
        if not parts:
            continue
        module_name = ".".join(parts)
        previous = module_sources.get(module_name)
        if previous is not None and previous != relative:
            raise WitnessContractError(
                f"duplicate governed module identity {module_name}: "
                f"{previous}, {relative}"
            )
        module_sources[module_name] = relative
    return module_sources


def _reject_preloaded_governed_solver_modules(
    source_closure: Mapping[str, Any],
) -> None:
    """Reject solver modules loaded before the content snapshot was bound."""

    allowed = {__name__, base.__name__}
    governed = _governed_module_sources(source_closure)
    preloaded = sorted(
        module_name
        for module_name in governed
        if module_name in sys.modules and module_name not in allowed
    )
    if preloaded:
        raise WitnessContractError(
            "governed solver modules were preloaded before source binding: "
            + ", ".join(preloaded[:20])
        )


def _loaded_governed_module_identities(
    source_closure: Mapping[str, Any],
    *,
    required: Sequence[str],
) -> dict[str, dict[str, Any]]:
    members = source_closure["members"]
    module_sources = _governed_module_sources(source_closure)
    identities: dict[str, dict[str, Any]] = {}
    root = ROOT.resolve()
    for module_name, relative in sorted(module_sources.items()):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        identity = base._module_file_identity(
            module,
            module_name=module_name,
            required_root=root,
        )
        expected_path = (root / relative).resolve()
        if (
            Path(identity["path"]) != expected_path
            or identity["relative_path"] != relative
            or identity["sha256"] != members.get(relative)
        ):
            raise WitnessContractError(
                f"{module_name} execution bytes are not bound to the "
                f"G0c source closure"
            )
        identities[module_name] = identity
    missing = sorted(set(required) - set(identities))
    if missing:
        raise WitnessContractError(
            "required governed execution modules were not loaded: "
            + ", ".join(missing)
        )
    return identities


def _load_bound_solver(
    source_closure: Mapping[str, Any],
    cases: Sequence[CaseContract],
) -> tuple[
    Callable[..., Mapping[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load solver and call dictionary only after rejecting stale modules."""

    _reject_preloaded_governed_solver_modules(source_closure)
    solver, runtime = base._load_solver()
    call_config_module = importlib.import_module("lb_sweep118")
    identities = _loaded_governed_module_identities(
        source_closure,
        required=("_v2_robo", "lb_sweep118"),
    )
    solver_module = sys.modules["_v2_robo"]
    if (
        getattr(solver, "__module__", None) != "_v2_robo"
        or getattr(solver_module, "gpu_run_twist", None) is not solver
    ):
        raise WitnessContractError(
            "gpu_run_twist callable is not owned by bound _v2_robo"
        )
    raw_base_config = getattr(call_config_module, "BASE", None)
    if not isinstance(raw_base_config, Mapping):
        raise WitnessContractError("bound lb_sweep118.BASE is not a mapping")
    base_config = dict(raw_base_config)
    binding: dict[str, Any] = {
        "schema": "g0c-execution-binding-v1",
        "source_closure_sha256": source_closure["members_sha256"],
        "entry_modules": {
            name: identities[name]
            for name in ("_v2_robo", "lb_sweep118")
        },
        "loaded_governed_modules": identities,
        "solver_callable": {
            "module": "_v2_robo",
            "qualname": getattr(solver, "__qualname__", None),
        },
        "base_config": _jsonable(base_config),
        "base_config_sha256": _canonical_hash(base_config),
        "resolved_calls": {
            case.case_id: base._resolved_call(
                solver,
                _call_from_base(case, base_config),
            )
            for case in (PRECONDITIONER_CASE, *cases)
        },
    }
    binding["binding_sha256"] = _canonical_hash(binding)
    return solver, base_config, runtime, binding


def _validate_bound_base_config(
    base_config: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> None:
    payload = dict(execution_binding)
    binding_sha256 = payload.pop("binding_sha256", None)
    if (
        execution_binding.get("schema") != "g0c-execution-binding-v1"
        or execution_binding.get("source_closure_sha256") is None
        or _canonical_hash(payload) != binding_sha256
        or _canonical_hash(dict(base_config))
        != execution_binding.get("base_config_sha256")
        or _jsonable(dict(base_config))
        != execution_binding.get("base_config")
    ):
        raise WitnessContractError("G0c execution binding drifted")


def _call_from_base(
    case: CaseContract,
    base_config: Mapping[str, Any],
) -> dict[str, Any]:
    call = dict(base_config)
    call.update(
        U=case.U_m_s,
        aoa_deg=case.aoa_deg,
        freq=case.frequency_Hz,
        twist_amp_deg=case.solver_twist_amplitude_deg,
        twist_phase_deg=case.twist_phase_deg,
        nc=G0_NC,
        ns=G0_NS,
        n_cycle=G0_N_CYCLE,
        steps_per_cycle=G0_STEPS_PER_CYCLE,
        wake_rows=G0_WAKE_ROWS,
        closure="v41",
    )
    if call.get("closure") != "v41":
        raise WitnessContractError("candidate closure is forbidden")
    return call


def _build_solver_call(
    case: CaseContract,
    *,
    base_config: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_bound_base_config(base_config, execution_binding)
    return _call_from_base(case, base_config)


def _assert_bound_resolved_call(
    solver: Callable[..., Mapping[str, Any]],
    case: CaseContract,
    call: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = base._resolved_call(solver, call)
    expected = execution_binding.get("resolved_calls", {}).get(case.case_id)
    if resolved != expected:
        raise WitnessContractError(
            f"{case.case_id}: resolved solver call drifted from the "
            "pre-execution binding"
        )
    return resolved


def _expected_claim_raw_config(
    case: CaseContract,
) -> dict[str, Any]:
    """Exact recorder identity for the frozen V4.1 closure."""

    return {
        "closure": "v41",
        "nc": G0_NC,
        "ns": G0_NS,
        "n_cycle": G0_N_CYCLE,
        "steps_per_cycle": G0_STEPS_PER_CYCLE,
        "wake_rows": G0_WAKE_ROWS,
        "U_m_s": case.U_m_s,
        "aoa_deg": case.aoa_deg,
        "freq_hz": case.frequency_Hz,
        "flap_amp_deg": 22.5,
        "twist_amp_deg": case.solver_twist_amplitude_deg,
        "twist_phase_deg": case.twist_phase_deg,
        "real_geom": True,
        "sym": True,
        "lb_closure": True,
        "lb_hybrid": 0.0,
        "lb_cds": 2.5,
        "lb_cds_mem": True,
        "lb_cds_f2gate": False,
        "lb_cds_zonly": False,
        "lb_cds_signed": False,
        "lb_chop_zonly": True,
        "lb_ct": False,
        "lb_cla3d": True,
        "lb_lesp_crit": 0.18,
        "d_para_at_U8_N": 0.5,
        "attached_drag": "uiuc",
    }


def _git_metadata() -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {"head": head}


def _contrast_contract() -> dict[str, Any]:
    return {
        "residual_definition": "experiment_minus_model",
        "component_order": ["T", "L"],
        "frequency_contexts_Hz": list(FREQUENCY_CONTEXTS_HZ),
        "within_each_frequency": [
            {"id": "C_5_0", "high_aoa_deg": 5.0, "low_aoa_deg": 0.0},
            {
                "id": "C_10_5",
                "high_aoa_deg": 10.0,
                "low_aoa_deg": 5.0,
            },
            {
                "id": "C_15_10",
                "high_aoa_deg": 15.0,
                "low_aoa_deg": 10.0,
            },
        ],
        "material_strictly_greater_than_N": (
            MATERIAL_CONTRAST_TOLERANCE_N
        ),
        "force_gate_N": PRECONDITIONER_FORCE_TOLERANCE_N,
        "three_column_cond2_fail_strictly_greater_than": (
            COLLINEAR_CONDITION_LIMIT
        ),
        "numeric_zero": NUMERIC_ZERO,
    }


def _template_contract() -> dict[str, Any]:
    return {
        "component_order": ["T", "L"],
        "Q1": {
            "definition": "-Delta(N1 leading-edge-suction force)",
            "source": (
                "diagnostic.n1.leading_edge_suction_"
                "solver_accumulator_body_force_N"
            ),
            "half_wing_to_reported_pair_factor": 2.0,
            "role": "withdrawal_direction_only",
            "implementation_authorized": False,
        },
        "Q2": {
            "definition": (
                "Delta(cycle reduction of per-step unitized old N2 "
                "separation-panel observer in wind-axis (T,L))"
            ),
            "source": "n2.separation_panel_candidate_force_body_N",
            "unitize_before_time_reduction": True,
            "zero_norm_threshold_N": NUMERIC_ZERO,
            "newton_magnitude_interpretation_allowed": False,
            "implementation_authorized": False,
        },
        "Q3": {
            "definition": "-Delta(N3 booked direct vortex force)",
            "source": "n3.ds_booked_solver_accumulator_N",
            "half_wing_to_reported_pair_factor": 2.0,
            "role": "withdrawal_direction_only",
            "implementation_authorized": False,
        },
    }


def _campaign_contract(
    cases: Sequence[CaseContract],
) -> dict[str, Any]:
    case_payload = _jsonable([asdict(case) for case in cases])
    return {
        "schema": SCHEMA_VERSION,
        "campaign_stage": CAMPAIGN_STAGE,
        "raw_schema": RAW_SCHEMA_VERSION,
        "raw_stage": RAW_STAGE,
        "raw_expected_field_count": EXPECTED_RAW_FIELD_COUNT,
        "cases": case_payload,
        "case_contract_sha256": _canonical_hash(case_payload),
        "expected_unique_science_solver_calls": 8,
        "excluded_preconditioner_calls_per_session": 1,
        "expected_fresh_session_total_solver_invocations": 9,
        "closure": "v41",
        "grid": {
            "nc": G0_NC,
            "ns": G0_NS,
            "n_cycle": G0_N_CYCLE,
            "steps_per_cycle": G0_STEPS_PER_CYCLE,
            "wake_rows": G0_WAKE_ROWS,
        },
        "U_m_s": 8.0,
        "nominal_twist_deg": NOMINAL_TWIST_DEG,
        "nominal_twist_to_solver_amplitude": 0.5,
        "science_twist_phase_deg": EXPERIMENT_PHASE_DEG,
        "preconditioner_case": _jsonable(asdict(PRECONDITIONER_CASE)),
        "figure19_experiment": {
            "source": "platform/docs/data.md",
            "figures": ["19(a)", "19(b)"],
            "frequency_axis_endpoints_Hz": list(
                FREQUENCY_CONTEXTS_HZ
            ),
            "force_interpolation_allowed": False,
            "endpoint_projection": (
                "identity only via frozen benchmark tolerance; preserve "
                "raw digitized x/y"
            ),
        },
        "contrasts": _contrast_contract(),
        "templates": _template_contract(),
        "raw_and_robust_must_agree": True,
        "both_frequency_contexts_must_uniquely_support_same_column": True,
        "alternatives_require_reverse_evidence": True,
        "multi_support_conflict_allowed": False,
        "production_grid_claim_allowed": False,
        "candidate_implementation_authorized": False,
        "claim_writeback_authorized": False,
    }


def _new_campaign(
    *,
    source_closure: Mapping[str, Any],
    cases: Sequence[CaseContract],
    identity_gate: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "status": "running",
        "created_at": _now(),
        "updated_at": _now(),
        "campaign_stage": CAMPAIGN_STAGE,
        "scientific_scope": CAMPAIGN_SCOPE,
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": _sha256_file(PREREG),
        },
        "source_closure": dict(source_closure),
        "source_closure_sha256": source_closure["members_sha256"],
        "execution_binding": dict(execution_binding),
        "execution_binding_sha256": execution_binding["binding_sha256"],
        "git": _git_metadata(),
        "contract": _campaign_contract(cases),
        "kinematic_identity_gate": dict(identity_gate),
        "numeric_runtime": None,
        "common_claim_graph_identity_sha256": None,
        "sessions": [],
        "cases": {},
        "failures": {},
        "active_case": None,
    }


def _open_campaign(
    output: Path,
    *,
    resume: bool,
    source_closure: Mapping[str, Any],
    cases: Sequence[CaseContract],
    identity_gate: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = output / "run_manifest.json"
    expected_contract = _campaign_contract(cases)
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"{output} exists; use a new path or explicit --resume"
            )
        if not manifest_path.is_file():
            raise WitnessContractError(
                f"{output}: resume manifest is missing"
            )
        campaign = base._load_json(manifest_path)
        if campaign.get("schema") != SCHEMA_VERSION:
            raise WitnessContractError("resume G0c schema mismatch")
        if campaign.get("contract") != expected_contract:
            raise WitnessContractError("resume G0c case contract mismatch")
        if campaign.get("source_closure") != source_closure:
            raise WitnessContractError("resume G0c source closure mismatch")
        if campaign.get("kinematic_identity_gate") != identity_gate:
            raise WitnessContractError("resume G0c N5 gate mismatch")
        if campaign.get("execution_binding") != execution_binding:
            raise WitnessContractError(
                "resume G0c execution binding mismatch"
            )
        if campaign.get("execution_binding_sha256") != (
            execution_binding.get("binding_sha256")
        ):
            raise WitnessContractError(
                "resume G0c execution-binding hash mismatch"
            )

        completed = campaign.get("cases")
        sessions = campaign.get("sessions")
        failures = campaign.get("failures")
        if (
            not isinstance(completed, Mapping)
            or not isinstance(sessions, list)
            or not isinstance(failures, Mapping)
        ):
            raise WitnessContractError("resume G0c manifest is malformed")
        expected_ids = {case.case_id for case in cases}
        extra = sorted(set(completed) - expected_ids)
        if extra:
            raise WitnessContractError(
                "resume contains unexpected G0c science cases: "
                + ", ".join(extra)
            )
        for session in sessions:
            if not isinstance(session, Mapping):
                raise WitnessContractError(
                    "resume G0c session is malformed"
                )
            session_case_ids = session.get("completed_case_ids", ())
            if not isinstance(session_case_ids, list):
                raise WitnessContractError(
                    "resume completed_case_ids is malformed"
                )
            extra_session = sorted(set(session_case_ids) - expected_ids)
            if extra_session:
                raise WitnessContractError(
                    "resume session contains unexpected G0c cases: "
                    + ", ".join(extra_session)
                )
        for case_id, record in completed.items():
            base._validate_completed_case_artifacts(
                output,
                case_id,
                record,
            )

        campaign["status"] = "running"
        campaign["active_case"] = None
        campaign["updated_at"] = _now()
        base._write_json_atomic(manifest_path, campaign)
        return campaign

    output.mkdir(parents=True)
    campaign = _new_campaign(
        source_closure=source_closure,
        cases=cases,
        identity_gate=identity_gate,
        execution_binding=execution_binding,
    )
    base._write_json_atomic(manifest_path, campaign)
    return campaign


def _execute_science_case(
    solver: Callable[..., Mapping[str, Any]],
    case: CaseContract,
    *,
    base_config: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
    case_contract_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    call = _build_solver_call(
        case,
        base_config=base_config,
        execution_binding=execution_binding,
    )
    expected_resolved = _assert_bound_resolved_call(
        solver,
        case,
        call,
        execution_binding,
    )
    bundle, evidence = base._execute_case(
        solver,
        case,
        call_override=call,
    )
    if len(bundle) != EXPECTED_RAW_FIELD_COUNT:
        raise WitnessContractError(
            f"{case.case_id}: expected exactly "
            f"{EXPECTED_RAW_FIELD_COUNT} raw fields, got {len(bundle)}"
        )
    if evidence.get("stage") != RAW_STAGE:
        raise WitnessContractError(
            f"{case.case_id}: audited raw stage drift"
        )
    if evidence.get("resolved_call") != expected_resolved:
        raise WitnessContractError(
            f"{case.case_id}: executed call differs from bound call"
        )
    if evidence.get("claim_raw_config") != (
        _expected_claim_raw_config(case)
    ):
        raise WitnessContractError(
            f"{case.case_id}: full V4.1 raw configuration drifted"
        )
    evidence = dict(evidence)
    evidence.update(
        {
            "schema": SCHEMA_VERSION,
            "campaign_schema": SCHEMA_VERSION,
            "campaign_stage": CAMPAIGN_STAGE,
            "campaign_scope": CAMPAIGN_SCOPE,
            "campaign_case_contract_sha256": case_contract_sha256,
            "execution_binding_sha256": execution_binding[
                "binding_sha256"
            ],
            "witness_role": (
                "read_only_g0c_supplementary_aoa_attribution"
            ),
            "figure19_endpoint_contract": {
                "frequency_context_Hz": case.frequency_Hz,
                "aoa_deg": case.aoa_deg,
                "experimental_force_interpolation_allowed": False,
                "experimental_force_used_by_runner": False,
            },
            "frozen_template_contract": _template_contract(),
            "candidate_implementation_authorized": False,
            "claim_writeback_authorized": False,
        }
    )
    return bundle, evidence


def _run_preconditioner(
    solver: Callable[..., Mapping[str, Any]],
    source_closure: Mapping[str, Any],
    *,
    base_config: Mapping[str, Any],
    execution_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_source_closure(source_closure)
    call = _build_solver_call(
        PRECONDITIONER_CASE,
        base_config=base_config,
        execution_binding=execution_binding,
    )
    resolved_call = _assert_bound_resolved_call(
        solver,
        PRECONDITIONER_CASE,
        call,
        execution_binding,
    )
    started = datetime.now().timestamp()
    result = solver(**call)
    _assert_source_closure(source_closure)
    base._validate_claim_guards(result.get("claim_guards"))
    manifest = result.get("claim_manifest")
    if not isinstance(manifest, Mapping):
        raise WitnessContractError(
            "G0c preconditioner claim manifest is missing"
        )
    return {
        "started_at": _now(),
        "purpose": (
            "current-process numerical preconditioning; excluded from "
            "G0c scientific witnesses"
        ),
        "excluded_from_scientific_metrics": True,
        "case_contract": asdict(PRECONDITIONER_CASE),
        "resolved_call": resolved_call,
        "execution_binding_sha256": execution_binding[
            "binding_sha256"
        ],
        "L_wind_N": float(result["L_wind"]),
        "T_wind_N": float(result["T_wind"]),
        "claim_graph_identity_sha256": (
            base._claim_graph_identity_sha256(manifest)
        ),
        "claim_guards": dict(result["claim_guards"]),
        "wall_s": datetime.now().timestamp() - started,
    }


def run(output: Path, *, resume: bool = False) -> int:
    output = output.resolve()
    cases = _case_contracts()
    source_closure, identity_gate = _campaign_inputs(cases)

    with base._campaign_lock(output):
        _assert_source_closure(source_closure)
        solver, base_config, runtime, execution_binding = (
            _load_bound_solver(source_closure, cases)
        )
        _assert_source_closure(source_closure)
        campaign = _open_campaign(
            output,
            resume=resume,
            source_closure=source_closure,
            cases=cases,
            identity_gate=identity_gate,
            execution_binding=execution_binding,
        )
        manifest_path = output / "run_manifest.json"
        _assert_source_closure(source_closure)

        try:
            runtime = base._register_numeric_runtime(campaign, runtime)
        except WitnessContractError as exc:
            campaign["status"] = "failed"
            campaign["failures"]["__numeric_runtime__"] = (
                f"{type(exc).__name__}: {exc}"
            )
            campaign["updated_at"] = _now()
            base._write_json_atomic(manifest_path, campaign)
            raise

        session: dict[str, Any] = {
            "started_at": _now(),
            "process_id": os.getpid(),
            "source_closure_sha256": source_closure["members_sha256"],
            "numeric_runtime": runtime,
            "execution_binding": execution_binding,
            "execution_binding_sha256": execution_binding[
                "binding_sha256"
            ],
            "preconditioner": None,
            "preconditioner_resume_gate": None,
            "completed_case_ids": [],
        }
        campaign["sessions"].append(session)
        campaign["updated_at"] = _now()
        base._write_json_atomic(manifest_path, campaign)

        try:
            print(
                "[n1-n2-n3-aoa-ladder] "
                "excluded current-source preconditioner",
                flush=True,
            )
            session["preconditioner"] = _run_preconditioner(
                solver,
                source_closure,
                base_config=base_config,
                execution_binding=execution_binding,
            )
            session["preconditioner_resume_gate"] = (
                base._preconditioner_resume_gate(campaign, session)
            )
            if (
                session["preconditioner_resume_gate"].get("passed")
                is not True
            ):
                raise WitnessContractError(
                    "G0c preconditioner resume gate did not pass"
                )
            preconditioner_graph = session["preconditioner"][
                "claim_graph_identity_sha256"
            ]
            base._write_json_atomic(manifest_path, campaign)

            common_graph = campaign.get(
                "common_claim_graph_identity_sha256"
            )
            contract_hash = campaign["contract"][
                "case_contract_sha256"
            ]
            for index, case in enumerate(cases, start=1):
                if case.case_id in campaign["cases"]:
                    print(
                        "[n1-n2-n3-aoa-ladder] resume skip "
                        f"{case.case_id}",
                        flush=True,
                    )
                    continue
                _assert_source_closure(source_closure)
                campaign["active_case"] = case.case_id
                campaign["updated_at"] = _now()
                base._write_json_atomic(manifest_path, campaign)
                print(
                    f"[n1-n2-n3-aoa-ladder] {index}/{len(cases)} "
                    f"{case.case_id}",
                    flush=True,
                )
                bundle, evidence = _execute_science_case(
                    solver,
                    case,
                    base_config=base_config,
                    execution_binding=execution_binding,
                    case_contract_sha256=contract_hash,
                )
                _assert_source_closure(source_closure)
                evidence["source_closure_sha256"] = source_closure[
                    "members_sha256"
                ]
                graph_identity = evidence[
                    "claim_graph_identity_sha256"
                ]
                if graph_identity != preconditioner_graph:
                    raise WitnessContractError(
                        f"{case.case_id}: claim graph differs from "
                        "same-session preconditioner"
                    )
                if common_graph is None:
                    common_graph = graph_identity
                    campaign[
                        "common_claim_graph_identity_sha256"
                    ] = graph_identity
                elif graph_identity != common_graph:
                    raise WitnessContractError(
                        f"{case.case_id}: claim graph identity drift"
                    )

                record = base._save_case_artifacts(
                    output,
                    case,
                    bundle,
                    evidence,
                )
                _assert_source_closure(source_closure)
                campaign["cases"][case.case_id] = record
                campaign["failures"].pop(case.case_id, None)
                campaign["active_case"] = None
                campaign["updated_at"] = _now()
                session["completed_case_ids"].append(case.case_id)
                base._write_json_atomic(manifest_path, campaign)
                summary = evidence["diagnostic_summary"]
                force = summary["reported_robust_cycle_force"]
                print(
                    f"[n1-n2-n3-aoa-ladder] {case.case_id}: "
                    f"L={force['L_N']:+.3f} "
                    f"T={force['T_N']:+.3f}",
                    flush=True,
                )

            _assert_source_closure(source_closure)
            if len(campaign["cases"]) != len(cases):
                raise WitnessContractError(
                    "G0c completed case count does not match contract"
                )
            campaign["status"] = "complete"
            campaign["active_case"] = None
            campaign["completed_case_count"] = len(campaign["cases"])
            campaign["production_grid_claim_allowed"] = False
            campaign["candidate_implementation_authorized"] = False
            campaign["next_gate"] = (
                "score frozen Q1/Q2/Q3 AoA contrasts; at most authorize "
                "one next-round shadow preregistration"
            )
            campaign["updated_at"] = _now()
            session["completed_at"] = _now()
            base._write_json_atomic(manifest_path, campaign)
        except Exception as exc:
            active = campaign.get("active_case") or "__campaign__"
            campaign["status"] = "failed"
            campaign["failures"][active] = (
                f"{type(exc).__name__}: {exc}"
            )
            campaign["updated_at"] = _now()
            session["failed_at"] = _now()
            session["failure"] = campaign["failures"][active]
            base._write_json_atomic(manifest_path, campaign)
            raise

    print(
        f"[n1-n2-n3-aoa-ladder] COMPLETE -> {output}",
        flush=True,
    )
    return 0


def _plan_payload() -> dict[str, Any]:
    cases = _case_contracts()
    source_closure, identity_gate = _campaign_inputs(cases)
    return {
        "schema": SCHEMA_VERSION,
        "gpu_initialized": False,
        "campaign_stage": CAMPAIGN_STAGE,
        "scientific_scope": CAMPAIGN_SCOPE,
        "contract": _campaign_contract(cases),
        "kinematic_identity_gate": identity_gate,
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": _sha256_file(PREREG),
        },
        "source_closure": source_closure,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help=(
            "validate and print the frozen G0c contract without "
            "importing Warp"
        ),
    )
    args = parser.parse_args(argv)
    if args.print_plan:
        if args.output is not None or args.resume:
            parser.error("--print-plan cannot be combined with run options")
        print(
            json.dumps(
                _jsonable(_plan_payload()),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.output is None:
        parser.error("--output is required unless --print-plan is used")
    return run(args.output, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
