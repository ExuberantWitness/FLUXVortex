"""Run the preregistered read-only N1/N2 ledger and phase witnesses.

This runner freezes the *current* source closure.  It deliberately does not
reproduce, consume, or update an older V4.1 numerical cache.  The only solver
observer is ``claim_raw_out``; all saved diagnostics are derived from values
already computed by the unchanged ``closure="v41"`` call.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import hashlib
import importlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform as py_platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


PLATFORM = Path(__file__).resolve().parent
ROOT = PLATFORM.parent
SRC = ROOT / "src"
DOCS = PLATFORM / "docs"
DIAG = DOCS / "diag"
PREREG = (
    DIAG / "n1_n2_ledger_phase_witness_prereg_20260729.md"
)
N5_YAML = PLATFORM / "claim_nodes" / "n5_twist_coupling.yaml"
FIG16_SOURCE = DOCS / "datav2.md"

SCHEMA_VERSION = "n1-n2-ledger-phase-witness-v1"
RAW_SCHEMA_VERSION = "claim-raw-ledger-v1"
PRODUCTION_PHASE_DEG = 90.0
EXPERIMENT_PHASE_DEG = -90.0
G2N = 9.81 / 1000.0
LEDGER_TOLERANCE_N = 2.0e-12
GRAPH_LEDGER_TOLERANCE_N = 1.0e-9
PRECONDITIONER_FORCE_TOLERANCE_N = 0.15
G0_NC = 4
G0_NS = 8
G0_N_CYCLE = 2
G0_STEPS_PER_CYCLE = 60 * G0_NC

REQUIRED_CLAIM_GUARDS = (
    "force_ledger",
    "unclassified_force",
    "unclassified_physical_force",
    "cycle_reduction",
    "aero_output_invariance",
)

MEAN_PHYSICAL_WITNESSES = (
    ("W1", 6.0, 2.6, 22.5, 5.0, "fig18_U_boundary_low"),
    ("W2", 10.0, 2.6, 22.5, 5.0, "fig18_U_boundary_high"),
    ("W3", 10.0, 1.4, 22.5, 5.0, "fig18_frequency_boundary_low"),
    ("W4", 8.0, 2.6, 22.5, 5.0, "fig17_18_shared_turn_neighbour"),
    ("W5", 8.0, 2.6, 22.5, 0.0, "fig19_aoa_boundary_low"),
    ("W6", 8.0, 2.6, 22.5, 15.0, "fig19_aoa_boundary_high"),
)

FIG16_PHYSICAL_WITNESSES = (
    ("F16_0", 8.0, 2.0, 0.0, 5.0, "fig16_twist_boundary_zero"),
    ("F16_22p5", 8.0, 2.0, 22.5, 5.0, "fig16_twist_turn"),
    ("F16_45", 8.0, 2.0, 45.0, 5.0, "fig16_twist_boundary_deep"),
)

DIRECT_SOURCE_FILES = (
    "platform/run_n1_n2_ledger_phase_witnesses.py",
    "platform/score_n1_n2_ledger_phase_witnesses.py",
    "platform/claim_dag.py",
    "platform/docs/diag/n1_n2_ledger_phase_witness_prereg_20260729.md",
    "platform/docs/diag/fig171819_active_disease_prereg_v3_20260729.md",
    "platform/docs/diag/research_n2_chordwise_pressure_primary_literature_20260729.md",
    "platform/docs/diag/research_n3_spatial_loads_20260727.md",
    "platform/_v2_robo.py",
    "platform/_v2_repro_nc12.py",
    "platform/_v2_robogeom.py",
    "platform/airfoil_geometry.py",
    "platform/lb_sweep118.py",
    "platform/lb_dyn.py",
    "platform/lb_static.py",
    "platform/cd_table.py",
    "platform/diff_coupled_fsi.py",
    "platform/diff_uvlm_unsteady_gpu.py",
    "platform/diff_uvlm_unsteady.py",
    "platform/diff_coupled_unsteady.py",
    "platform/diff_struct_design.py",
    "platform/diff_vlm.py",
    "platform/flap_flight_validate.py",
    "platform/fig16_compare.py",
    "platform/fig171819_benchmark.py",
    "platform/docs/data.md",
    "platform/docs/datav2.md",
    "researchpaper/Meng2025_Drones_FlappingTwist_RoboEagle_SOURCE.pdf",
)

CONDITIONAL_SOURCE_FILES = (
    "platform/diff_solve.py",
    "platform/rvpm3d.py",
)

SOURCE_GLOBS = (
    "platform/claim_runtime/**/*.py",
    "platform/claim_nodes/*.yaml",
    "platform/claim_nodes/*.yml",
    "src/fluxvortex/**/*.py",
    "researchpaper/uiuc_polars/SD7003.DRG",
)


class WitnessContractError(RuntimeError):
    """The preregistered diagnostic contract is not satisfied."""


@dataclass(frozen=True)
class CaseContract:
    case_id: str
    family: str
    physical_id: str
    U_m_s: float
    frequency_Hz: float
    nominal_twist_deg: float
    aoa_deg: float
    twist_phase_deg: float
    roles: tuple[str, ...]
    coverage: str

    @property
    def solver_twist_amplitude_deg(self) -> float:
        return self.nominal_twist_deg / 2.0

    @property
    def condition_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.U_m_s,
            self.frequency_Hz,
            self.nominal_twist_deg,
            self.aoa_deg,
            self.twist_phase_deg,
        )


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(
                _jsonable(value),
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _write_npz_atomic(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_directory(path.parent)
    finally:
        if partial.exists():
            partial.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WitnessContractError(f"{path}: expected a JSON object")
    return value


@contextmanager
def _campaign_lock(output: Path) -> Iterator[None]:
    lock = output.parent / f".{output.name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WitnessContractError(
                f"another process holds the campaign lock {lock}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _case_contracts() -> tuple[CaseContract, ...]:
    cases: list[CaseContract] = []
    for physical_id, U, frequency, twist, aoa, coverage in (
        MEAN_PHYSICAL_WITNESSES
    ):
        if twist == 0.0:
            cases.append(
                CaseContract(
                    case_id=f"mean_{physical_id}_tw0_phase_equivalent",
                    family="mean",
                    physical_id=physical_id,
                    U_m_s=U,
                    frequency_Hz=frequency,
                    nominal_twist_deg=twist,
                    aoa_deg=aoa,
                    twist_phase_deg=PRODUCTION_PHASE_DEG,
                    roles=(
                        "production_plus90",
                        "experiment_identity_corrected_minus90",
                        "zero_twist_phase_equivalent",
                    ),
                    coverage=coverage,
                )
            )
            continue
        for phase, role, suffix in (
            (PRODUCTION_PHASE_DEG, "production_plus90", "plus90"),
            (
                EXPERIMENT_PHASE_DEG,
                "experiment_identity_corrected_minus90",
                "minus90",
            ),
        ):
            cases.append(
                CaseContract(
                    case_id=f"mean_{physical_id}_{suffix}",
                    family="mean",
                    physical_id=physical_id,
                    U_m_s=U,
                    frequency_Hz=frequency,
                    nominal_twist_deg=twist,
                    aoa_deg=aoa,
                    twist_phase_deg=phase,
                    roles=(role,),
                    coverage=coverage,
                )
            )

    for physical_id, U, frequency, twist, aoa, coverage in (
        FIG16_PHYSICAL_WITNESSES
    ):
        roles = ["figure16_primary_experiment_identity_minus90"]
        if twist == 0.0:
            roles.append("zero_twist_phase_equivalent")
        cases.append(
            CaseContract(
                case_id=f"phase_{physical_id}_minus90",
                family="figure16_phase",
                physical_id=physical_id,
                U_m_s=U,
                frequency_Hz=frequency,
                nominal_twist_deg=twist,
                aoa_deg=aoa,
                twist_phase_deg=EXPERIMENT_PHASE_DEG,
                roles=tuple(roles),
                coverage=coverage,
            )
        )

    identities = [case.case_id for case in cases]
    if len(identities) != len(set(identities)):
        raise WitnessContractError("duplicate case_id in witness contract")
    calls = [case.condition_tuple for case in cases]
    if len(calls) != len(set(calls)):
        raise WitnessContractError(
            "duplicate solver call survived witness de-duplication"
        )
    if len(cases) != 15:
        raise WitnessContractError(
            f"expected 15 unique solver calls, found {len(cases)}"
        )
    return tuple(cases)


def _find_claim_child(
    nodes: Sequence[Mapping[str, Any]],
    claim_id: str,
) -> Mapping[str, Any] | None:
    for node in nodes:
        if node.get("id") == claim_id:
            return node
        children = node.get("children", ())
        if isinstance(children, list):
            match = _find_claim_child(children, claim_id)
            if match is not None:
                return match
    return None


def _load_yaml(
    path: Path,
    *,
    content: bytes | None = None,
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - production dependency
        raise WitnessContractError(
            "PyYAML is required for the N5.1c gate"
        ) from exc
    if content is None:
        content = path.read_bytes()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WitnessContractError(f"{path}: claim YAML is not UTF-8") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise WitnessContractError(f"{path}: malformed claim YAML")
    return value


def _kinematic_identity_gate(
    cases: Sequence[CaseContract],
    *,
    n5_yaml: Path = N5_YAML,
    n5_yaml_bytes: bytes | None = None,
    expected_n5_yaml_sha256: str | None = None,
    source_closure_sha256: str | None = None,
) -> dict[str, Any]:
    if n5_yaml_bytes is None:
        n5_yaml_bytes = n5_yaml.read_bytes()
    n5_yaml_sha256 = _sha256_bytes(n5_yaml_bytes)
    if (
        expected_n5_yaml_sha256 is not None
        and n5_yaml_sha256 != expected_n5_yaml_sha256
    ):
        raise WitnessContractError(
            "N5 claim bytes do not match the captured source closure"
        )
    root = _load_yaml(n5_yaml, content=n5_yaml_bytes)
    claim = _find_claim_child([root], "N5.1c")
    if claim is None:
        raise WitnessContractError("N5.1c is missing")
    if claim.get("state") != "validated" or claim.get("freeze") is not True:
        raise WitnessContractError(
            "N5.1c must remain validated/frozen before this diagnostic"
        )

    upper_stop = 0.5 * math.pi
    rate_minus90 = math.cos(
        upper_stop + math.radians(EXPERIMENT_PHASE_DEG)
    )
    rate_plus90 = math.cos(
        upper_stop + math.radians(PRODUCTION_PHASE_DEG)
    )
    if not (rate_minus90 > 0.0 and rate_plus90 < 0.0):
        raise WitnessContractError("analytic Figure10 phase identity failed")

    errors: list[str] = []
    for case in cases:
        if (
            case.family == "figure16_phase"
            and case.nominal_twist_deg != 0.0
            and case.twist_phase_deg != EXPERIMENT_PHASE_DEG
        ):
            errors.append(f"{case.case_id}: Figure16 primary must be -90")
        if (
            "production_plus90" in case.roles
            and case.nominal_twist_deg != 0.0
            and case.twist_phase_deg != PRODUCTION_PHASE_DEG
        ):
            errors.append(f"{case.case_id}: production role is not +90")
        if (
            "experiment_identity_corrected_minus90" in case.roles
            and case.nominal_twist_deg != 0.0
            and case.twist_phase_deg != EXPERIMENT_PHASE_DEG
        ):
            errors.append(f"{case.case_id}: experiment role is not -90")
    if errors:
        raise WitnessContractError("; ".join(errors))

    return {
        "passed": True,
        "claim_id": "N5.1c",
        "claim_state": claim["state"],
        "claim_frozen": claim["freeze"],
        "claim_yaml_path": str(n5_yaml.relative_to(ROOT)),
        "claim_yaml_sha256": n5_yaml_sha256,
        "source_snapshot_bound": (
            expected_n5_yaml_sha256 is not None
            and source_closure_sha256 is not None
        ),
        "source_closure_sha256": source_closure_sha256,
        "paper_identity_twist_phase_deg": EXPERIMENT_PHASE_DEG,
        "production_mismatch_twist_phase_deg": PRODUCTION_PHASE_DEG,
        "upper_stop_dpsi_sign_minus90": 1,
        "upper_stop_dpsi_sign_plus90": -1,
        "figure16_primary_uses_minus90": True,
    }


def _source_paths() -> tuple[Path, ...]:
    paths = {ROOT / relative for relative in DIRECT_SOURCE_FILES}
    missing = [
        str(path.relative_to(ROOT))
        for path in sorted(paths)
        if not path.is_file()
    ]
    if missing:
        raise WitnessContractError(
            f"mandatory source-closure files are missing: {missing}"
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
    members = {
        str(path.resolve().relative_to(root.resolve())): _sha256_file(path)
        for path in sorted({path.resolve() for path in paths})
    }
    return {
        "schema": "content-addressed-source-closure-v1",
        "members": members,
        "members_sha256": _canonical_hash(members),
    }


def _source_closure() -> dict[str, Any]:
    closure, _ = _source_closure_snapshot()
    return closure


def _source_closure_snapshot() -> tuple[dict[str, Any], bytes]:
    """Capture closure hashes and N5 bytes in one read-only snapshot."""
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
        raise WitnessContractError("N5 YAML is absent from the source closure")
    closure = {
        "schema": "content-addressed-source-closure-v1",
        "members": members,
        "members_sha256": _canonical_hash(members),
    }
    closure["governed_globs"] = list(SOURCE_GLOBS)
    closure["conditional_files"] = list(CONDITIONAL_SOURCE_FILES)
    return closure, n5_yaml_bytes


def _campaign_inputs(
    cases: Sequence[CaseContract],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_closure, n5_yaml_bytes = _source_closure_snapshot()
    n5_relative = str(N5_YAML.resolve().relative_to(ROOT.resolve()))
    n5_sha256 = source_closure["members"].get(n5_relative)
    if not isinstance(n5_sha256, str):
        raise WitnessContractError("N5 YAML hash is missing from source closure")
    identity_gate = _kinematic_identity_gate(
        cases,
        n5_yaml_bytes=n5_yaml_bytes,
        expected_n5_yaml_sha256=n5_sha256,
        source_closure_sha256=source_closure["members_sha256"],
    )
    if identity_gate["source_snapshot_bound"] is not True:
        raise WitnessContractError("N5 gate is not bound to source snapshot")
    return source_closure, identity_gate


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
            "current source closure drifted: " + ", ".join(changed[:20])
        )


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


def _campaign_contract(cases: Sequence[CaseContract]) -> dict[str, Any]:
    case_payload = _jsonable([asdict(case) for case in cases])
    return {
        "schema": SCHEMA_VERSION,
        "cases": case_payload,
        "case_contract_sha256": _canonical_hash(case_payload),
        "expected_unique_witness_solver_calls": len(cases),
        "excluded_preconditioner_calls_per_session": 1,
        "expected_fresh_session_total_solver_invocations": (
            len(cases) + 1
        ),
        "resume_preconditioner_force_tolerance_N": (
            PRECONDITIONER_FORCE_TOLERANCE_N
        ),
        "old_baseline_reproduction": False,
        "closure": "v41",
        "stage": "G0_exploratory_quick_identity",
        "grid": {
            "nc": G0_NC,
            "ns": G0_NS,
            "n_cycle": G0_N_CYCLE,
            "steps_per_cycle": G0_STEPS_PER_CYCLE,
            "wake_rows": G0_STEPS_PER_CYCLE,
        },
        "production_grid_claim_allowed": False,
        "nominal_twist_to_solver_amplitude": 0.5,
        "phase_identity": {
            "figure16_primary_deg": EXPERIMENT_PHASE_DEG,
            "production_baseline_deg": PRODUCTION_PHASE_DEG,
        },
    }


def _new_campaign(
    *,
    source_closure: Mapping[str, Any],
    cases: Sequence[CaseContract],
    identity_gate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "status": "running",
        "created_at": _now(),
        "updated_at": _now(),
        "scientific_scope": (
            "read-only N1 leading-edge-suction versus N2 separated-"
            "chordwise-pressure identifiability"
        ),
        "preregistration": {
            "path": str(PREREG.relative_to(ROOT)),
            "sha256": _sha256_file(PREREG),
        },
        "source_closure": dict(source_closure),
        "source_closure_sha256": source_closure["members_sha256"],
        "git": _git_metadata(),
        "contract": _campaign_contract(cases),
        "kinematic_identity_gate": dict(identity_gate),
        "figure16_experiment": None,
        "numeric_runtime": None,
        "common_claim_graph_identity_sha256": None,
        "sessions": [],
        "cases": {},
        "failures": {},
        "active_case": None,
    }


def _case_artifact_paths(output: Path, case_id: str) -> dict[str, Path]:
    base = output / "cases" / case_id
    return {
        "raw_npz": base.with_suffix(".raw.npz"),
        "schema_json": base.with_suffix(".schema.json"),
        "evidence_json": base.with_suffix(".evidence.json"),
    }


def _validate_completed_case_artifacts(
    output: Path,
    case_id: str,
    record: Mapping[str, Any],
) -> None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise WitnessContractError(f"{case_id}: malformed resume artifact map")
    expected_paths = _case_artifact_paths(output, case_id)
    for name, expected_path in expected_paths.items():
        item = artifacts.get(name)
        if not isinstance(item, Mapping):
            raise WitnessContractError(f"{case_id}: missing {name} identity")
        relative = str(expected_path.relative_to(output))
        if item.get("path") != relative or not expected_path.is_file():
            raise WitnessContractError(f"{case_id}: missing {name} artifact")
        if item.get("sha256") != _sha256_file(expected_path):
            raise WitnessContractError(f"{case_id}: {name} hash drift")


def _open_campaign(
    output: Path,
    *,
    resume: bool,
    source_closure: Mapping[str, Any],
    cases: Sequence[CaseContract],
    identity_gate: Mapping[str, Any],
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
        campaign = _load_json(manifest_path)
        if campaign.get("schema") != SCHEMA_VERSION:
            raise WitnessContractError("resume schema mismatch")
        if campaign.get("contract") != expected_contract:
            raise WitnessContractError("resume witness contract mismatch")
        if campaign.get("source_closure") != source_closure:
            raise WitnessContractError("resume source closure mismatch")
        if campaign.get("kinematic_identity_gate") != identity_gate:
            raise WitnessContractError("resume kinematic gate mismatch")
        completed_cases = campaign.get("cases")
        if not isinstance(completed_cases, Mapping):
            raise WitnessContractError("resume case map is malformed")
        expected_case_ids = {case.case_id for case in cases}
        extra_case_ids = sorted(set(completed_cases) - expected_case_ids)
        if extra_case_ids:
            raise WitnessContractError(
                "resume contains unexpected scientific cases: "
                + ", ".join(extra_case_ids)
            )
        sessions = campaign.get("sessions")
        if not isinstance(sessions, list):
            raise WitnessContractError("resume session list is malformed")
        for session in sessions:
            if not isinstance(session, Mapping):
                raise WitnessContractError("resume session is malformed")
            session_case_ids = session.get("completed_case_ids", ())
            if not isinstance(session_case_ids, list):
                raise WitnessContractError(
                    "resume session completed_case_ids is malformed"
                )
            extra_session_ids = sorted(
                set(session_case_ids) - expected_case_ids
            )
            if extra_session_ids:
                raise WitnessContractError(
                    "resume session contains unexpected scientific cases: "
                    + ", ".join(extra_session_ids)
                )
        for case_id, record in completed_cases.items():
            _validate_completed_case_artifacts(
                output,
                case_id,
                record,
            )
        campaign["status"] = "running"
        campaign["active_case"] = None
        campaign["updated_at"] = _now()
        _write_json_atomic(manifest_path, campaign)
        return campaign

    output.mkdir(parents=True)
    campaign = _new_campaign(
        source_closure=source_closure,
        cases=cases,
        identity_gate=identity_gate,
    )
    _write_json_atomic(manifest_path, campaign)
    return campaign


def _parse_fig16_digitization(
    source: Path = FIG16_SOURCE,
) -> dict[str, np.ndarray]:
    text = source.read_text(encoding="utf-8")
    try:
        thrust_start = text.index("Figure 16. (a)")
        lift_start = text.index("Figure 16. (b)", thrust_start + 10)
    except ValueError as exc:
        raise WitnessContractError(
            f"{source}: Figure16 sections are missing"
        ) from exc

    output: dict[str, np.ndarray] = {}
    for kind, segment in (
        ("T", text[thrust_start:lift_start]),
        ("L", text[lift_start:]),
    ):
        matches = list(
            re.finditer(
                r"Twist amplitude\((\d+(?:\.\d+)?)°?\)",
                segment,
            )
        )
        for index, match in enumerate(matches):
            twist = float(match.group(1))
            block_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(segment)
            )
            block = segment[match.end():block_end]
            pairs = re.findall(
                r"(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)",
                block,
                flags=re.IGNORECASE,
            )
            if not pairs:
                raise WitnessContractError(
                    f"{source}: no Figure16 {kind}/tw{twist:g} points"
                )
            values = np.asarray(
                [(float(t), float(force) * G2N) for t, force in pairs],
                dtype=np.float64,
            )
            order = np.argsort(values[:, 0])
            values = values[order]
            tag = f"{kind}_tw{twist:g}".replace(".", "p")
            output[f"{tag}_t_over_T"] = values[:, 0]
            output[f"{tag}_force_N"] = values[:, 1]

    expected = {
        f"{kind}_tw{str(twist).replace('.', 'p')}_{suffix}"
        for kind in ("T", "L")
        for twist in (0, 22.5, 45)
        for suffix in ("t_over_T", "force_N")
    }
    if set(output) != expected:
        raise WitnessContractError(
            "Figure16 digitization fields mismatch: "
            f"missing={sorted(expected - set(output))}, "
            f"extra={sorted(set(output) - expected)}"
        )
    for name, values in output.items():
        if not np.all(np.isfinite(values)):
            raise WitnessContractError(f"Figure16 {name} is non-finite")
        if name.endswith("_t_over_T"):
            if np.any(np.diff(values) <= 0.0):
                raise WitnessContractError(
                    f"Figure16 {name} is not strictly increasing"
                )
            # Preserve the published digitization at the periodic seam.  The
            # tw0 lift trace contains one point at t/T=-0.00145; wrapping or
            # clipping it here would silently mutate the raw GT asset.
            if values[0] < -0.01 or values[-1] > 1.01:
                raise WitnessContractError(
                    f"Figure16 {name} falls outside one cycle"
                )
    return dict(sorted(output.items()))


def _array_bundle_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, raw in sorted(arrays.items()):
        value = np.ascontiguousarray(raw)
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "schema": RAW_SCHEMA_VERSION,
        "fields": {
            name: {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
            for name, value in sorted(arrays.items())
        },
    }


def _ensure_fig16_artifact(
    output: Path,
    campaign: dict[str, Any],
) -> None:
    manifest_path = output / "run_manifest.json"
    npz_path = output / "figure16_published_filtered_digitization.npz"
    schema_path = output / "figure16_published_filtered_digitization.json"
    existing = campaign.get("figure16_experiment")
    if existing is not None:
        if not isinstance(existing, Mapping):
            raise WitnessContractError("malformed Figure16 artifact identity")
        for path, key in ((npz_path, "npz"), (schema_path, "schema")):
            item = existing.get(key)
            if (
                not isinstance(item, Mapping)
                or item.get("path") != str(path.relative_to(output))
                or not path.is_file()
                or item.get("sha256") != _sha256_file(path)
            ):
                raise WitnessContractError(
                    f"Figure16 {key} artifact drift"
                )
        return

    arrays = _parse_fig16_digitization()
    schema = _array_schema(arrays)
    schema.update(
        {
            "data_role": "published_filtered_gt",
            "source": str(FIG16_SOURCE.relative_to(ROOT)),
            "source_sha256": _sha256_file(FIG16_SOURCE),
            "force_conversion": "published grams-force * 9.81 / 1000 -> N",
            "published_processing": (
                "5th-order Butterworth, 8 Hz; instrument raw unavailable"
            ),
            "runner_processing": "none",
            "refilter_digitization": False,
            "alignment_status": "unresolved_external_kinematics",
            "model_force_cross_correlation_allowed": False,
            "array_bundle_sha256": _array_bundle_hash(arrays),
        }
    )
    _write_npz_atomic(npz_path, arrays)
    _write_json_atomic(schema_path, schema)
    campaign["figure16_experiment"] = {
        "npz": {
            "path": str(npz_path.relative_to(output)),
            "sha256": _sha256_file(npz_path),
        },
        "schema": {
            "path": str(schema_path.relative_to(output)),
            "sha256": _sha256_file(schema_path),
        },
        "array_bundle_sha256": schema["array_bundle_sha256"],
        "data_role": "published_filtered_gt",
        "alignment_status": "unresolved_external_kinematics",
    }
    campaign["updated_at"] = _now()
    _write_json_atomic(manifest_path, campaign)


def _flatten_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    leaves: dict[str, list[Any]] = {}

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                visit(child, item)
            return
        leaves.setdefault(prefix, []).append(value)

    for record in records:
        visit("", record)
    expected = len(records)
    malformed = [
        name for name, values in leaves.items() if len(values) != expected
    ]
    if malformed:
        raise WitnessContractError(
            f"inconsistent raw fields across steps: {malformed[:10]}"
        )
    return {
        name: np.asarray(values)
        for name, values in sorted(leaves.items())
    }


def _wind_lift(force: np.ndarray, aoa_deg: float) -> np.ndarray:
    angle = math.radians(aoa_deg)
    return force[..., 2] * math.cos(angle) - force[..., 0] * math.sin(angle)


def _wind_thrust(force: np.ndarray, aoa_deg: float) -> np.ndarray:
    angle = math.radians(aoa_deg)
    return -(
        force[..., 0] * math.cos(angle)
        + force[..., 2] * math.sin(angle)
    )


def _last_series(
    result: Mapping[str, Any],
    x_name: str,
    z_name: str,
    steps_per_cycle: int,
) -> np.ndarray:
    x_values = np.asarray(result[x_name], dtype=np.float64)
    z_values = np.asarray(result[z_name], dtype=np.float64)
    if (
        x_values.ndim != 1
        or z_values.shape != x_values.shape
        or x_values.size < steps_per_cycle
    ):
        raise WitnessContractError(
            f"malformed result series {x_name}/{z_name}"
        )
    output = np.zeros((steps_per_cycle, 3), dtype=np.float64)
    output[:, 0] = x_values[-steps_per_cycle:]
    output[:, 2] = z_values[-steps_per_cycle:]
    return output


def _augment_raw_bundle(
    flat: dict[str, np.ndarray],
    result: Mapping[str, Any],
    case: CaseContract,
    steps_per_cycle: int,
) -> dict[str, np.ndarray]:
    output = dict(flat)
    n1_suction = _last_series(
        result,
        "Xh_les",
        "Lh_les",
        steps_per_cycle,
    )
    output[
        "diagnostic.n1.leading_edge_suction_"
        "solver_accumulator_body_force_N"
    ] = n1_suction

    n2_booked = np.asarray(
        flat["n2.separation_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    n2_candidate = np.sum(
        np.asarray(
            flat["n2.separation_panel_candidate_force_body_N"],
            dtype=np.float64,
        ),
        axis=1,
    )
    output[
        "diagnostic.n2.separation_panel_candidate_resultant_body_force_N"
    ] = n2_candidate
    output[
        "diagnostic.n2.candidate_minus_booked_body_force_N"
    ] = n2_candidate - n2_booked

    for prefix, body_force in (
        ("n1_leading_edge_suction", 2.0 * n1_suction),
        ("n2_separation_booked", 2.0 * n2_booked),
        ("n2_separation_panel_candidate", 2.0 * n2_candidate),
    ):
        output[f"diagnostic.wind.{prefix}_L_N"] = _wind_lift(
            body_force,
            case.aoa_deg,
        )
        output[f"diagnostic.wind.{prefix}_T_N"] = _wind_thrust(
            body_force,
            case.aoa_deg,
        )

    phase_solver = np.asarray(flat["phase_solver_rad"], dtype=np.float64)
    phase_paper = np.asarray(flat["phase_paper_rad"], dtype=np.float64)
    output["diagnostic.phase_solver_t_over_T"] = np.mod(
        phase_solver / (2.0 * math.pi),
        1.0,
    )
    output["diagnostic.phase_paper_t_over_T"] = np.mod(
        phase_paper / (2.0 * math.pi),
        1.0,
    )
    output["diagnostic.alignment_source_code"] = np.full(
        steps_per_cycle,
        "unresolved_external_kinematics",
    )
    return output


def _claim_graph_identity_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    closure = manifest.get("closure")
    topology = manifest.get("topology")
    nodes = manifest.get("nodes")
    sources = manifest.get("parameter_sources")
    if closure != "v41":
        raise WitnessContractError(
            f"claim manifest closure must be v41, got {closure!r}"
        )
    if (
        not isinstance(topology, list)
        or not topology
        or len(topology) != len(set(topology))
    ):
        raise WitnessContractError("claim manifest topology is malformed")
    if not isinstance(nodes, list) or len(nodes) != len(topology):
        raise WitnessContractError("claim manifest nodes are malformed")
    identity_fields = (
        "id",
        "state",
        "freeze",
        "runtime_role",
        "implementation",
        "implementation_version",
        "implementation_hash",
    )
    normalized = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise WitnessContractError("claim manifest node is malformed")
        item = {field: node.get(field) for field in identity_fields}
        if item["id"] != topology[index]:
            raise WitnessContractError("claim node order differs from topology")
        normalized.append(item)
    if not isinstance(sources, Mapping):
        raise WitnessContractError("claim parameter sources are malformed")
    return {
        "closure": closure,
        "topology": topology,
        "nodes": normalized,
        "parameter_sources": dict(sorted(sources.items())),
    }


def _claim_graph_identity_sha256(
    manifest: Mapping[str, Any],
) -> str:
    return _canonical_hash(_claim_graph_identity_payload(manifest))


def _claim_channel_force(
    contributions: Mapping[str, Any],
    channel: str,
) -> np.ndarray:
    matches: list[np.ndarray] = []
    for raw_items in contributions.values():
        if not isinstance(raw_items, list):
            raise WitnessContractError("malformed claim contribution list")
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise WitnessContractError("malformed claim contribution")
            if raw.get("channel") == channel:
                force = np.asarray(raw.get("body_force"), dtype=np.float64)
                if force.shape != (3,) or not np.all(np.isfinite(force)):
                    raise WitnessContractError(
                        f"invalid {channel} claim force"
                    )
                matches.append(force)
    if len(matches) != 1:
        raise WitnessContractError(
            f"expected one {channel!r} provider, found {len(matches)}"
        )
    return matches[0]


def _claim_ledger_total(
    contributions: Mapping[str, Any],
) -> np.ndarray:
    total = np.zeros(3, dtype=np.float64)
    for raw_items in contributions.values():
        if not isinstance(raw_items, list):
            raise WitnessContractError("malformed claim contribution list")
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise WitnessContractError("malformed claim contribution")
            force = np.asarray(raw.get("body_force"), dtype=np.float64)
            if force.shape != (3,) or not np.all(np.isfinite(force)):
                raise WitnessContractError("invalid claim ledger force")
            total += force
    return total


def _validate_claim_guards(guards: Any) -> None:
    if not isinstance(guards, Mapping):
        raise WitnessContractError("claim_guards is missing")
    failed = [
        name
        for name in REQUIRED_CLAIM_GUARDS
        if not isinstance(guards.get(name), Mapping)
        or guards[name].get("passed") is not True
    ]
    if failed:
        raise WitnessContractError(f"claim guards failed: {failed}")


def _maximum_abs(value: np.ndarray) -> float:
    return float(np.max(np.abs(value), initial=0.0))


def _raw_guard_report(
    *,
    bundle: Mapping[str, np.ndarray],
    result: Mapping[str, Any],
    case: CaseContract,
    steps_per_cycle: int,
) -> dict[str, Any]:
    required = (
        "last_cycle_step",
        "cycle_index",
        "step",
        "time_s",
        "dt_s",
        "phase_solver_rad",
        "n1.panel_force_body_N",
        "n1.bernoulli_booked_solver_accumulator_N",
        "n1.booked_solver_accumulator_total_N",
        "n2.separation_panel_candidate_force_body_N",
        "n2.separation_booked_solver_accumulator_N",
        "n2.booked_solver_accumulator_total_N",
        "n3.ds_panel_force_solver_legacy_N",
        "n3.ds_booked_solver_accumulator_N",
        "n3.booked_solver_accumulator_total_N",
        "total_solver_accumulator_body_force_N",
        "reported_pair_wind_lift_N",
        "reported_pair_wind_thrust_N",
        (
            "diagnostic.n1.leading_edge_suction_"
            "solver_accumulator_body_force_N"
        ),
    )
    missing = [name for name in required if name not in bundle]
    if missing:
        raise WitnessContractError(
            f"{case.case_id}: raw fields missing: {missing}"
        )

    finite = all(
        np.isfinite(value).all()
        for value in bundle.values()
        if value.dtype.kind in "biufc"
    )
    expected_steps = np.arange(steps_per_cycle)
    step_identity = np.array_equal(
        bundle["last_cycle_step"],
        expected_steps,
    )
    cycle_identity = (
        np.array_equal(
            bundle["cycle_index"],
            bundle["step"] // steps_per_cycle,
        )
        and np.unique(bundle["cycle_index"]).size == 1
    )
    dt = np.asarray(bundle["dt_s"], dtype=np.float64)
    time_values = np.asarray(bundle["time_s"], dtype=np.float64)
    time_step_error = _maximum_abs(np.diff(time_values) - dt[:-1])

    n1_panel = np.asarray(
        bundle["n1.panel_force_body_N"],
        dtype=np.float64,
    )
    n1_bernoulli = np.asarray(
        bundle["n1.bernoulli_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    n1_panel_error = _maximum_abs(
        np.sum(n1_panel, axis=1)[..., (0, 2)]
        - n1_bernoulli[..., (0, 2)]
    )

    n3_panel = np.asarray(
        bundle["n3.ds_panel_force_solver_legacy_N"],
        dtype=np.float64,
    )
    n3_booked = np.asarray(
        bundle["n3.ds_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    n3_panel_error = _maximum_abs(
        np.sum(n3_panel, axis=1)[..., (0, 2)]
        - n3_booked[..., (0, 2)]
    )

    reconstructed = (
        np.asarray(
            bundle["n1.booked_solver_accumulator_total_N"],
            dtype=np.float64,
        )
        + np.asarray(
            bundle["n2.booked_solver_accumulator_total_N"],
            dtype=np.float64,
        )
        + np.asarray(
            bundle["n3.booked_solver_accumulator_total_N"],
            dtype=np.float64,
        )
    )
    raw_total = np.asarray(
        bundle["total_solver_accumulator_body_force_N"],
        dtype=np.float64,
    )
    # N1 is deliberately recorded in _v2_robo.py as total - N2 - N3.  This is
    # therefore an algebraic recorder-consistency identity, not an independent
    # physical closure and never evidence for choosing N1 versus N2.
    raw_node_algebraic_identity_error = _maximum_abs(
        reconstructed - raw_total
    )

    model_lift = np.asarray(result["L_inst"], dtype=np.float64)[
        -steps_per_cycle:
    ]
    model_thrust = np.asarray(result["T_inst"], dtype=np.float64)[
        -steps_per_cycle:
    ]
    lift_trace_error = _maximum_abs(
        np.asarray(
            bundle["reported_pair_wind_lift_N"],
            dtype=np.float64,
        )
        - model_lift
    )
    thrust_trace_error = _maximum_abs(
        np.asarray(
            bundle["reported_pair_wind_thrust_N"],
            dtype=np.float64,
        )
        - model_thrust
    )

    n2_booked_pair = 2.0 * np.asarray(
        bundle["n2.separation_booked_solver_accumulator_N"],
        dtype=np.float64,
    )
    n2_booked_thrust = _wind_thrust(
        n2_booked_pair,
        case.aoa_deg,
    )
    n2_zero_thrust_error = _maximum_abs(n2_booked_thrust)

    manifest = result.get("claim_manifest")
    contributions = result.get("claim_contributions")
    guards = result.get("claim_guards")
    if not isinstance(manifest, Mapping):
        raise WitnessContractError("claim_manifest is missing")
    if not isinstance(contributions, Mapping):
        raise WitnessContractError("claim_contributions is missing")
    _validate_claim_guards(guards)
    if manifest.get("guards") != guards:
        raise WitnessContractError(
            "claim manifest guards differ from claim_guards"
        )

    graph_total = _claim_ledger_total(contributions)
    graph_target = np.asarray(
        [float(result["Fx_body"]), 0.0, float(result["Fz_body"])],
        dtype=np.float64,
    )
    graph_ledger_error = _maximum_abs(graph_total - graph_target)

    n1_suction = np.asarray(
        bundle[
            "diagnostic.n1.leading_edge_suction_"
            "solver_accumulator_body_force_N"
        ],
        dtype=np.float64,
    )
    n1_suction_mean_error = _maximum_abs(
        2.0 * np.mean(n1_suction, axis=0)
        - _claim_channel_force(contributions, "leading_edge_suction")
    )
    n2_separation_mean_error = _maximum_abs(
        2.0
        * np.mean(
            np.asarray(
                bundle["n2.separation_booked_solver_accumulator_N"],
                dtype=np.float64,
            ),
            axis=0,
        )
        - _claim_channel_force(contributions, "separation")
    )
    n3_ds_mean_error = _maximum_abs(
        2.0 * np.mean(n3_booked, axis=0)
        - _claim_channel_force(contributions, "ds_vortex")
    )

    raw_config = result.get("claim_raw_config")
    if not isinstance(raw_config, Mapping):
        raise WitnessContractError("claim_raw_config is missing")
    expected_config = {
        "closure": "v41",
        "nc": G0_NC,
        "ns": G0_NS,
        "n_cycle": G0_N_CYCLE,
        "steps_per_cycle": G0_STEPS_PER_CYCLE,
        "wake_rows": G0_STEPS_PER_CYCLE,
        "U_m_s": case.U_m_s,
        "aoa_deg": case.aoa_deg,
        "freq_hz": case.frequency_Hz,
        "twist_amp_deg": case.solver_twist_amplitude_deg,
        "twist_phase_deg": case.twist_phase_deg,
    }
    config_errors = {
        key: {"expected": expected, "actual": raw_config.get(key)}
        for key, expected in expected_config.items()
        if raw_config.get(key) != expected
    }

    numeric_errors = {
        "time_step_error_s": time_step_error,
        "n1_panel_to_bernoulli_error_N": n1_panel_error,
        "n3_panel_to_booked_error_N": n3_panel_error,
        (
            "n1_plus_n2_plus_n3_raw_algebraic_"
            "identity_error_N"
        ): raw_node_algebraic_identity_error,
        "reported_lift_trace_error_N": lift_trace_error,
        "reported_thrust_trace_error_N": thrust_trace_error,
        "n2_booked_wind_thrust_identity_error_N": n2_zero_thrust_error,
        "graph_force_ledger_error_N": graph_ledger_error,
        "n1_suction_mean_to_graph_error_N": n1_suction_mean_error,
        "n2_separation_mean_to_graph_error_N": n2_separation_mean_error,
        "n3_ds_mean_to_graph_error_N": n3_ds_mean_error,
    }
    ledger_errors = (
        n1_panel_error,
        n3_panel_error,
        raw_node_algebraic_identity_error,
        lift_trace_error,
        thrust_trace_error,
        n2_zero_thrust_error,
        n1_suction_mean_error,
        n2_separation_mean_error,
        n3_ds_mean_error,
    )
    passed = (
        len(bundle["last_cycle_step"]) == steps_per_cycle
        and finite
        and step_identity
        and cycle_identity
        and not config_errors
        and max(ledger_errors, default=0.0) <= LEDGER_TOLERANCE_N
        and graph_ledger_error <= GRAPH_LEDGER_TOLERANCE_N
    )
    report = {
        "passed": bool(passed),
        "n_steps": int(len(bundle["last_cycle_step"])),
        "finite": bool(finite),
        "step_identity": bool(step_identity),
        "cycle_identity": bool(cycle_identity),
        "config_errors": config_errors,
        **numeric_errors,
    }
    if not passed:
        raise WitnessContractError(
            f"{case.case_id}: raw/ledger guard failed: {report}"
        )
    return report


def _diagnostic_summary(
    bundle: Mapping[str, np.ndarray],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    names = (
        "n1_leading_edge_suction",
        "n2_separation_booked",
        "n2_separation_panel_candidate",
    )
    channels = {
        name: {
            "L_N": float(
                np.mean(bundle[f"diagnostic.wind.{name}_L_N"])
            ),
            "T_N": float(
                np.mean(bundle[f"diagnostic.wind.{name}_T_N"])
            ),
        }
        for name in names
    }
    channels["n2_candidate_minus_booked"] = {
        axis: (
            channels["n2_separation_panel_candidate"][axis]
            - channels["n2_separation_booked"][axis]
        )
        for axis in ("L_N", "T_N")
    }
    return {
        "processing": "no clipping, filtering, or phase alignment",
        "n2_panel_candidate_interpretation": (
            "falsified legacy full-vector internal counterfactual; "
            "direction/phase/collinearity observer only, not missing force"
        ),
        "reported_robust_cycle_force": {
            "L_N": float(result["L_wind"]),
            "T_N": float(result["T_wind"]),
        },
        "raw_cycle_mean_total": {
            "L_N": float(
                np.mean(bundle["reported_pair_wind_lift_N"])
            ),
            "T_N": float(
                np.mean(bundle["reported_pair_wind_thrust_N"])
            ),
        },
        "raw_cycle_mean_channels": channels,
    }


def _resolved_call(
    solver: Callable[..., Mapping[str, Any]],
    call: Mapping[str, Any],
) -> dict[str, Any]:
    bound = inspect.signature(solver).bind_partial(**call)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    resolved.pop("claim_raw_out", None)
    resolved.pop("frames_out", None)
    return _jsonable(resolved)


def _build_solver_call(case: CaseContract) -> dict[str, Any]:
    _prepend_solver_import_paths()
    from lb_sweep118 import BASE

    call = dict(BASE)
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
        wake_rows=G0_STEPS_PER_CYCLE,
        closure="v41",
    )
    if call["closure"] != "v41":
        raise WitnessContractError("candidate closure is forbidden")
    return call


def _execute_case(
    solver: Callable[..., Mapping[str, Any]],
    case: CaseContract,
    *,
    call_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    call = (
        _build_solver_call(case)
        if call_override is None
        else dict(call_override)
    )
    raw: list[dict[str, Any]] = []
    started = time.time()
    result = solver(**call, claim_raw_out=raw)
    if not isinstance(result, Mapping):
        raise WitnessContractError(
            f"{case.case_id}: solver returned a non-mapping"
        )
    if len(raw) != G0_STEPS_PER_CYCLE:
        raise WitnessContractError(
            f"{case.case_id}: expected {G0_STEPS_PER_CYCLE} raw steps, "
            f"got {len(raw)}"
        )
    flat = _flatten_records(raw)
    bundle = _augment_raw_bundle(
        flat,
        result,
        case,
        G0_STEPS_PER_CYCLE,
    )
    guard = _raw_guard_report(
        bundle=bundle,
        result=result,
        case=case,
        steps_per_cycle=G0_STEPS_PER_CYCLE,
    )
    manifest = result["claim_manifest"]
    contributions = result["claim_contributions"]
    evidence = {
        "schema": SCHEMA_VERSION,
        "case_contract": asdict(case),
        "stage": "G0_exploratory_quick_identity",
        "production_grid_claim_allowed": False,
        "resolved_call": _resolved_call(solver, call),
        "source_closure_sha256": None,
        "claim_graph_identity_sha256": (
            _claim_graph_identity_sha256(manifest)
        ),
        "claim_manifest_sha256": _canonical_hash(manifest),
        "claim_manifest": dict(manifest),
        "claim_guards": dict(result["claim_guards"]),
        "claim_contributions": dict(contributions),
        "claim_raw_config": dict(result["claim_raw_config"]),
        "raw_guard": guard,
        "diagnostic_summary": _diagnostic_summary(bundle, result),
        "figure16_alignment": {
            "status": "unresolved_external_kinematics",
            "model_force_cross_correlation_allowed": False,
            "model_processing": "raw",
        },
        "n2_panel_candidate_contract": {
            "role": "internal_counterfactual_support_observer",
            "legacy_full_vector_route_state": "falsified",
            "allowed_uses": [
                "phase",
                "direction",
                "collinearity_with_n1_leading_edge_suction",
            ],
            "forbidden_uses": [
                "interpret_newton_magnitude_as_N2.5_or_N2.6_missing_force",
                "add_to_production_force",
                "construct_candidate_from_candidate_minus_booked_magnitude",
            ],
        },
        "observer_role": "read_only",
        "aerodynamic_formula_modified": False,
        "force_added_by_runner": False,
        "wall_s": time.time() - started,
    }
    return bundle, evidence


def _prepend_solver_import_paths() -> tuple[str, str]:
    """Make the current checkout authoritative for all solver imports."""
    required = (SRC.resolve(), PLATFORM.resolve())
    for path in required:
        if not path.is_dir():
            raise WitnessContractError(
                f"required import root is missing: {path}"
            )

    filtered: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.curdir).resolve()
        except (OSError, RuntimeError):
            filtered.append(entry)
            continue
        if resolved not in required:
            filtered.append(entry)
    sys.path[:] = [str(path) for path in required] + filtered
    return str(required[0]), str(required[1])


def _module_file_identity(
    module: Any,
    *,
    module_name: str,
    required_root: Path | None = None,
) -> dict[str, Any]:
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str):
        raise WitnessContractError(f"{module_name} has no import file")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise WitnessContractError(
            f"{module_name} import file is missing: {path}"
        )
    identity: dict[str, Any] = {
        "module": module_name,
        "path": str(path),
        "sha256": _sha256_file(path),
    }
    if required_root is not None:
        root = required_root.resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise WitnessContractError(
                f"{module_name} resolved outside {root}: {path}"
            ) from exc
        identity["required_root"] = str(root)
        identity["relative_path"] = str(relative)
    return identity


def _numpy_build_identity() -> dict[str, Any]:
    module_identity = _module_file_identity(
        np,
        module_name="numpy",
    )
    try:
        build_config = np.show_config(mode="dicts")
    except TypeError:  # pragma: no cover - NumPy < 2 fallback
        stream = io.StringIO()
        with redirect_stdout(stream):
            np.show_config()
        build_config = {"show_config": stream.getvalue()}

    core_identity: dict[str, Any] | None = None
    for module_name in (
        "numpy._core._multiarray_umath",
        "numpy.core._multiarray_umath",
    ):
        try:
            core = importlib.import_module(module_name)
            core_identity = _module_file_identity(
                core,
                module_name=module_name,
            )
            break
        except (ImportError, WitnessContractError):
            continue
    identity = {
        "version": np.__version__,
        "module": module_identity,
        "core_extension": core_identity,
        "build_config": _jsonable(build_config),
    }
    identity["build_sha256"] = _canonical_hash(identity)
    return identity


def _optional_runtime_call(module: Any, name: str) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        return None
    try:
        return _jsonable(function())
    except Exception as exc:  # pragma: no cover - backend-specific
        return {"error": f"{type(exc).__name__}: {exc}"}


def _warp_runtime_identity(wp: Any, device: Any) -> dict[str, Any]:
    warp_module = _module_file_identity(wp, module_name="warp")
    device_arch = getattr(device, "arch", None)
    device_identity = {
        "text": str(device),
        "alias": getattr(device, "alias", None),
        "name": getattr(device, "name", None),
        "ordinal": getattr(device, "ordinal", None),
        "is_cuda": getattr(device, "is_cuda", None),
        "arch": device_arch,
        "compute_arch": (
            f"sm_{device_arch}" if device_arch not in (None, 0) else None
        ),
        "uuid": getattr(device, "uuid", None),
        "pci_bus_id": getattr(device, "pci_bus_id", None),
    }
    config = getattr(wp, "config", None)
    config_identity = {
        name: _jsonable(getattr(config, name, None))
        for name in (
            "version",
            "_git_commit_hash",
            "cuda_arch_suffix",
            "llvm_cuda",
            "verify_cuda",
            "fast_math",
            "mode",
        )
    }
    return {
        "version": getattr(wp, "__version__", None),
        "module": warp_module,
        "native_version": _optional_runtime_call(
            wp,
            "get_warp_version",
        ),
        "clang_version": _optional_runtime_call(
            wp,
            "get_warp_clang_version",
        ),
        "llvm_version": _optional_runtime_call(wp, "get_llvm_version"),
        "host_compiler_version": _optional_runtime_call(
            wp,
            "get_host_compiler_version",
        ),
        "cuda_available": _optional_runtime_call(
            wp,
            "is_cuda_available",
        ),
        "cuda_driver_version": _optional_runtime_call(
            wp,
            "get_cuda_driver_version",
        ),
        "cuda_toolkit_version": _optional_runtime_call(
            wp,
            "get_cuda_toolkit_version",
        ),
        "nvrtc_version": _optional_runtime_call(
            wp,
            "get_nvrtc_version",
        ),
        "cuda_supported_archs": _optional_runtime_call(
            wp,
            "get_cuda_supported_archs",
        ),
        "config": config_identity,
        "device": device_identity,
    }


def _solver_config_identity(config: Any) -> dict[str, Any]:
    dtype_name = getattr(config, "dtype_name", None)
    return {
        "dtype_name": (
            dtype_name() if callable(dtype_name) else None
        ),
        "dtype": str(getattr(config, "DTYPE", None)),
        "numpy_dtype": str(np.dtype(getattr(config, "NP_DTYPE"))),
        "device": str(getattr(config, "DEVICE", None)),
    }


def _numeric_runtime_fingerprint(
    *,
    wp: Any,
    device: Any,
    fluxvortex_module: Any,
    solver_config: Any,
) -> dict[str, Any]:
    environment_names = (
        "FLUXV_DTYPE",
        "FLUXV_DEVICE",
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    runtime = {
        "python": {
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
            "implementation": py_platform.python_implementation(),
        },
        "platform": py_platform.platform(),
        "environment": {
            name: os.environ.get(name) for name in environment_names
        },
        "numpy": _numpy_build_identity(),
        "warp_runtime": _warp_runtime_identity(wp, device),
        "solver_config": _solver_config_identity(solver_config),
        "fluxvortex_module": _module_file_identity(
            fluxvortex_module,
            module_name="fluxvortex",
            required_root=SRC,
        ),
        "authoritative_import_roots": [str(SRC.resolve()), str(PLATFORM)],
    }
    runtime["fingerprint_sha256"] = _canonical_hash(runtime)
    return runtime


def _load_solver() -> tuple[Callable[..., Mapping[str, Any]], dict[str, Any]]:
    _prepend_solver_import_paths()
    wp = importlib.import_module("warp")
    wp.init()
    fluxvortex_module = importlib.import_module("fluxvortex")
    _module_file_identity(
        fluxvortex_module,
        module_name="fluxvortex",
        required_root=SRC,
    )
    solver_module = importlib.import_module("_v2_robo")
    solver_config = importlib.import_module("fluxvortex.warp_fsi.config")
    gpu_run_twist = solver_module.gpu_run_twist
    device = wp.get_device(str(solver_config.DEVICE))
    runtime = _numeric_runtime_fingerprint(
        wp=wp,
        device=device,
        fluxvortex_module=fluxvortex_module,
        solver_config=solver_config,
    )
    return gpu_run_twist, runtime


def _register_numeric_runtime(
    campaign: dict[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one numerical runtime across every session in a campaign."""
    normalized = _jsonable(dict(runtime))
    previous = campaign.get("numeric_runtime")
    if previous is not None and previous != normalized:
        raise WitnessContractError(
            "resume numeric runtime mismatch; refusing to mix "
            "Python/NumPy/Warp/GPU/thread environments"
        )
    campaign["numeric_runtime"] = normalized
    return normalized


def _run_preconditioner(
    solver: Callable[..., Mapping[str, Any]],
    source_closure: Mapping[str, Any],
) -> dict[str, Any]:
    case = CaseContract(
        case_id="excluded_current_source_preconditioner",
        family="excluded_preconditioner",
        physical_id="P0",
        U_m_s=8.0,
        frequency_Hz=2.6,
        nominal_twist_deg=0.0,
        aoa_deg=5.0,
        twist_phase_deg=PRODUCTION_PHASE_DEG,
        roles=("excluded_numeric_runtime_preconditioner",),
        coverage="none",
    )
    _assert_source_closure(source_closure)
    call = _build_solver_call(case)
    started = time.time()
    result = solver(**call)
    _assert_source_closure(source_closure)
    _validate_claim_guards(result.get("claim_guards"))
    manifest = result.get("claim_manifest")
    if not isinstance(manifest, Mapping):
        raise WitnessContractError(
            "preconditioner claim manifest is missing"
        )
    return {
        "started_at": _now(),
        "purpose": (
            "current-process numerical preconditioning; excluded from "
            "scientific witnesses"
        ),
        "excluded_from_scientific_metrics": True,
        "case_contract": asdict(case),
        "resolved_call": _resolved_call(solver, call),
        "L_wind_N": float(result["L_wind"]),
        "T_wind_N": float(result["T_wind"]),
        "claim_graph_identity_sha256": (
            _claim_graph_identity_sha256(manifest)
        ),
        "claim_guards": dict(result["claim_guards"]),
        "wall_s": time.time() - started,
    }


def _preconditioner_resume_gate(
    campaign: Mapping[str, Any],
    current_session: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every resumed process to the first session's force/graph anchor."""
    sessions = campaign.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise WitnessContractError("campaign has no session anchor")
    current = current_session.get("preconditioner")
    if not isinstance(current, Mapping):
        raise WitnessContractError("current preconditioner is missing")
    reference_session = sessions[0]
    if not isinstance(reference_session, Mapping):
        raise WitnessContractError("first session is malformed")
    reference = reference_session.get("preconditioner")
    if reference_session is current_session:
        reference = current
    if not isinstance(reference, Mapping):
        raise WitnessContractError(
            "first-session preconditioner is unavailable"
        )

    force_deltas: dict[str, float] = {}
    for key in ("L_wind_N", "T_wind_N"):
        try:
            reference_value = float(reference[key])
            current_value = float(current[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise WitnessContractError(
                f"preconditioner {key} is malformed"
            ) from exc
        if not (
            math.isfinite(reference_value)
            and math.isfinite(current_value)
        ):
            raise WitnessContractError(
                f"preconditioner {key} is non-finite"
            )
        force_deltas[key] = abs(current_value - reference_value)

    reference_graph = reference.get("claim_graph_identity_sha256")
    current_graph = current.get("claim_graph_identity_sha256")
    if (
        not isinstance(reference_graph, str)
        or not isinstance(current_graph, str)
    ):
        raise WitnessContractError(
            "preconditioner claim graph identity is malformed"
        )
    if current_graph != reference_graph:
        raise WitnessContractError(
            "resume preconditioner claim graph differs from first session"
        )
    failed_forces = {
        key: delta
        for key, delta in force_deltas.items()
        if delta > PRECONDITIONER_FORCE_TOLERANCE_N
    }
    if failed_forces:
        raise WitnessContractError(
            "resume preconditioner force drift exceeds "
            f"tau_F={PRECONDITIONER_FORCE_TOLERANCE_N:.2f} N: "
            f"{failed_forces}"
        )
    return {
        "passed": True,
        "reference_session_index": 0,
        "reference_is_current_session": (
            reference_session is current_session
        ),
        "tau_F_N": PRECONDITIONER_FORCE_TOLERANCE_N,
        "force_abs_delta_N": force_deltas,
        "claim_graph_identity_sha256": current_graph,
    }


def _artifact_identity(path: Path, output: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(output)),
        "sha256": _sha256_file(path),
    }


def _save_case_artifacts(
    output: Path,
    case: CaseContract,
    bundle: Mapping[str, np.ndarray],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    paths = _case_artifact_paths(output, case.case_id)
    schema = _array_schema(bundle)
    schema.update(
        {
            "case_id": case.case_id,
            "stage": "G0_exploratory_quick_identity",
            "snapshot_phase": "post_force_pre_shed",
            "time_window": "last_cycle",
            "processing": "none",
            "force_scopes": {
                "raw_solver_accumulator": (
                    "one solved half-wing mesh before reported x2 mirror"
                ),
                "diagnostic_wind_channels": "reported mirror pair",
                "claim_contributions": "reported both-wing body force",
            },
            "n2_separation_panel_candidate_contract": {
                "role": "internal_counterfactual_support_observer",
                "legacy_full_vector_route_state": "falsified",
                "newton_magnitude_is_missing_force": False,
                "allowed_interpretation": (
                    "phase, direction, and collinearity only"
                ),
            },
            "figure16_alignment_status": (
                "unresolved_external_kinematics"
            ),
            "array_bundle_sha256": _array_bundle_hash(bundle),
        }
    )
    _write_npz_atomic(paths["raw_npz"], bundle)
    _write_json_atomic(paths["schema_json"], schema)
    evidence["raw_array_bundle_sha256"] = schema[
        "array_bundle_sha256"
    ]
    _write_json_atomic(paths["evidence_json"], evidence)
    return {
        "artifacts": {
            name: _artifact_identity(path, output)
            for name, path in paths.items()
        },
        "raw_array_bundle_sha256": schema[
            "array_bundle_sha256"
        ],
        "claim_graph_identity_sha256": evidence[
            "claim_graph_identity_sha256"
        ],
        "completed_at": _now(),
    }


def run(output: Path, *, resume: bool = False) -> int:
    output = output.resolve()
    cases = _case_contracts()
    source_closure, identity_gate = _campaign_inputs(cases)

    with _campaign_lock(output):
        campaign = _open_campaign(
            output,
            resume=resume,
            source_closure=source_closure,
            cases=cases,
            identity_gate=identity_gate,
        )
        manifest_path = output / "run_manifest.json"
        _assert_source_closure(source_closure)
        _ensure_fig16_artifact(output, campaign)
        _assert_source_closure(source_closure)

        solver, runtime = _load_solver()
        try:
            runtime = _register_numeric_runtime(campaign, runtime)
        except WitnessContractError as exc:
            campaign["status"] = "failed"
            campaign["failures"]["__numeric_runtime__"] = (
                f"{type(exc).__name__}: {exc}"
            )
            campaign["updated_at"] = _now()
            _write_json_atomic(manifest_path, campaign)
            raise
        session: dict[str, Any] = {
            "started_at": _now(),
            "process_id": os.getpid(),
            "source_closure_sha256": source_closure["members_sha256"],
            "numeric_runtime": runtime,
            "preconditioner": None,
            "preconditioner_resume_gate": None,
            "completed_case_ids": [],
        }
        campaign["sessions"].append(session)
        campaign["updated_at"] = _now()
        _write_json_atomic(manifest_path, campaign)

        try:
            print(
                "[n1-n2-witness] excluded current-source preconditioner",
                flush=True,
            )
            session["preconditioner"] = _run_preconditioner(
                solver,
                source_closure,
            )
            session["preconditioner_resume_gate"] = (
                _preconditioner_resume_gate(campaign, session)
            )
            preconditioner_graph = session["preconditioner"][
                "claim_graph_identity_sha256"
            ]
            _write_json_atomic(manifest_path, campaign)

            common_graph = campaign.get(
                "common_claim_graph_identity_sha256"
            )
            for index, case in enumerate(cases, start=1):
                if case.case_id in campaign["cases"]:
                    print(
                        f"[n1-n2-witness] resume skip {case.case_id}",
                        flush=True,
                    )
                    continue
                _assert_source_closure(source_closure)
                campaign["active_case"] = case.case_id
                campaign["updated_at"] = _now()
                _write_json_atomic(manifest_path, campaign)
                print(
                    f"[n1-n2-witness] {index}/{len(cases)} "
                    f"{case.case_id}",
                    flush=True,
                )
                bundle, evidence = _execute_case(solver, case)
                _assert_source_closure(source_closure)
                evidence["source_closure_sha256"] = source_closure[
                    "members_sha256"
                ]
                graph_identity = evidence[
                    "claim_graph_identity_sha256"
                ]
                if graph_identity != preconditioner_graph:
                    raise WitnessContractError(
                        f"{case.case_id}: claim graph differs from the "
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

                record = _save_case_artifacts(
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
                _write_json_atomic(manifest_path, campaign)
                summary = evidence["diagnostic_summary"]
                print(
                    f"[n1-n2-witness] {case.case_id}: "
                    f"L={summary['reported_robust_cycle_force']['L_N']:+.3f} "
                    f"T={summary['reported_robust_cycle_force']['T_N']:+.3f}",
                    flush=True,
                )

            _assert_source_closure(source_closure)
            campaign["status"] = "complete"
            campaign["active_case"] = None
            campaign["completed_case_count"] = len(campaign["cases"])
            campaign["production_grid_claim_allowed"] = False
            campaign["next_gate"] = (
                "analyze G0 identifiability; preregister at most three "
                "production-grid witnesses only after a unique claim decision"
            )
            campaign["updated_at"] = _now()
            session["completed_at"] = _now()
            _write_json_atomic(manifest_path, campaign)
        except Exception as exc:
            active = campaign.get("active_case") or "__campaign__"
            campaign["status"] = "failed"
            campaign["failures"][active] = (
                f"{type(exc).__name__}: {exc}"
            )
            campaign["updated_at"] = _now()
            session["failed_at"] = _now()
            session["failure"] = campaign["failures"][active]
            _write_json_atomic(manifest_path, campaign)
            raise

    print(f"[n1-n2-witness] COMPLETE -> {output}", flush=True)
    return 0


def _plan_payload() -> dict[str, Any]:
    cases = _case_contracts()
    source_closure, identity_gate = _campaign_inputs(cases)
    return {
        "schema": SCHEMA_VERSION,
        "gpu_initialized": False,
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
        help="validate and print the frozen contract without importing Warp",
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
