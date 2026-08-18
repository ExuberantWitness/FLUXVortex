"""Frozen raw-only Baik-W2 driver for the bounded FluxV v5h10 slice.

This module deliberately has no observation-data import and no scoring path.
It advances eight independent, source-only LDVM cells through source steps
1--6, then supplies the preregistered active steps 4--6 to the global release
row/Ptera/rVPM coupling.  The only persisted aerodynamic values are Ptera's
native raw force and moment vectors.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict
from datetime import datetime, timezone
import errno
from hashlib import sha256
from importlib import metadata as importlib_metadata
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Final, Sequence
from uuid import uuid4

import numpy as np
import pterasoftware as ps

from .baik2012 import BAIK_2012_CASES, baik_kinematics, build_baik_movement
from .fluxv_v5h10_baik_coupling import (
    ACTIVE_BIRTH_MODES,
    ACTIVE_PTERA_STEPS,
    ACTIVE_SOURCE_STEPS,
    COUPLING_INTERFACE_ID,
    FORCE_SCORING_STATUS,
    OBSERVATION_ACCESS,
    BaikW2RawStepRecord,
    GlobalRowCommitRequest,
    make_fluxv_v5h10_baik_w2_solver,
)
from .fluxv_v5h10_row_owner import (
    INTERFACE_ID as ROW_OWNER_INTERFACE_ID,
    ReleaseRowOwner,
    RowCommitResult,
    bootstrap_release_row_owner,
    commit_release_row_update,
    make_release_row,
    propose_release_row_update,
    validate_current_release_row_owner,
)
from .fluxv_v5h3_native_feedback import (
    FEEDBACK_INTERFACE_ID,
    FEEDBACK_OWNER,
    FORCE_SCORING_STATUS as FEEDBACK_SCORING_STATUS,
    SURFACE_LOAD_OWNER,
)
from .fluxv_v5h4_ptera_rvpm_transport import (
    FORCE_SCORING_STATUS as TRANSPORT_SCORING_STATUS,
    PTERA_FIELD_OWNER,
    PTERA_RVPM_TRANSPORT_INTERFACE_ID,
    RVPM_TRANSPORT_OWNER,
)
from .ldvm_uvlm_correction import LDVMSectionSettings, LESPThreshold
from .v5h_dvm_source import DVMSourceEvent, V5hDVMSource, validate_dvm_source_event


RUNNER_SCHEMA_ID: Final = "fluxv-v5h10-baik-w2-raw-run-v1"
RUN_MANIFEST_SCHEMA_ID: Final = "fluxv-v5h10-baik-w2-run-manifest-v1"
OWNER_EVENT_SCHEMA_ID: Final = "fluxv-v5h10-baik-w2-owner-event-v1"
CASE_ID: Final = "W2"
SOURCE_CELL_COUNT: Final = 8
SOURCE_STEPS: Final = (1, 2, 3, 4, 5, 6)
STEPS_PER_CYCLE: Final = 32
LESP_CRITICAL: Final = 0.11
SOURCE_NDIV: Final = 32
SOURCE_NATERM: Final = 14
SOURCE_MAX_WAKE_STEPS: Final = 64
SOURCE_CORE_RADIUS_CHORD: Final = 0.02
ROW_SMOOTHING_RADIUS_M: Final = 0.00152
ROW_TARGET_SPACING_M: Final = 0.0007152941176470589
ROW_PARTICLE_CAP: Final = 100_000
ROW_SHEET_ID: Final = "baik-w2-global-lev-row"
ROW_OWNER_ID: Final = "fluxv-v5h10-baik-w2-raw-owner"
MAX_SOURCE_KELVIN_M2_PER_S: Final = 1.0e-10
MAPPING_EPS_FACTOR: Final = 512.0
FROZEN_ACTIVE_PTERA_STEPS: Final = (3, 4, 5)
PASS_ARTIFACT_FILES: Final = (
    "raw_steps.csv",
    "source_events.csv",
    "owner_events.jsonl",
    "particle_counts.csv",
    "raw_loads.csv",
    "summary.json",
    "run_manifest.json",
    "SHA256SUMS",
    "run.log",
)
FROZEN_ACTIVE_NEW_LEV_OVER_UC: Final = (
    -0.11800430921826004,
    -0.4073225577074571,
    -0.14784149241185401,
)
FROZEN_ACTIVE_PERSISTED_LEV_OVER_UC: Final = (
    -0.11800430921826004,
    -0.5253268669257172,
    -0.6731683593375712,
)
# These two values are deliberately left unfrozen while their implementations
# are under fresh audit.  A raw PASS is impossible until an auditor accepts the
# repairs and these constants are replaced by their exact lowercase SHA-256s.
FROZEN_ROW_OWNER_SHA256: Final[str | None] = None
FROZEN_COUPLING_SHA256: Final[str | None] = None
FROZEN_SOURCE_DEPENDENCY_SHA256: Final = {
    "v5h_dvm_source.py": (
        "fe973845a7a396354c54f7636bcbf0e8d9871ae5223a413abc10d8a6e36ac3bc"
    ),
    "ldvm_fourier.py": (
        "3a2b5d4c0afa9096c4d58d34963ad25f87020aef389d44b1cf9d1e8fd59d90b9"
    ),
    "ldvm_uvlm_correction.py": (
        "d72d794242ab702bcd95ab1f5aaf6e623a87722eb1474f073124f6f60031db13"
    ),
    "baik2012.py": ("a8305a6058d02aad12b9acb376b8a38d217789bc767fcc91a15b3fa07434a86d"),
}


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: object) -> str:
    value = np.asarray(array)
    digest = sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(repr(value.shape).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _particle_state_arrays(
    state: object, *, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(state.positions)
    gamma = np.asarray(state.gamma)
    sigma = np.asarray(state.sigma)
    arrays = (positions, gamma, sigma)
    if any(
        value.dtype.kind not in "iuf" or value.dtype.kind == "b" for value in arrays
    ):
        raise RuntimeError(f"STOP: {label} particle state has a non-real dtype")
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or gamma.shape != positions.shape
        or sigma.shape != (positions.shape[0],)
        or any(not np.all(np.isfinite(value)) for value in arrays)
        or np.any(sigma <= 0.0)
    ):
        raise RuntimeError(
            f"STOP: {label} particle state has invalid shape/non-finite/non-positive sigma"
        )
    return positions, gamma, sigma


def _particle_state_sha256(state: object, *, label: str) -> str:
    positions, gamma, sigma = _particle_state_arrays(state, label=label)
    digest = sha256()
    for name, array in (
        ("positions", positions),
        ("gamma", gamma),
        ("sigma", sigma),
    ):
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_bytes_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish ``staging`` without ever replacing ``destination``.

    Linux ``renameat2(RENAME_NOREPLACE)`` closes the check/use race inherent in
    ``exists(); os.rename(...)``.  There is deliberately no overwrite-capable
    fallback: an unavailable kernel/libc primitive is a fail-closed STOP.
    """

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("STOP: atomic no-replace directory publish unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
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
        raise RuntimeError(
            "STOP: atomic no-replace directory publish unavailable"
        ) from OSError(error_number, os.strerror(error_number))
    raise OSError(error_number, os.strerror(error_number), destination)


def _run_provenance(
    output_dir: Path, invocation_argv: Sequence[str] | None
) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
        revision = (
            subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                capture_output=True,
            )
            .stdout.decode("ascii")
            .strip()
        )
        porcelain = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        git_error = None
    except (OSError, subprocess.CalledProcessError) as error:
        revision = None
        porcelain = b""
        git_error = f"{type(error).__name__}: {error}"

    def package_version(name: str) -> str | None:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return None

    return {
        "run_uuid": str(uuid4()),
        "start_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_dir),
        "argv": list(sys.argv if invocation_argv is None else invocation_argv),
        "cwd": os.getcwd(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "package_versions": {
            "numpy": package_version("numpy"),
            "pterasoftware": package_version("pterasoftware"),
            "numba": package_version("numba"),
        },
        "git": {
            "revision": revision,
            "dirty": bool(porcelain),
            "porcelain_sha256": sha256(porcelain).hexdigest(),
            "porcelain_line_count": len(porcelain.splitlines()),
            "error": git_error,
        },
    }


def _runtime_dependency_sha256() -> dict[str, str]:
    module_dir = Path(__file__).resolve().parent
    return {
        "row_owner": _sha256_file(module_dir / "fluxv_v5h10_row_owner.py"),
        "coupling": _sha256_file(module_dir / "fluxv_v5h10_baik_coupling.py"),
        "v5h_dvm_source.py": _sha256_file(module_dir / "v5h_dvm_source.py"),
        "ldvm_fourier.py": _sha256_file(module_dir.parent / "ldvm_fourier.py"),
        "ldvm_uvlm_correction.py": _sha256_file(module_dir / "ldvm_uvlm_correction.py"),
        "baik2012.py": _sha256_file(module_dir / "baik2012.py"),
    }


def _assert_runtime_dependency_manifest() -> dict[str, str]:
    actual = _runtime_dependency_sha256()
    if FROZEN_ROW_OWNER_SHA256 is None or FROZEN_COUPLING_SHA256 is None:
        raise RuntimeError(
            "STOP: repaired row-owner/coupling hashes are not audit-frozen"
        )
    if actual["row_owner"] != FROZEN_ROW_OWNER_SHA256:
        raise RuntimeError("STOP: row-owner source differs from the audited manifest")
    if actual["coupling"] != FROZEN_COUPLING_SHA256:
        raise RuntimeError("STOP: coupling source differs from the audited manifest")
    if (
        ACTIVE_PTERA_STEPS != FROZEN_ACTIVE_PTERA_STEPS
        or ACTIVE_SOURCE_STEPS != (4, 5, 6)
        or ACTIVE_BIRTH_MODES != ("first", "continuous", "continuous")
    ):
        raise RuntimeError("STOP: source/Ptera clock differs from prereg")
    for name, expected in FROZEN_SOURCE_DEPENDENCY_SHA256.items():
        if actual[name] != expected:
            raise RuntimeError(f"STOP: source dependency differs from prereg: {name}")
    return actual


def _source_threshold() -> LESPThreshold:
    return LESPThreshold(
        value=LESP_CRITICAL,
        section_family="rounded flat plate",
        reynolds=BAIK_2012_CASES[CASE_ID].reynolds,
        source=(
            "Ramesh 2013 thesis flat-plate Re=1000 sections 4.3.5 and "
            "Figures 4.19/4.21 use Lcrit=0.11; frozen cross-Re/thickness "
            "transfer hypothesis with no Baik force fit"
        ),
        source_role="published_source_input",
    )


def _source_settings() -> LDVMSectionSettings:
    return LDVMSectionSettings(
        ndiv=SOURCE_NDIV,
        naterm=SOURCE_NATERM,
        max_wake_steps=SOURCE_MAX_WAKE_STEPS,
        core_radius_chord=SOURCE_CORE_RADIUS_CHORD,
    )


def _make_source(cell_index: int) -> V5hDVMSource:
    if type(cell_index) is not int or not 0 <= cell_index < SOURCE_CELL_COUNT:
        raise ValueError("cell_index is outside the frozen eight-cell row")
    case = BAIK_2012_CASES[CASE_ID]
    convective_dt = case.freestream_m_s * case.period_s / case.chord_m / STEPS_PER_CYCLE
    return V5hDVMSource(
        physical_section_id=f"baik-w2:cell:{cell_index}:section",
        physical_strip_id=f"baik-w2:cell:{cell_index}:strip",
        geometry_identity="explicit zero-camber rounded-flat-plate mean-line surrogate",
        reference_speed_m_per_s=case.freestream_m_s,
        reference_chord_m=case.chord_m,
        zero_camber_surrogate=True,
        delta_time_convective=convective_dt,
        pivot_fraction_chord=case.pivot_fraction_chord,
        threshold=_source_threshold(),
        settings=_source_settings(),
    )


def build_w2_source_events() -> tuple[tuple[DVMSourceEvent, ...], ...]:
    """Return step-major, exact-attested source events for 6 steps x 8 cells."""

    case = BAIK_2012_CASES[CASE_ID]
    sources = tuple(_make_source(cell) for cell in range(SOURCE_CELL_COUNT))
    pitch_amplitude_rad = np.deg2rad(case.implemented_pitch_amplitude_deg)
    rows: list[tuple[DVMSourceEvent, ...]] = []
    for source_step in SOURCE_STEPS:
        phase = (source_step - 1) / STEPS_PER_CYCLE
        kinematics = baik_kinematics(phase, case)
        alpha_rad = float(np.deg2rad(kinematics["geometric_alpha_deg"]))
        alpha_rate = float(
            -2.0
            * case.reduced_frequency
            * pitch_amplitude_rad
            * np.cos(2.0 * np.pi * phase)
        )
        heave_rate = float(kinematics["heave_rate_over_u"])
        events = tuple(
            source.step(alpha_rad, alpha_rate, heave_rate) for source in sources
        )
        for event in events:
            if validate_dvm_source_event(event) is not event:
                raise RuntimeError("DVM event validator changed live identity")
            if event.lineage.source_step_index != source_step:
                raise RuntimeError("DVM source clock is not the frozen 1--6 schedule")
            dimensional_kelvin = abs(event.kelvin_residual_over_u_c) * (
                event.provenance.circulation_scale_u_times_c_m2_per_s
            )
            if dimensional_kelvin > MAX_SOURCE_KELVIN_M2_PER_S:
                raise RuntimeError("DVM source Kelvin residual exceeded the hard gate")
        rows.append(events)
    result = tuple(rows)
    modes = tuple(result[step - 1][0].lev_birth_mode for step in SOURCE_STEPS)
    if modes != ("none", "none", "none", *ACTIVE_BIRTH_MODES):
        raise RuntimeError(f"unexpected frozen W2 LEV schedule: {modes!r}")
    for active_index, source_step in enumerate(ACTIVE_SOURCE_STEPS):
        active_events = result[source_step - 1]
        expected_new = FROZEN_ACTIVE_NEW_LEV_OVER_UC[active_index]
        expected_persisted = FROZEN_ACTIVE_PERSISTED_LEV_OVER_UC[active_index]
        if any(event.gamma_lev_new_over_u_c != expected_new for event in active_events):
            raise RuntimeError("DVM newborn LEV circulation differs from prereg")
        if any(
            event.kelvin_ledger is None
            or event.kelvin_ledger.gamma_lev_persisted_after != expected_persisted
            for event in active_events
        ):
            raise RuntimeError("DVM cumulative LEV circulation differs from prereg")
    return result


def _ptera_edge_nodes_from_solver(solver: object) -> tuple[np.ndarray, np.ndarray]:
    airplanes = tuple(getattr(solver, "current_airplanes", ()))
    if len(airplanes) != 1 or len(airplanes[0].wings) != 1:
        raise RuntimeError("STOP: W2 mapping requires exactly one current Ptera wing")
    panels = np.asarray(airplanes[0].wings[0].panels, dtype=object)
    if panels.shape != (2, SOURCE_CELL_COUNT):
        raise RuntimeError("STOP: current Ptera lattice is not the frozen 2x8 grid")
    edge_rows: list[np.ndarray] = []
    for chordwise, edge_name in ((0, "leading"), (-1, "trailing")):
        nodes: list[np.ndarray] = []
        for cell, panel in enumerate(panels[chordwise]):
            expected_flag = (
                bool(panel.is_leading_edge)
                if edge_name == "leading"
                else bool(panel.is_trailing_edge)
            )
            if not expected_flag:
                raise RuntimeError(
                    f"STOP: Ptera {edge_name}-edge topology is inconsistent"
                )
            left = np.asarray(
                panel.Flpp_GP1_CgP1 if edge_name == "leading" else panel.Blpp_GP1_CgP1,
                dtype=np.float64,
            )
            right = np.asarray(
                panel.Frpp_GP1_CgP1 if edge_name == "leading" else panel.Brpp_GP1_CgP1,
                dtype=np.float64,
            )
            if left.shape != (3,) or right.shape != (3,):
                raise RuntimeError(
                    f"STOP: Ptera {edge_name}-edge vertex shape is invalid"
                )
            if cell == 0:
                nodes.append(left.copy())
            elif not np.array_equal(nodes[-1], left):
                raise RuntimeError(
                    f"STOP: adjacent Ptera {edge_name}-edge anchors do not close"
                )
            nodes.append(right.copy())
        edge = np.ascontiguousarray(nodes, dtype=np.float64)
        if edge.shape != (SOURCE_CELL_COUNT + 1, 3) or not np.all(np.isfinite(edge)):
            raise RuntimeError(f"STOP: mapped Ptera {edge_name}-edge nodes are invalid")
        if np.any(np.linalg.norm(np.diff(edge, axis=0), axis=1) <= 0.0):
            raise RuntimeError(f"STOP: Ptera {edge_name}-edge has a zero-length cell")
        edge_rows.append(edge)
    leading_edge, trailing_edge = edge_rows
    chord_lengths = np.linalg.norm(trailing_edge - leading_edge, axis=1)
    chord = BAIK_2012_CASES[CASE_ID].chord_m
    tolerance = MAPPING_EPS_FACTOR * np.finfo(np.float64).eps * chord
    if float(np.max(np.abs(chord_lengths - chord), initial=0.0)) > tolerance:
        raise RuntimeError("STOP: live Ptera LE/TE anchors do not preserve chord")
    return leading_edge, trailing_edge


def _leading_edge_nodes_from_solver(solver: object) -> np.ndarray:
    leading_edge, _ = _ptera_edge_nodes_from_solver(solver)
    return leading_edge


def _validate_fixed_inertial_dvm_axes(
    leading_edge_nodes: np.ndarray,
    trailing_edge_nodes: np.ndarray,
    events: Sequence[DVMSourceEvent],
) -> None:
    """Attest the direct DVM-world (x,z) map against every live Ptera chord."""

    if len(events) != SOURCE_CELL_COUNT:
        raise RuntimeError("STOP: inertial-axis gate requires eight source events")
    chord = BAIK_2012_CASES[CASE_ID].chord_m
    tolerance = MAPPING_EPS_FACTOR * np.finfo(np.float64).eps * chord
    for cell, event in enumerate(events):
        lev_anchor = event.lev_placement.edge_anchor_position_over_chord_backend_world
        tev_anchor = event.tev_placement.edge_anchor_position_over_chord_backend_world
        if lev_anchor is None or tev_anchor is None:
            raise RuntimeError("STOP: DVM event lacks live LE/TE anchor evidence")
        dvm_chord = np.asarray(
            (
                (tev_anchor[0] - lev_anchor[0]) * chord,
                0.0,
                (tev_anchor[1] - lev_anchor[1]) * chord,
            ),
            dtype=np.float64,
        )
        for node in (cell, cell + 1):
            ptera_chord = trailing_edge_nodes[node] - leading_edge_nodes[node]
            if float(np.max(np.abs(ptera_chord - dvm_chord), initial=0.0)) > tolerance:
                raise RuntimeError(
                    "STOP: fixed inertial DVM axes disagree with live Ptera LE/TE anchors"
                )


def _first_birth_nodes(
    leading_edge_nodes: np.ndarray,
    events: Sequence[DVMSourceEvent],
) -> np.ndarray:
    if len(events) != SOURCE_CELL_COUNT:
        raise RuntimeError("STOP: first row does not have eight DVM cell events")
    case = BAIK_2012_CASES[CASE_ID]
    displacements: list[tuple[float, float]] = []
    for event in events:
        if event.lev_birth_mode != "first":
            raise RuntimeError("STOP: first row was not sourced by first LEV births")
        displacement = (
            event.lev_placement.birth_displacement_from_edge_over_chord_backend_world
        )
        if displacement is None:
            raise RuntimeError("STOP: first LEV event lacks a birth displacement")
        displacements.append(displacement)
    first = displacements[0]
    if any(value != first for value in displacements[1:]):
        raise RuntimeError("STOP: cell-local first-birth mappings disagree")
    offset = np.asarray(
        (first[0] * case.chord_m, 0.0, first[1] * case.chord_m),
        dtype=np.float64,
    )
    downstream = np.ascontiguousarray(leading_edge_nodes + offset, dtype=np.float64)
    recovered = downstream - leading_edge_nodes
    scale = max(1.0, float(np.max(np.abs(downstream), initial=0.0)))
    tolerance = MAPPING_EPS_FACTOR * np.finfo(np.float64).eps * scale
    if float(np.max(np.abs(recovered - offset), initial=0.0)) > tolerance:
        raise RuntimeError("STOP: inertial DVM birth mapping does not close")
    return downstream


def _dimensional_cumulative_lev(events: Sequence[DVMSourceEvent]) -> np.ndarray:
    values: list[float] = []
    for event in events:
        ledger = event.kelvin_ledger
        if ledger is None or not event.lesp_active:
            raise RuntimeError("STOP: active row lacks a persisted LEV ledger")
        values.append(
            float(ledger.gamma_lev_persisted_after)
            * event.provenance.circulation_scale_u_times_c_m2_per_s
        )
    result = np.ascontiguousarray(values, dtype=np.float64)
    if result.shape != (SOURCE_CELL_COUNT,) or not np.all(np.isfinite(result)):
        raise RuntimeError("STOP: cumulative LEV row circulation is invalid")
    if np.any(result == 0.0):
        raise RuntimeError("STOP: active cumulative LEV circulation is zero")
    return result


class BaikW2RawRowCommitter:
    """Immediate source-event to global-row transaction boundary."""

    def __init__(
        self,
        events_by_step: tuple[tuple[DVMSourceEvent, ...], ...],
        *,
        owner_id: str = ROW_OWNER_ID,
    ) -> None:
        if len(events_by_step) != len(SOURCE_STEPS) or any(
            len(row) != SOURCE_CELL_COUNT for row in events_by_step
        ):
            raise ValueError("events_by_step must be the frozen 6x8 source ledger")
        self._events_by_step = events_by_step
        self._owner_id = owner_id
        self._solver: object | None = None
        self.requests: list[GlobalRowCommitRequest] = []
        self.commits: list[RowCommitResult] = []

    def bind_solver(self, solver: object) -> None:
        if self._solver is not None:
            raise RuntimeError("row committer solver binding is single-assignment")
        self._solver = solver

    def __call__(self, request: GlobalRowCommitRequest) -> RowCommitResult:
        if type(request) is not GlobalRowCommitRequest:
            raise TypeError("row request must use the exact frozen coupling schema")
        if self._solver is None:
            raise RuntimeError("row committer has no live Ptera solver binding")
        if request.source_step_index not in ACTIVE_SOURCE_STEPS:
            raise RuntimeError("STOP: coupling requested an unregistered source step")
        active_index = ACTIVE_SOURCE_STEPS.index(request.source_step_index)
        events = self._events_by_step[request.source_step_index - 1]
        if any(event.lev_birth_mode != request.expected_birth_mode for event in events):
            raise RuntimeError("STOP: DVM birth mode disagrees with coupling schedule")
        leading_edge, trailing_edge = _ptera_edge_nodes_from_solver(self._solver)
        circulation = _dimensional_cumulative_lev(events)
        _validate_fixed_inertial_dvm_axes(leading_edge, trailing_edge, events)

        if request.previous_owner is None:
            if active_index != 0 or request.transported_parent is not None:
                raise RuntimeError("STOP: first row unexpectedly has a parent")
            upstream = leading_edge
            downstream = _first_birth_nodes(leading_edge, events)
        else:
            if active_index == 0 or request.transported_parent is None:
                raise RuntimeError("STOP: continuous row lacks transported ancestry")
            upstream = np.ascontiguousarray(
                request.previous_owner.state.live_boundary_nodes, dtype=np.float64
            )
            downstream = np.ascontiguousarray(
                leading_edge + (upstream - leading_edge) / 3.0,
                dtype=np.float64,
            )
            reconstructed = leading_edge + (upstream - leading_edge) / 3.0
            if not np.array_equal(downstream, reconstructed):
                raise RuntimeError("STOP: continuous one-third geometry is not exact")

        row = make_release_row(
            upstream,
            downstream,
            circulation,
            release_index=active_index + 1,
            source_time_s=request.source_time_s,
            sheet_id=ROW_SHEET_ID,
        )
        if request.previous_owner is None:
            owner = bootstrap_release_row_owner(
                row,
                smoothing_radius_m=ROW_SMOOTHING_RADIUS_M,
                target_spacing_m=ROW_TARGET_SPACING_M,
                release_dt_s=BAIK_2012_CASES[CASE_ID].period_s / STEPS_PER_CYCLE,
                particle_cap=ROW_PARTICLE_CAP,
                owner_id=self._owner_id,
            )
            result = RowCommitResult(
                committed=True,
                status="compatible",
                owner=owner,
                state=owner.state,
                event=None,
                first_mismatch=None,
            )
        else:
            proposal = propose_release_row_update(
                request.previous_owner,
                row,
                proposal_id=f"baik-w2-source-step-{request.source_step_index}",
            )
            if proposal.status != "compatible":
                raise RuntimeError(
                    f"STOP: row remesh required: {proposal.first_mismatch}"
                )
            result = commit_release_row_update(request.previous_owner, proposal)
            if not result.committed or result.status != "compatible":
                raise RuntimeError("STOP: compatible row proposal did not commit")
        self.requests.append(request)
        self.commits.append(result)
        return result


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_sha256(value):
        raise RuntimeError(f"STOP: invalid SHA-256: {name}")
    return value


def _assert_no_penetration(feedback: object) -> None:
    residual = np.asarray(feedback.no_penetration_residual)
    maximum = feedback.no_penetration_max_abs
    if (
        residual.shape != (16,)
        or not np.all(np.isfinite(residual))
        or not np.isfinite(float(maximum))
        or maximum != float(np.max(np.abs(residual), initial=0.0))
        or maximum > 1.0e-12
    ):
        raise RuntimeError("STOP: no-penetration evidence is invalid")


def _format_csv_value(value: object) -> object:
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("CSV payload contains a non-finite float")
        return format(value, ".17g")
    return value


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _format_csv_value(row[name]) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _source_event_rows(
    events_by_step: tuple[tuple[DVMSourceEvent, ...], ...]
) -> list[dict[str, object]]:
    case = BAIK_2012_CASES[CASE_ID]
    rows: list[dict[str, object]] = []
    for source_step, events in zip(SOURCE_STEPS, events_by_step, strict=True):
        phase = (source_step - 1) / STEPS_PER_CYCLE
        kinematics = baik_kinematics(phase, case)
        alpha_rad = float(np.deg2rad(kinematics["geometric_alpha_deg"]))
        alpha_rate = float(
            -2.0
            * case.reduced_frequency
            * np.deg2rad(case.implemented_pitch_amplitude_deg)
            * np.cos(2.0 * np.pi * phase)
        )
        for cell_index, event in enumerate(events):
            ledger = event.kelvin_ledger
            if ledger is None:
                raise RuntimeError("STOP: enabled source event lacks Kelvin ledger")
            scale = event.provenance.circulation_scale_u_times_c_m2_per_s
            rows.append(
                {
                    "source_step_index": source_step,
                    "ptera_step_index": source_step - 1,
                    "source_input_phase": phase,
                    "source_input_time_s": phase * case.period_s,
                    "source_internal_end_convective_time": (
                        source_step * event.provenance.delta_time_convective_nominal
                    ),
                    "cell_index": cell_index,
                    "geometric_alpha_rad": alpha_rad,
                    "alpha_rate_per_convective_time": alpha_rate,
                    "heave_rate_over_u": float(kinematics["heave_rate_over_u"]),
                    "a0_pre": float(event.a0_pre),
                    "a0_post": float(event.a0_post),
                    "lesp_active": event.lesp_active,
                    "birth_mode": event.lev_birth_mode,
                    "gamma_lev_new_over_u_c": event.gamma_lev_new_over_u_c,
                    "gamma_lev_new_m2_s": event.gamma_lev_new_over_u_c * scale,
                    "gamma_lev_persisted_after_over_u_c": (
                        ledger.gamma_lev_persisted_after
                    ),
                    "gamma_lev_persisted_after_m2_s": (
                        ledger.gamma_lev_persisted_after * scale
                    ),
                    "gamma_tev_new_solved_over_u_c": (
                        event.gamma_tev_new_solved_over_u_c
                    ),
                    "gamma_tev_new_persisted_over_u_c": (
                        event.gamma_tev_new_persisted_over_u_c
                    ),
                    "gamma_tev_persisted_after_over_u_c": (
                        ledger.gamma_tev_persisted_after
                    ),
                    "kelvin_residual_over_u_c": event.kelvin_residual_over_u_c,
                    "kelvin_residual_m2_s": event.kelvin_residual_over_u_c * scale,
                    "event_manifest_sha256": event.producer_manifest_sha256,
                    "parent_event_manifest_sha256": (
                        event.parent_event_manifest_sha256
                    ),
                }
            )
    return rows


def _owner_event_rows(final_owner: ReleaseRowOwner) -> list[dict[str, object]]:
    state = final_owner.state
    rows: list[dict[str, object]] = []
    for release_index in (1, 2, 3):
        commit_event = (
            None if release_index == 1 else final_owner.events[release_index - 2]
        )
        transport_event = final_owner.transport_events[release_index - 1]
        changed_indices = () if commit_event is None else commit_event.changed_indices
        changed_ids = [state.particle_ids[index] for index in changed_indices]
        appended_ids = [
            particle_id
            for particle_id, lineage in zip(
                state.particle_ids, state.lineage, strict=True
            )
            if lineage.birth_release_index == release_index
        ]
        rows.append(
            {
                "schema_id": OWNER_EVENT_SCHEMA_ID,
                "case_id": CASE_ID,
                "release_index": release_index,
                "row_sha256": state.rows[release_index - 1].row_sha256,
                "changed_particle_ids": changed_ids,
                "changed_particle_ids_sha256": sha256(
                    _json_bytes(changed_ids)
                ).hexdigest(),
                "appended_particle_ids": appended_ids,
                "appended_particle_ids_sha256": sha256(
                    _json_bytes(appended_ids)
                ).hexdigest(),
                "commit_event": None if commit_event is None else asdict(commit_event),
                "transport_event": asdict(transport_event),
            }
        )
    return rows


def _panel_load_arrays(record: BaikW2RawStepRecord) -> tuple[np.ndarray, np.ndarray]:
    forces = np.asarray(record.ptera_panel_forces_w, dtype=np.float64)
    moments = np.asarray(record.ptera_panel_moments_w_cgp1, dtype=np.float64)
    total_force = np.asarray(record.ptera_forces_w, dtype=np.float64)
    total_moment = np.asarray(record.ptera_moments_w_cgp1, dtype=np.float64)
    if forces.shape != (16, 3) or moments.shape != (16, 3):
        raise RuntimeError("STOP: coupling raw panel-load shape is not (16,3)")
    if total_force.shape != (3,) or total_moment.shape != (3,):
        raise RuntimeError("STOP: coupling raw total-load shape is not (3,)")
    if any(
        not np.all(np.isfinite(value))
        for value in (forces, moments, total_force, total_moment)
    ):
        raise RuntimeError("STOP: coupling raw panel loads are non-finite")
    expected_ids = tuple(
        f"airplane:0/wing:0/chord:{chord}/span:{span}"
        for chord in range(2)
        for span in range(8)
    )
    if record.ptera_panel_ids != expected_ids:
        raise RuntimeError("STOP: coupling raw panel topology/order changed")
    if (
        _array_sha256(forces) != record.ptera_panel_forces_w_sha256
        or _array_sha256(moments) != record.ptera_panel_moments_w_cgp1_sha256
    ):
        raise RuntimeError("STOP: coupling raw panel-load hash changed")
    force_sum = np.sum(forces, axis=0)
    moment_sum = np.sum(moments, axis=0)
    force_residual = float(np.max(np.abs(force_sum - total_force)))
    moment_residual = float(np.max(np.abs(moment_sum - total_moment)))
    force_atol = float(
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.sum(np.abs(forces))))
    )
    moment_atol = float(
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.sum(np.abs(moments))))
    )
    if (
        not np.array_equal(force_sum, record.ptera_panel_force_sum_w)
        or not np.array_equal(moment_sum, record.ptera_panel_moment_sum_w_cgp1)
        or not np.isfinite(record.ptera_panel_force_sum_max_abs_residual)
        or not np.isfinite(record.ptera_panel_moment_sum_max_abs_residual)
        or not np.isfinite(record.ptera_panel_force_sum_atol)
        or not np.isfinite(record.ptera_panel_moment_sum_atol)
        or record.ptera_panel_force_sum_max_abs_residual != force_residual
        or record.ptera_panel_moment_sum_max_abs_residual != moment_residual
        or record.ptera_panel_force_sum_atol != force_atol
        or record.ptera_panel_moment_sum_atol != moment_atol
        or force_residual > force_atol
        or moment_residual > moment_atol
    ):
        raise RuntimeError("STOP: coupling raw panel-sum ledger failed")
    return forces, moments


def _transport_stage_evidence(
    record: BaikW2RawStepRecord,
) -> tuple[dict[str, object], ...]:
    transport = record.transport_result
    if len(transport.stages) != 3:
        raise RuntimeError("STOP: transport does not expose exactly three stages")
    evidence: list[dict[str, object]] = []
    previous_post_sha256: str | None = None
    for stage_index, stage in enumerate(transport.stages, start=1):
        pre_positions, _, _ = _particle_state_arrays(
            stage.pre, label=f"stage {stage_index} pre"
        )
        post_positions, _, _ = _particle_state_arrays(
            stage.post, label=f"stage {stage_index} post"
        )
        pre_sha256 = _particle_state_sha256(stage.pre, label=f"stage {stage_index} pre")
        post_sha256 = _particle_state_sha256(
            stage.post, label=f"stage {stage_index} post"
        )
        field = stage.ptera_field
        targets = np.asarray(field.target_positions_gp1_m)
        velocity = np.asarray(field.velocity_gp1_m_per_s)
        jacobian = np.asarray(field.jacobian_per_s)
        if (
            stage.stage != stage_index
            or pre_positions.shape != (record.particle_count, 3)
            or post_positions.shape != (record.particle_count, 3)
            or targets.shape != (record.particle_count, 3)
            or velocity.shape != (record.particle_count, 3)
            or jacobian.shape != (record.particle_count, 3, 3)
            or not np.array_equal(targets, pre_positions)
            or any(
                not np.all(np.isfinite(value))
                for value in (targets, velocity, jacobian)
            )
            or field.center_call_count != 1
            or field.finite_difference_call_count != 6
            or not _is_sha256(field.target_sha256)
            or not _is_sha256(field.velocity_sha256)
            or not _is_sha256(field.jacobian_sha256)
            or _array_sha256(targets) != field.target_sha256
            or _array_sha256(velocity) != field.velocity_sha256
            or _array_sha256(jacobian) != field.jacobian_sha256
            or (previous_post_sha256 is not None and pre_sha256 != previous_post_sha256)
        ):
            raise RuntimeError("STOP: nested finite-difference stage is invalid")
        evidence.append(
            {
                "stage": stage_index,
                "pre_particle_state_sha256": pre_sha256,
                "post_particle_state_sha256": post_sha256,
                "ptera_target_sha256": field.target_sha256,
                "ptera_velocity_sha256": field.velocity_sha256,
                "ptera_jacobian_sha256": field.jacobian_sha256,
                "ptera_center_call_count": field.center_call_count,
                "ptera_finite_difference_call_count": (
                    field.finite_difference_call_count
                ),
            }
        )
        previous_post_sha256 = post_sha256
    final_sha256 = _particle_state_sha256(
        transport.final_state, label="transport final"
    )
    if final_sha256 != previous_post_sha256:
        raise RuntimeError("STOP: final transport state is not stage-three post state")
    return tuple(evidence)


def _assert_evaluation(value: object, *, channel: str, ptera_step: int) -> None:
    if value.channel != channel or value.ptera_step_index != ptera_step:
        raise RuntimeError("STOP: nested Ptera evaluation identity changed")
    targets = np.asarray(value.target_points_gp1_m)
    velocity = np.asarray(value.induced_velocity_gp1_m_per_s)
    if (
        targets.ndim != 2
        or targets.shape[1:] != (3,)
        or velocity.shape != targets.shape
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(velocity))
        or not _is_sha256(value.target_sha256)
        or not _is_sha256(value.velocity_sha256)
        or _array_sha256(targets) != value.target_sha256
        or _array_sha256(velocity) != value.velocity_sha256
    ):
        raise RuntimeError("STOP: nested Ptera evaluation payload is invalid")


def _assert_raw_records(
    records: Sequence[object],
    committer: BaikW2RawRowCommitter,
    final_owner: ReleaseRowOwner,
) -> None:
    if ACTIVE_PTERA_STEPS != FROZEN_ACTIVE_PTERA_STEPS:
        raise RuntimeError("STOP: coupling Ptera clock is not frozen at 3/4/5")
    if tuple(record.ptera_step_index for record in records) != ACTIVE_PTERA_STEPS:
        raise RuntimeError("STOP: raw Ptera step sequence differs from prereg")
    if tuple(record.source_step_index for record in records) != ACTIVE_SOURCE_STEPS:
        raise RuntimeError("STOP: raw source step sequence differs from prereg")
    if tuple(record.birth_mode for record in records) != ACTIVE_BIRTH_MODES:
        raise RuntimeError("STOP: raw birth-mode sequence differs from prereg")
    if any(type(record) is not BaikW2RawStepRecord for record in records):
        raise TypeError("STOP: raw record uses a foreign schema")
    if len(committer.requests) != 3 or len(committer.commits) != 3:
        raise RuntimeError("STOP: row commit evidence is incomplete")
    if validate_current_release_row_owner(final_owner) is not final_owner:
        raise RuntimeError("STOP: final owner is not the exact live owner")
    if (
        final_owner.state.interface_id != ROW_OWNER_INTERFACE_ID
        or final_owner.epoch != 3
        or len(final_owner.state.rows) != 3
        or len(final_owner.events) != 2
        or len(final_owner.transport_events) != 3
        or final_owner.state.clone_count != 0
        or final_owner.state.counter_particle_count != 0
        or final_owner.state.fresh_upstream_particle_count != 0
    ):
        raise RuntimeError("STOP: final owner lineage/schema is inconsistent")

    digest_fields = (
        "row_commit_sha256",
        "row_state_sha256",
        "transported_state_sha256",
        "transported_arrays_sha256",
        "transport_parent_digest",
        "common_transport_sha256",
        "common_transport_attestation_sha256",
        "transported_material_tracers_sha256",
        "advanced_owner_sha256",
        "ptera_parent_sha256_before_transport",
        "ptera_parent_sha256_after_transport",
        "ptera_parent_sha256_after_raw_record",
        "ptera_forces_w_sha256",
        "ptera_force_coefficients_w_sha256",
        "ptera_moments_w_cgp1_sha256",
        "ptera_moment_coefficients_w_sha256",
        "ptera_panel_forces_w_sha256",
        "ptera_panel_moments_w_cgp1_sha256",
    )
    previous: BaikW2RawStepRecord | None = None
    previous_count = 0
    for index, candidate in enumerate(records):
        record = candidate
        request = committer.requests[index]
        commit = committer.commits[index]
        ptera_step = ACTIVE_PTERA_STEPS[index]
        source_step = ACTIVE_SOURCE_STEPS[index]
        source_time = ptera_step * BAIK_2012_CASES[CASE_ID].period_s / STEPS_PER_CYCLE
        if (
            record.interface_id != COUPLING_INTERFACE_ID
            or record.case_id != CASE_ID
            or record.ptera_step_index != ptera_step
            or record.source_step_index != source_step
            or record.source_time_s != source_time
            or record.birth_mode != ACTIVE_BIRTH_MODES[index]
            or request.ptera_step_index != ptera_step
            or request.source_step_index != source_step
            or request.source_time_s != source_time
            or request.expected_birth_mode != ACTIVE_BIRTH_MODES[index]
        ):
            raise RuntimeError("STOP: raw record/request identity or time changed")
        if (
            record.row_commit_sha256 != commit.owner.owner_sha256
            or record.row_state_sha256 != commit.state.state_sha256
            or commit.state.rows[-1].source_time_s != source_time
            or commit.state.release_index != index + 1
            or commit.state.phase != "post_commit_pre_transport"
        ):
            raise RuntimeError("STOP: raw record is not bound to its row commit")
        transport_event = final_owner.transport_events[index]
        if (
            record.transported_state_sha256 != transport_event.transported_state_sha256
            or record.transport_parent_digest != transport_event.parent_transport_digest
            or record.common_transport_attestation_sha256
            != transport_event.common_transport_attestation_sha256
            or transport_event.source_step_index != source_step
            or record.advanced_owner_sha256
            != (
                committer.requests[index + 1].previous_owner.owner_sha256
                if index < 2
                else final_owner.owner_sha256
            )
        ):
            raise RuntimeError("STOP: raw owner/transport ancestry is broken")
        if previous is None:
            if (
                record.transported_parent_sha256 is not None
                or request.previous_owner is not None
                or request.transported_parent is not None
            ):
                raise RuntimeError("STOP: first raw row has foreign ancestry")
        elif (
            record.transported_parent_sha256 != previous.transported_arrays_sha256
            or request.transported_parent_sha256 != previous.transported_arrays_sha256
            or request.previous_owner is None
            or request.previous_owner.owner_sha256 != previous.advanced_owner_sha256
        ):
            raise RuntimeError("STOP: continuous raw row ancestry is broken")

        for name in digest_fields:
            _require_sha256(name, getattr(record, name))
        if record.transported_parent_sha256 is not None and not _is_sha256(
            record.transported_parent_sha256
        ):
            raise RuntimeError("STOP: invalid transported-parent digest")

        final = record.transport_result.final_state
        positions, gamma, sigma = _particle_state_arrays(
            final, label="raw transport final"
        )
        final_particle_state_sha256 = _particle_state_sha256(
            final, label="raw transport final"
        )
        arrays = (
            record.ptera_forces_w,
            record.ptera_force_coefficients_w,
            record.ptera_moments_w_cgp1,
            record.ptera_moment_coefficients_w,
            record.transported_live_boundary_nodes,
            positions,
            gamma,
            sigma,
        )
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise RuntimeError("STOP: raw Ptera/transport arrays are non-finite")
        if (
            np.asarray(record.ptera_forces_w).shape != (3,)
            or np.asarray(record.ptera_force_coefficients_w).shape != (3,)
            or np.asarray(record.ptera_moments_w_cgp1).shape != (3,)
            or np.asarray(record.ptera_moment_coefficients_w).shape != (3,)
            or np.asarray(record.transported_live_boundary_nodes).shape != (9, 3)
            or positions.ndim != 2
            or positions.shape[1:] != (3,)
            or gamma.shape != positions.shape
            or sigma.shape != (positions.shape[0],)
            or np.any(sigma <= 0.0)
        ):
            raise RuntimeError("STOP: raw Ptera/transport shape is invalid")
        if record.transported_arrays_sha256 != final_particle_state_sha256:
            raise RuntimeError(
                "STOP: coupling transported particle-state hash changed"
            )
        if (
            _array_sha256(record.ptera_forces_w) != record.ptera_forces_w_sha256
            or _array_sha256(record.ptera_force_coefficients_w)
            != record.ptera_force_coefficients_w_sha256
            or _array_sha256(record.ptera_moments_w_cgp1)
            != record.ptera_moments_w_cgp1_sha256
            or _array_sha256(record.ptera_moment_coefficients_w)
            != record.ptera_moment_coefficients_w_sha256
        ):
            raise RuntimeError("STOP: raw total Ptera load hash changed")
        _panel_load_arrays(record)

        expected_support_tracers = sum(
            len(cell) for cell in commit.state.live_boundary_indices_by_cell
        )
        expected_tracers = expected_support_tracers + 9
        if (
            type(record.particle_count) is not int
            or not np.isfinite(float(record.particle_count))
            or record.particle_count != positions.shape[0]
            or not 0 < record.particle_count <= ROW_PARTICLE_CAP
            or record.particle_count <= previous_count
            or type(record.material_tracer_count) is not int
            or not np.isfinite(float(record.material_tracer_count))
            or record.material_tracer_count != expected_tracers
            or record.material_support_tracer_count != expected_support_tracers
            or record.frontier_node_tracer_count != 9
        ):
            raise RuntimeError("STOP: particle/material-tracer count gate failed")

        feedback = record.feedback_report
        if (
            feedback.ptera_step_index != ptera_step
            or feedback.dvm_for_source_step_index != source_step
            or feedback.source_time_s != source_time
            or feedback.feedback_report_sha256 != record.row_commit_sha256
            or feedback.particle_count != record.particle_count
            or feedback.collocation_evaluation_count != 1
            or feedback.load_leg_evaluation_count != 4
            or feedback.parent_load_call_count != 1
            or feedback.extension_force_write_count != 0
            or feedback.extension_moment_write_count != 0
            or feedback.extension_load_processor_call_count != 0
            or not feedback.prescribed_wake
            or len(feedback.load_evaluations) != 4
            or not _is_sha256(feedback.feedback_report_sha256)
            or feedback.interface_id != FEEDBACK_INTERFACE_ID
            or feedback.feedback_owner != FEEDBACK_OWNER
            or feedback.surface_load_owner != SURFACE_LOAD_OWNER
            or feedback.force_scoring_status != FEEDBACK_SCORING_STATUS
        ):
            raise RuntimeError("STOP: nested feedback ledger is invalid")
        _assert_evaluation(
            feedback.collocation_evaluation,
            channel="collocation_rhs",
            ptera_step=ptera_step,
        )
        for channel, evaluation in zip(
            ("load_right", "load_front", "load_left", "load_back"),
            feedback.load_evaluations,
            strict=True,
        ):
            _assert_evaluation(evaluation, channel=channel, ptera_step=ptera_step)
        _assert_no_penetration(feedback)

        transport = record.transport_result
        if (
            not transport.enabled
            or transport.interface_id != PTERA_RVPM_TRANSPORT_INTERFACE_ID
            or transport.ptera_step_index != ptera_step
            or transport.dvm_for_source_step_index != source_step
            or transport.source_report_sha256 != record.row_commit_sha256
            or transport.initial_particle_count != record.particle_count
            or len(transport.stages) != 3
            or transport.self_field_call_count != 3
            or transport.ptera_center_call_count != 3
            or transport.ptera_finite_difference_call_count != 18
            or not transport.ptera_parent_state_unchanged
            or transport.feedback_write_count != 0
            or transport.parent_write_count != 0
            or transport.load_write_count != 0
            or transport.ptera_field_owner != PTERA_FIELD_OWNER
            or transport.rvpm_transport_owner != RVPM_TRANSPORT_OWNER
            or transport.force_scoring_status != TRANSPORT_SCORING_STATUS
        ):
            raise RuntimeError("STOP: nested transport ledger is invalid")
        stage_evidence = _transport_stage_evidence(record)
        if (
            stage_evidence[-1]["post_particle_state_sha256"]
            != record.transported_arrays_sha256
        ):
            raise RuntimeError(
                "STOP: stage-three/final coupling particle-state hash changed"
            )

        parent_before = record.ptera_parent_sha256_before_transport
        if (
            parent_before != record.ptera_parent_sha256_after_transport
            or parent_before != record.ptera_parent_sha256_after_raw_record
            or not record.ptera_parent_state_unchanged
            or not record.ptera_parent_state_unchanged_after_raw_record
        ):
            raise RuntimeError("STOP: Ptera parent changed during transport/raw copy")
        if (
            record.collocation_evaluation_count != 1
            or record.load_batch_evaluation_count != 4
            or record.native_load_call_count != 1
            or record.transport_call_count != 1
            or record.transport_stage_count != 3
            or record.common_transport_self_field_call_count != 3
            or record.common_transport_ptera_center_call_count != 3
            or record.common_transport_stage_count != 3
            or record.frontier_self_field_call_count != 3
            or record.frontier_ptera_center_call_count != 3
            or record.frontier_transport_stage_count != 3
        ):
            raise RuntimeError("STOP: raw top-level call ledger is invalid")
        if (
            record.observation_access != OBSERVATION_ACCESS
            or record.surface_load_owner != SURFACE_LOAD_OWNER
            or record.source_load_owner != "forbidden"
            or record.force_scoring_status != FORCE_SCORING_STATUS
        ):
            raise RuntimeError("STOP: raw-only ownership boundary changed")
        previous = record
        previous_count = record.particle_count


def _raw_step_rows(
    records: Sequence[BaikW2RawStepRecord],
    events_by_step: tuple[tuple[DVMSourceEvent, ...], ...],
    owner_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    case = BAIK_2012_CASES[CASE_ID]
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        event = events_by_step[record.source_step_index - 1][0]
        ledger = event.kelvin_ledger
        if ledger is None:
            raise RuntimeError("STOP: active raw row lacks its Kelvin ledger")
        coefficients = np.asarray(record.ptera_force_coefficients_w)
        raw_ct = float(coefficients[0])
        stage_fields: dict[str, object] = {}
        for stage in _transport_stage_evidence(record):
            stage_index = stage["stage"]
            for name, value in stage.items():
                if name != "stage":
                    stage_fields[f"stage_{stage_index}_{name}"] = value
        rows.append(
            {
                "ptera_step_index": record.ptera_step_index,
                "ptera_phase": record.ptera_step_index / STEPS_PER_CYCLE,
                "ptera_time_s": record.source_time_s,
                "source_step_index": record.source_step_index,
                "source_input_phase": (
                    (record.source_step_index - 1) / STEPS_PER_CYCLE
                ),
                "source_input_time_s": (
                    (record.source_step_index - 1) * case.period_s / STEPS_PER_CYCLE
                ),
                "birth_mode": record.birth_mode,
                "a0_pre": float(event.a0_pre),
                "a0_post": float(event.a0_post),
                "gamma_lev_new_over_u_c": event.gamma_lev_new_over_u_c,
                "gamma_lev_persisted_after_over_u_c": (
                    ledger.gamma_lev_persisted_after
                ),
                "gamma_tev_new_solved_over_u_c": (event.gamma_tev_new_solved_over_u_c),
                "gamma_tev_new_persisted_over_u_c": (
                    event.gamma_tev_new_persisted_over_u_c
                ),
                "kelvin_residual_over_u_c": event.kelvin_residual_over_u_c,
                "raw_cl_ptera_w": -float(coefficients[2]),
                "raw_ct_ptera_w": raw_ct,
                "raw_cd_ptera_w": -raw_ct,
                "particle_count": record.particle_count,
                "material_tracer_count": record.material_tracer_count,
                "material_support_tracer_count": (record.material_support_tracer_count),
                "frontier_node_tracer_count": record.frontier_node_tracer_count,
                "changed_particle_count": len(
                    owner_rows[index]["changed_particle_ids"]
                ),
                "changed_particle_ids_sha256": owner_rows[index][
                    "changed_particle_ids_sha256"
                ],
                "row_commit_sha256": record.row_commit_sha256,
                "row_state_sha256": record.row_state_sha256,
                "transported_parent_sha256": record.transported_parent_sha256 or "",
                "transported_state_sha256": record.transported_state_sha256,
                "transported_arrays_sha256": record.transported_arrays_sha256,
                "transport_parent_digest": record.transport_parent_digest,
                "common_transport_sha256": record.common_transport_sha256,
                "common_transport_attestation_sha256": (
                    record.common_transport_attestation_sha256
                ),
                "transported_material_tracers_sha256": (
                    record.transported_material_tracers_sha256
                ),
                "advanced_owner_sha256": record.advanced_owner_sha256,
                "feedback_report_sha256": (
                    record.feedback_report.feedback_report_sha256
                ),
                "ptera_parent_sha256_before_transport": (
                    record.ptera_parent_sha256_before_transport
                ),
                "ptera_parent_sha256_after_transport": (
                    record.ptera_parent_sha256_after_transport
                ),
                "ptera_parent_sha256_after_raw_record": (
                    record.ptera_parent_sha256_after_raw_record
                ),
                "collocation_evaluation_count": (record.collocation_evaluation_count),
                "load_batch_evaluation_count": record.load_batch_evaluation_count,
                "native_load_call_count": record.native_load_call_count,
                "transport_call_count": record.transport_call_count,
                "transport_stage_count": record.transport_stage_count,
                "ptera_fd_center_call_count": (
                    record.transport_result.ptera_center_call_count
                ),
                "ptera_fd_offset_call_count": (
                    record.transport_result.ptera_finite_difference_call_count
                ),
                "no_penetration_max_abs": (
                    record.feedback_report.no_penetration_max_abs
                ),
                **stage_fields,
            }
        )
    return rows


def _raw_load_rows(
    records: Sequence[BaikW2RawStepRecord],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        forces, moments = _panel_load_arrays(record)
        base = {
            "ptera_step_index": record.ptera_step_index,
            "source_step_index": record.source_step_index,
            "ptera_phase": record.ptera_step_index / STEPS_PER_CYCLE,
            "ptera_time_s": record.source_time_s,
        }
        for chordwise in range(2):
            for spanwise in range(8):
                panel_index = chordwise * 8 + spanwise
                force = forces[panel_index]
                moment = moments[panel_index]
                rows.append(
                    {
                        **base,
                        "scope": "panel",
                        "frame": "W",
                        "chordwise_index": chordwise,
                        "spanwise_index": spanwise,
                        "force_x_n": float(force[0]),
                        "force_y_n": float(force[1]),
                        "force_z_n": float(force[2]),
                        "moment_x_nm": float(moment[0]),
                        "moment_y_nm": float(moment[1]),
                        "moment_z_nm": float(moment[2]),
                        "force_coefficient_x": "",
                        "force_coefficient_y": "",
                        "force_coefficient_z": "",
                        "raw_cl_ptera_w": "",
                        "raw_ct_ptera_w": "",
                        "raw_cd_ptera_w": "",
                    }
                )
        force = np.asarray(record.ptera_forces_w)
        moment = np.asarray(record.ptera_moments_w_cgp1)
        coefficients = np.asarray(record.ptera_force_coefficients_w)
        raw_ct = float(coefficients[0])
        rows.append(
            {
                **base,
                "scope": "total",
                "frame": "W",
                "chordwise_index": "",
                "spanwise_index": "",
                "force_x_n": float(force[0]),
                "force_y_n": float(force[1]),
                "force_z_n": float(force[2]),
                "moment_x_nm": float(moment[0]),
                "moment_y_nm": float(moment[1]),
                "moment_z_nm": float(moment[2]),
                "force_coefficient_x": float(coefficients[0]),
                "force_coefficient_y": float(coefficients[1]),
                "force_coefficient_z": float(coefficients[2]),
                "raw_cl_ptera_w": -float(coefficients[2]),
                "raw_ct_ptera_w": raw_ct,
                "raw_cd_ptera_w": -raw_ct,
            }
        )
    return rows


def _write_pass_bundle(
    directory: Path,
    *,
    events: tuple[tuple[DVMSourceEvent, ...], ...],
    records: tuple[BaikW2RawStepRecord, ...],
    final_owner: ReleaseRowOwner,
    movement_metadata: dict[str, object],
    dependency_sha256: dict[str, str],
    run_provenance: dict[str, object],
) -> dict[str, object]:
    owner_rows = _owner_event_rows(final_owner)
    raw_step_rows = _raw_step_rows(records, events, owner_rows)
    source_rows = _source_event_rows(events)
    load_rows = _raw_load_rows(records)
    particle_rows = [
        {
            "ptera_step_index": record.ptera_step_index,
            "source_step_index": record.source_step_index,
            "particle_count": record.particle_count,
            "material_tracer_count": record.material_tracer_count,
            "material_support_tracer_count": record.material_support_tracer_count,
            "frontier_node_tracer_count": record.frontier_node_tracer_count,
            "common_transport_self_field_call_count": (
                record.common_transport_self_field_call_count
            ),
            "common_transport_ptera_center_call_count": (
                record.common_transport_ptera_center_call_count
            ),
            "common_transport_stage_count": record.common_transport_stage_count,
            "changed_particle_count": len(owner_rows[index]["changed_particle_ids"]),
            "appended_particle_count": len(owner_rows[index]["appended_particle_ids"]),
        }
        for index, record in enumerate(records)
    ]
    csv_payloads = {
        "raw_steps.csv": _csv_bytes(tuple(raw_step_rows[0]), raw_step_rows),
        "source_events.csv": _csv_bytes(tuple(source_rows[0]), source_rows),
        "particle_counts.csv": _csv_bytes(tuple(particle_rows[0]), particle_rows),
        "raw_loads.csv": _csv_bytes(tuple(load_rows[0]), load_rows),
    }
    for name, payload in csv_payloads.items():
        _write_bytes_atomic(directory / name, payload)
    owner_payload = b"".join(_json_bytes(row) + b"\n" for row in owner_rows)
    _write_bytes_atomic(directory / "owner_events.jsonl", owner_payload)

    raw_files = (
        "raw_steps.csv",
        "source_events.csv",
        "owner_events.jsonl",
        "particle_counts.csv",
        "raw_loads.csv",
    )
    raw_sha256 = {name: _sha256_file(directory / name) for name in raw_files}
    summary: dict[str, object] = {
        "schema_id": RUNNER_SCHEMA_ID,
        "status": "GO_v5h10_baik_w2_three_layer_raw_mechanics_only",
        "case_id": CASE_ID,
        "evaluation_type": "raw_simulation_only",
        "paper_accuracy_claim": False,
        "full_cycle_complete": False,
        "production_decision": "blocked",
        "observation_access": OBSERVATION_ACCESS,
        "force_scoring_status": FORCE_SCORING_STATUS,
        "surface_load_owner": records[0].surface_load_owner,
        "source_load_owner": records[0].source_load_owner,
        "source_to_ptera_clock": [[4, 3], [5, 4], [6, 5]],
        "raw_coefficient_labels": {
            "axes": "Ptera W: +x thrust, +z down",
            "raw_CL": "-forceCoefficients_W[2]",
            "raw_CT": "forceCoefficients_W[0]",
            "raw_CD": "-raw_CT",
            "filtering": "none",
            "fitting": "none",
        },
        "raw_rows": raw_step_rows,
        "raw_file_sha256": raw_sha256,
    }
    summary["summary_payload_sha256"] = sha256(_json_bytes(summary)).hexdigest()
    _write_bytes_atomic(directory / "summary.json", _json_bytes(summary) + b"\n")

    module_dir = Path(__file__).resolve().parent
    manifest: dict[str, object] = {
        "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "status": "GO_v5h10_baik_w2_three_layer_raw_mechanics_only",
        "case_id": CASE_ID,
        "paper_accuracy_claim": False,
        "full_cycle_complete": False,
        "production_decision": "blocked",
        "artifact_files": list(PASS_ARTIFACT_FILES),
        "observation_access": OBSERVATION_ACCESS,
        "target_read_count": 0,
        "candidate_score_authorized": False,
        "dependency_sha256": dependency_sha256,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "runner_test_sha256": _sha256_file(
            module_dir.parent / "tests" / "test_run_fluxv_v5h10_baik_w2.py"
        ),
        "movement_metadata": movement_metadata,
        "frozen": {
            "active_ptera_steps": list(FROZEN_ACTIVE_PTERA_STEPS),
            "active_source_steps": list(ACTIVE_SOURCE_STEPS),
            "active_birth_modes": list(ACTIVE_BIRTH_MODES),
            "lesp_critical": LESP_CRITICAL,
            "source_ndiv": SOURCE_NDIV,
            "source_naterm": SOURCE_NATERM,
            "source_max_wake_steps": SOURCE_MAX_WAKE_STEPS,
            "source_core_radius_chord": SOURCE_CORE_RADIUS_CHORD,
            "row_smoothing_radius_m": ROW_SMOOTHING_RADIUS_M,
            "row_target_spacing_m": ROW_TARGET_SPACING_M,
            "row_particle_cap": ROW_PARTICLE_CAP,
        },
        "result_sha256": {
            **raw_sha256,
            "summary.json": _sha256_file(directory / "summary.json"),
        },
        "run_provenance": run_provenance,
    }
    manifest["manifest_payload_sha256"] = sha256(_json_bytes(manifest)).hexdigest()
    _write_bytes_atomic(directory / "run_manifest.json", _json_bytes(manifest) + b"\n")
    log = (
        "status=GO_v5h10_baik_w2_three_layer_raw_mechanics_only\n"
        "scope=raw_only_no_ground_truth_no_scoring\n"
        "paper_accuracy_claim=false\n"
        "full_cycle_complete=false\n"
        "production_decision=blocked\n"
        "source_to_ptera=4:3,5:4,6:5\n"
        "artifact_file_count=9\n"
    ).encode("utf-8")
    _write_bytes_atomic(directory / "run.log", log)

    checksum_names = tuple(name for name in PASS_ARTIFACT_FILES if name != "SHA256SUMS")
    checksums = "".join(
        f"{_sha256_file(directory / name)}  {name}\n" for name in checksum_names
    ).encode("ascii")
    _write_bytes_atomic(directory / "SHA256SUMS", checksums)
    if set(path.name for path in directory.iterdir()) != set(PASS_ARTIFACT_FILES):
        raise RuntimeError("STOP: PASS artifact does not contain exactly nine files")
    return summary


class RawSliceStopped(RuntimeError):
    def __init__(self, error: Exception, stop_dir: Path) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.original_error = error
        self.stop_dir = stop_dir


def _publish_stop_bundle(
    staging: Path,
    output_dir: Path,
    error: Exception,
    run_provenance: dict[str, object],
) -> Path:
    for path in staging.iterdir():
        if not path.is_file():
            raise RuntimeError("refusing to clean a non-file from staging")
        path.unlink()
    payload: dict[str, object] = {
        "schema_id": RUNNER_SCHEMA_ID,
        "status": "STOP",
        "case_id": CASE_ID,
        "observation_access": OBSERVATION_ACCESS,
        "force_scoring_status": FORCE_SCORING_STATUS,
        "error_type": type(error).__name__,
        "error": str(error),
        "pass_output_dir": str(output_dir),
    }
    payload["summary_payload_sha256"] = sha256(_json_bytes(payload)).hexdigest()
    _write_bytes_atomic(staging / "summary.json", _json_bytes(payload) + b"\n")
    manifest = {
        "schema_id": RUN_MANIFEST_SCHEMA_ID,
        "status": "STOP",
        "case_id": CASE_ID,
        "target_read_count": 0,
        "candidate_score_authorized": False,
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "error_type": type(error).__name__,
        "run_provenance": run_provenance,
    }
    _write_bytes_atomic(staging / "run_manifest.json", _json_bytes(manifest) + b"\n")
    _write_bytes_atomic(
        staging / "run.log",
        f"status=STOP\nerror_type={type(error).__name__}\nerror={error}\n".encode(
            "utf-8"
        ),
    )
    names = ("summary.json", "run_manifest.json", "run.log")
    sums = "".join(
        f"{_sha256_file(staging / name)}  {name}\n" for name in names
    ).encode("ascii")
    _write_bytes_atomic(staging / "SHA256SUMS", sums)
    if set(path.name for path in staging.iterdir()) != {
        "summary.json",
        "run_manifest.json",
        "run.log",
        "SHA256SUMS",
    }:
        raise RuntimeError("STOP bundle contains stale PASS files")
    stop_dir = output_dir.with_name(f"{output_dir.name}.STOP-{staging.name[-12:]}")
    _publish_directory_noreplace(staging, stop_dir)
    return stop_dir


def run_raw_slice(
    output_dir: Path, *, invocation_argv: Sequence[str] | None = None
) -> dict[str, object]:
    """Run once in staging and atomically publish an exact nine-file bundle."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    run_provenance = _run_provenance(output_dir, invocation_argv)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        dependency_sha256 = _assert_runtime_dependency_manifest()
        events = build_w2_source_events()
        movement, movement_metadata = build_baik_movement(
            BAIK_2012_CASES[CASE_ID], "smoke"
        )
        problem = ps.problems.UnsteadyProblem(
            movement=movement,
            only_final_results=False,
        )
        committer = BaikW2RawRowCommitter(events)
        solver = make_fluxv_v5h10_baik_w2_solver(
            problem,
            row_committer=committer,
            max_particles=ROW_PARTICLE_CAP,
            stretch=False,
            free_wake=False,
        )
        committer.bind_solver(solver)
        solver.run(
            prescribed_wake=True,
            calculate_streamlines=False,
            show_progress=False,
        )
        records = tuple(solver.v5h10_raw_step_records)
        if len(records) != 3:
            raise RuntimeError("STOP: bounded coupling did not produce three layers")
        final_owner = solver._v5h10_previous_owner
        if type(final_owner) is not ReleaseRowOwner:
            raise RuntimeError("STOP: coupling did not expose an exact final owner")
        _assert_raw_records(records, committer, final_owner)
        summary = _write_pass_bundle(
            staging,
            events=events,
            records=records,
            final_owner=final_owner,
            movement_metadata=movement_metadata,
            dependency_sha256=dependency_sha256,
            run_provenance=run_provenance,
        )
        _publish_directory_noreplace(staging, output_dir)
        return summary
    except Exception as error:
        stop_dir = _publish_stop_bundle(staging, output_dir, error, run_provenance)
        raise RawSliceStopped(error, stop_dir) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/fluxv-v5h10-baik-w2-raw"),
    )
    args = parser.parse_args(argv)
    try:
        summary = run_raw_slice(args.output_dir, invocation_argv=tuple(sys.argv))
    except RawSliceStopped as error:
        print(f"STOP {error}; isolated_bundle={error.stop_dir}")
        return 2
    except FileExistsError as error:
        print(f"STOP {error}; no files written")
        return 2
    print(
        "GO_v5h10_baik_w2_three_layer_raw_mechanics_only; "
        f"summary={args.output_dir.resolve() / 'summary.json'}; "
        f"payload_sha256={summary['summary_payload_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
