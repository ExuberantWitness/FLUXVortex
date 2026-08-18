"""Run the observation-free FluxV v5h7 temporal-topology oracle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from fluxvortex.rvpm_edge_bridge import (
    BridgeNode,
    DirectedRing,
    EdgeGraph,
    ShadowBridgeResult,
    assemble_ring_edge_graph,
    deposit_edge_graph_prescribed_sigma_and_spacing,
)
from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian


Topology = Literal["independent", "connected"]

RUN_ID: Final = "20260815_fluxv_v5h7_temporal_topology_oracle"
SCHEMA_ID: Final = "fluxv-v5h7-temporal-topology-oracle-v1"
PHYSICAL_HORIZON_S: Final = 0.08
SPEED_M_PER_S: Final = 2.0
SPAN_M: Final = 0.60
GAMMA_RATE_M2_PER_S2: Final = 0.8
SMOOTHING_RADIUS_M: Final = 0.085
TARGET_SPACING_M: Final = 0.02
DELTA_TIME_S: Final = (0.02, 0.01, 0.005, 0.0025)
HALVING_LOWER: Final = 0.49
HALVING_UPPER: Final = 0.51
CONNECTED_RATIO_LOWER: Final = 1.8
CONNECTED_RATIO_UPPER: Final = 2.2
RICHARDSON_RELATIVE_LIMIT: Final = 0.01
MIN_EXTRAPOLATED_FIELD_NORM: Final = 1.0e-4
MAX_PARTICLES: Final = 5_000
FIXED_PROBES_GP1_M: Final = np.asarray(
    ((0.05, 0.30, 0.30), (0.10, 0.15, 0.40), (0.20, 0.45, 0.50)),
    dtype=np.float64,
)
FIXED_PROBES_GP1_M.setflags(write=False)


@dataclass(frozen=True, slots=True)
class OracleRow:
    topology: Topology
    delta_time_s: float
    interval_count: int
    particle_count: int
    source_edge_count: int
    retained_edge_count: int
    two_incidence_edge_count: int
    delta_gamma_m2_per_s: float
    impulse_gp1_m4_per_s: np.ndarray
    analytic_impulse_gp1_m4_per_s: np.ndarray | None
    probe_velocity_gp1_m_per_s: np.ndarray
    max_shared_edge_gamma_residual: float
    max_conservation_abs: float
    mechanics_passed: bool


def _strict_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _interval_count(delta_time_s: float) -> int:
    if isinstance(delta_time_s, bool):
        raise TypeError("delta_time_s must be a finite positive float")
    value = float(delta_time_s)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("delta_time_s must be finite and positive")
    count = int(round(PHYSICAL_HORIZON_S / value))
    if count < 1 or abs(count * value - PHYSICAL_HORIZON_S) > 1.0e-15:
        raise ValueError("delta_time_s must divide the fixed physical horizon")
    return count


def _independent_graph(delta_time_s: float) -> EdgeGraph:
    count = _interval_count(delta_time_s)
    delta_x = SPEED_M_PER_S * delta_time_s
    delta_gamma = GAMMA_RATE_M2_PER_S2 * delta_time_s
    nodes: list[BridgeNode] = []
    rings: list[DirectedRing] = []
    for interval in range(count):
        for plane, x_position in (
            ("upstream", interval * delta_x),
            ("downstream", (interval + 1) * delta_x),
        ):
            nodes.extend(
                (
                    BridgeNode(
                        f"independent:{interval}:{plane}:left",
                        (x_position, 0.0, 0.0),
                    ),
                    BridgeNode(
                        f"independent:{interval}:{plane}:right",
                        (x_position, SPAN_M, 0.0),
                    ),
                )
            )
        rings.append(
            DirectedRing(
                ring_id=f"independent:ring:{interval}",
                node_ids=(
                    f"independent:{interval}:upstream:left",
                    f"independent:{interval}:upstream:right",
                    f"independent:{interval}:downstream:right",
                    f"independent:{interval}:downstream:left",
                ),
                circulation=delta_gamma,
            )
        )
    return assemble_ring_edge_graph(tuple(nodes), tuple(rings))


def _connected_graph(delta_time_s: float) -> EdgeGraph:
    count = _interval_count(delta_time_s)
    delta_x = SPEED_M_PER_S * delta_time_s
    delta_gamma = GAMMA_RATE_M2_PER_S2 * delta_time_s
    nodes: list[BridgeNode] = []
    for plane in range(count + 1):
        x_position = plane * delta_x
        nodes.extend(
            (
                BridgeNode(f"connected:{plane}:left", (x_position, 0.0, 0.0)),
                BridgeNode(f"connected:{plane}:right", (x_position, SPAN_M, 0.0)),
            )
        )
    rings = tuple(
        DirectedRing(
            ring_id=f"connected:ring:{interval}",
            node_ids=(
                f"connected:{interval}:left",
                f"connected:{interval}:right",
                f"connected:{interval + 1}:right",
                f"connected:{interval + 1}:left",
            ),
            circulation=(interval + 1) * delta_gamma,
        )
        for interval in range(count)
    )
    return assemble_ring_edge_graph(tuple(nodes), rings)


def _deposit(graph: EdgeGraph) -> ShadowBridgeResult:
    result = deposit_edge_graph_prescribed_sigma_and_spacing(
        graph,
        smoothing_radius=SMOOTHING_RADIUS_M,
        target_spacing=TARGET_SPACING_M,
        step=0,
    )
    if result.diagnostics.particle_count > MAX_PARTICLES:
        raise RuntimeError("v5h7 particle cap exceeded")
    return result


def _analytic_connected_impulse(delta_time_s: float) -> np.ndarray:
    magnitude = (
        GAMMA_RATE_M2_PER_S2
        * SPEED_M_PER_S
        * SPAN_M
        * (PHYSICAL_HORIZON_S**2 + PHYSICAL_HORIZON_S * delta_time_s)
        / 2.0
    )
    return np.asarray((0.0, 0.0, -magnitude), dtype=np.float64)


def _run_row(topology: Topology, delta_time_s: float) -> OracleRow:
    graph = (
        _independent_graph(delta_time_s)
        if topology == "independent"
        else _connected_graph(delta_time_s)
    )
    deposited = _deposit(graph)
    delta_gamma = GAMMA_RATE_M2_PER_S2 * delta_time_s
    shared_edges = tuple(edge for edge in graph.edges if len(edge.incidences) == 2)
    shared_residual = max(
        (abs(edge.circulation - delta_gamma) for edge in shared_edges), default=0.0
    )
    state_finite = (
        np.all(np.isfinite(deposited.positions))
        and np.all(np.isfinite(deposited.gamma))
        and np.all(np.isfinite(deposited.sigma))
        and np.all(deposited.sigma > 0.0)
    )
    impulse = 0.5 * np.sum(np.cross(deposited.positions, deposited.gamma), axis=0)
    field = direct_gaussian_erf_velocity_jacobian(
        deposited.positions,
        deposited.gamma,
        deposited.sigma,
        target_positions=FIXED_PROBES_GP1_M,
    ).velocity
    analytic = (
        None if topology == "independent" else _analytic_connected_impulse(delta_time_s)
    )
    topology_passed = (
        len(shared_edges) == 0
        if topology == "independent"
        else (
            len(shared_edges) == _interval_count(delta_time_s) - 1
            and shared_residual <= 1.0e-14
            and all(len(edge.incidences) == 2 for edge in shared_edges)
        )
    )
    conservation = max(
        deposited.diagnostics.max_edge_conservation_abs,
        deposited.diagnostics.global_conservation_abs,
        graph.incidence_residual,
        graph.edge_reconstruction_residual,
    )
    mechanics = (
        state_finite
        and np.all(np.isfinite(impulse))
        and np.all(np.isfinite(field))
        and topology_passed
        and conservation <= 1.0e-14
        and deposited.diagnostics.clip_count == 0
        and deposited.diagnostics.nonfinite_count == 0
        and deposited.diagnostics.owner_conflict_count == 0
        and deposited.diagnostics.feedback_call_count == 0
        and (
            analytic is None
            or float(np.max(np.abs(impulse - analytic), initial=0.0)) <= 1.0e-14
        )
    )
    return OracleRow(
        topology=topology,
        delta_time_s=delta_time_s,
        interval_count=_interval_count(delta_time_s),
        particle_count=deposited.diagnostics.particle_count,
        source_edge_count=deposited.diagnostics.source_edge_count,
        retained_edge_count=deposited.diagnostics.retained_edge_count,
        two_incidence_edge_count=len(shared_edges),
        delta_gamma_m2_per_s=delta_gamma,
        impulse_gp1_m4_per_s=np.asarray(impulse, dtype=np.float64),
        analytic_impulse_gp1_m4_per_s=analytic,
        probe_velocity_gp1_m_per_s=np.asarray(field, dtype=np.float64),
        max_shared_edge_gamma_residual=shared_residual,
        max_conservation_abs=conservation,
        mechanics_passed=bool(mechanics),
    )


def _halving_ratios(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(values[index + 1] / values[index] for index in range(len(values) - 1))


def _difference_ratios(arrays: tuple[np.ndarray, ...]) -> tuple[float, ...]:
    differences = tuple(
        float(np.linalg.norm(arrays[index] - arrays[index + 1]))
        for index in range(len(arrays) - 1)
    )
    return tuple(
        differences[index] / differences[index + 1]
        for index in range(len(differences) - 1)
    )


def _row_payload(row: OracleRow) -> dict[str, Any]:
    return {
        "topology": row.topology,
        "delta_time_s": row.delta_time_s,
        "interval_count": row.interval_count,
        "particle_count": row.particle_count,
        "source_edge_count": row.source_edge_count,
        "retained_edge_count": row.retained_edge_count,
        "two_incidence_edge_count": row.two_incidence_edge_count,
        "delta_gamma_m2_per_s": row.delta_gamma_m2_per_s,
        "impulse_gp1_m4_per_s": row.impulse_gp1_m4_per_s.tolist(),
        "analytic_impulse_gp1_m4_per_s": (
            None
            if row.analytic_impulse_gp1_m4_per_s is None
            else row.analytic_impulse_gp1_m4_per_s.tolist()
        ),
        "probe_velocity_gp1_m_per_s": row.probe_velocity_gp1_m_per_s.tolist(),
        "max_shared_edge_gamma_residual": row.max_shared_edge_gamma_residual,
        "max_conservation_abs": row.max_conservation_abs,
        "mechanics_passed": row.mechanics_passed,
    }


def run_gate() -> dict[str, Any]:
    independent = tuple(_run_row("independent", dt) for dt in DELTA_TIME_S)
    connected = tuple(_run_row("connected", dt) for dt in DELTA_TIME_S)

    independent_impulse_norm = tuple(
        float(np.linalg.norm(row.impulse_gp1_m4_per_s)) for row in independent
    )
    independent_probe_norm = tuple(
        float(np.linalg.norm(row.probe_velocity_gp1_m_per_s)) for row in independent
    )
    independent_impulse_halving = _halving_ratios(independent_impulse_norm)
    independent_probe_halving = _halving_ratios(independent_probe_norm)
    independent_pathology_passed = all(
        HALVING_LOWER <= ratio <= HALVING_UPPER
        for ratio in (*independent_impulse_halving, *independent_probe_halving)
    )

    impulse_limit = np.asarray(
        (
            0.0,
            0.0,
            -GAMMA_RATE_M2_PER_S2
            * SPEED_M_PER_S
            * SPAN_M
            * PHYSICAL_HORIZON_S**2
            / 2.0,
        )
    )
    impulse_errors = tuple(
        float(np.linalg.norm(row.impulse_gp1_m4_per_s - impulse_limit))
        for row in connected
    )
    connected_impulse_ratios = tuple(
        impulse_errors[index] / impulse_errors[index + 1]
        for index in range(len(impulse_errors) - 1)
    )
    connected_fields = tuple(row.probe_velocity_gp1_m_per_s for row in connected)
    connected_probe_ratios = _difference_ratios(connected_fields)
    extrapolated_middle = 2.0 * connected_fields[2] - connected_fields[1]
    extrapolated_fine = 2.0 * connected_fields[3] - connected_fields[2]
    extrapolated_norm = float(np.linalg.norm(extrapolated_fine))
    richardson_relative = float(
        np.linalg.norm(extrapolated_middle - extrapolated_fine) / extrapolated_norm
    )
    connected_limit_passed = (
        all(
            1.999999999999 <= ratio <= 2.000000000001
            for ratio in connected_impulse_ratios
        )
        and all(
            CONNECTED_RATIO_LOWER <= ratio <= CONNECTED_RATIO_UPPER
            for ratio in connected_probe_ratios
        )
        and richardson_relative <= RICHARDSON_RELATIVE_LIMIT
        and extrapolated_norm >= MIN_EXTRAPOLATED_FIELD_NORM
    )
    mechanics = all(row.mechanics_passed for row in (*independent, *connected))
    passed = mechanics and independent_pathology_passed and connected_limit_passed
    return {
        "schema_id": SCHEMA_ID,
        "run_id": RUN_ID,
        "status": (
            "go_temporal_connected_topology_oracle_mechanics_only"
            if passed
            else "stop_v5h7_temporal_topology_oracle_failed"
        ),
        "passed": passed,
        "scope": {
            "evaluation": "manufactured_simulation_only",
            "observation_access": "none",
            "target_case_branch": "none",
            "ptera_solver_call_count": 0,
            "load_call_count": 0,
            "feedback_call_count": 0,
            "claim_limit": (
                "manufactured temporal-topology oracle only; no production "
                "solver, stability, load, or accuracy claim"
            ),
        },
        "contract": {
            "physical_horizon_s": PHYSICAL_HORIZON_S,
            "speed_m_per_s": SPEED_M_PER_S,
            "span_m": SPAN_M,
            "gamma_rate_m2_per_s2": GAMMA_RATE_M2_PER_S2,
            "smoothing_radius_m": SMOOTHING_RADIUS_M,
            "target_spacing_m": TARGET_SPACING_M,
            "delta_time_s": list(DELTA_TIME_S),
            "halving_interval": [HALVING_LOWER, HALVING_UPPER],
            "connected_difference_ratio_interval": [
                CONNECTED_RATIO_LOWER,
                CONNECTED_RATIO_UPPER,
            ],
            "richardson_relative_limit": RICHARDSON_RELATIVE_LIMIT,
            "minimum_extrapolated_field_norm": MIN_EXTRAPOLATED_FIELD_NORM,
            "particle_cap": MAX_PARTICLES,
        },
        "rows": [_row_payload(row) for row in (*independent, *connected)],
        "diagnostics": {
            "independent_impulse_norm": list(independent_impulse_norm),
            "independent_probe_norm": list(independent_probe_norm),
            "independent_impulse_halving_ratios": list(independent_impulse_halving),
            "independent_probe_halving_ratios": list(independent_probe_halving),
            "connected_impulse_limit_gp1_m4_per_s": impulse_limit.tolist(),
            "connected_impulse_errors": list(impulse_errors),
            "connected_impulse_error_ratios": list(connected_impulse_ratios),
            "connected_probe_difference_ratios": list(connected_probe_ratios),
            "connected_probe_richardson_relative_difference": richardson_relative,
            "connected_probe_extrapolated_norm": extrapolated_norm,
        },
        "gates": {
            "all_row_mechanics_passed": mechanics,
            "independent_closed_ring_pathology_confirmed": (
                independent_pathology_passed
            ),
            "connected_cumulative_sheet_limit_passed": connected_limit_passed,
            "target_access_count": 0,
            "ptera_solver_call_count": 0,
            "load_call_count": 0,
        },
    }


def _source_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / relative
        for relative in (
            "platform/forward_flight_benchmarks/run_fluxv_v5h7_temporal_topology_oracle.py",
            "platform/tests/test_run_fluxv_v5h7_temporal_topology_oracle.py",
            "src/fluxvortex/rvpm_edge_bridge.py",
            "src/fluxvortex/rvpm_reference.py",
        )
    )


def write_artifact(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {output_dir}")
    output_dir.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    summary = run_gate()
    (output_dir / "summary.json").write_text(_strict_json(summary), encoding="utf-8")
    source_manifest = {
        "schema_id": "source-sha256-v1",
        "files": {
            str(path.relative_to(root)): _file_sha256(path)
            for path in _source_paths(root)
        },
    }
    (output_dir / "source_manifest.json").write_text(
        _strict_json(source_manifest), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# FluxV v5h7 temporal-topology oracle\n\n"
        "Manufactured, observation-free topology evidence only. No Ptera solve, "
        "target data, load, feedback, or aerodynamic-accuracy claim is present.\n",
        encoding="utf-8",
    )
    payloads = ("README.md", "source_manifest.json", "summary.json")
    result_manifest = {name: _file_sha256(output_dir / name) for name in payloads}
    (output_dir / "result_manifest.json").write_text(
        _strict_json({"schema_id": "result-sha256-v1", "files": result_manifest}),
        encoding="utf-8",
    )
    checksum_files = (*payloads, "result_manifest.json")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(
            f"{_file_sha256(output_dir / name)}  {name}" for name in checksum_files
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    summary = write_artifact(arguments.output_dir)
    print(_strict_json(summary), end="")
    return 0 if bool(summary["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
