"""No-force canonical diagnostic for first-LEV-ring placement claims.

The runner stops conceptually at the first Hirato Eq.6 event.  It compares
the historical full-freestream-step geometry with the two preregistered
published identities, but it does not insert any candidate ring into the
UVLM, solve for Gamma_L, or calculate aerodynamic force.

``P-A-kinematic`` is intentionally *not* a faithful Ansari finite-wing
candidate: it supplies only freestream minus leading-edge body velocity for
the otherwise published half-step identity.  The missing bound/TEV/LEV local
edge-velocity adapter remains visible in the output and prevents promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
import yaml

from _v2_robo import gpu_run_twist
from claim_runtime.hirato_equations import (
    embed_chord_normal_displacement,
    first_lev_displacement_ramesh_2d,
    first_vortex_displacement_ansari,
)


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "docs" / "diag" / "hirato_canonical_cases.yaml"


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError(f"{name} contains a zero or non-finite vector")
    return vector / norm[..., None]


def _strip_basis(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct an explicit chord/span orthonormal basis at strip centers."""
    left_chord = corners[-1, :-1] - corners[0, :-1]
    right_chord = corners[-1, 1:] - corners[0, 1:]
    chord_tangent = _unit(left_chord + right_chord, "chord tangent")
    leading_span = _unit(
        corners[0, 1:] - corners[0, :-1],
        "leading-edge span tangent",
    )
    suction_normal = _unit(
        np.cross(chord_tangent, leading_span),
        "suction normal",
    )
    # Recompute the chord direction in the plane normal to span.  This is an
    # explicit geometric construction, not silent normalization inside the
    # published-formula adapter.
    chord_tangent = _unit(
        np.cross(leading_span, suction_normal),
        "orthogonal chord tangent",
    )
    return chord_tangent, suction_normal


def _ring_metrics(
    leading_edges: np.ndarray,
    displacement: np.ndarray,
    chord: float,
) -> dict:
    aft = leading_edges + displacement[:, None, :]
    rings = np.stack(
        [leading_edges[:, 0], leading_edges[:, 1], aft[:, 1], aft[:, 0]],
        axis=1,
    )
    edges = np.roll(rings, -1, axis=1) - rings
    lengths = np.linalg.norm(edges, axis=-1)
    area_vector = 0.5 * np.sum(
        np.cross(rings, np.roll(rings, -1, axis=1)),
        axis=1,
    )
    return {
        "displacement_over_c": (np.linalg.norm(displacement, axis=1) / chord).tolist(),
        "minimum_edge_over_c": (np.min(lengths, axis=1) / chord).tolist(),
        "area_over_c2": (np.linalg.norm(area_vector, axis=1) / chord**2).tolist(),
        "finite_nonzero": bool(
            np.all(np.isfinite(rings))
            and np.all(np.min(lengths, axis=1) > 0.0)
            and np.all(np.linalg.norm(area_vector, axis=1) > 0.0)
        ),
    }


def run_case(
    *,
    case_name: str,
    tip_twist_deg: float,
    nc: int,
    ns: int,
    steps: int,
    spec: dict,
) -> dict:
    shared = spec["shared"]
    chord = 0.1
    half_span = 0.5 * shared["aspect_ratio"] * chord
    speed = shared["chord_reynolds"] * 1.5e-5 / chord
    frequency = speed / (3.0 * chord)
    dt = 1.0 / (frequency * steps)
    frames: list[dict] = []
    result = gpu_run_twist(
        U=speed,
        aoa_deg=0.0,
        freq=frequency,
        n_cycle=1,
        steps_per_cycle=steps,
        wake_rows=steps,
        nc=nc,
        ns=ns,
        chord=chord,
        half_span=half_span,
        flap_amp_deg=0.0,
        twist_amp_deg=tip_twist_deg,
        real_geom=False,
        section_geometry="sd7003",
        sym=True,
        pitch_ramp=True,
        pitch_max=shared["alpha_end_deg"],
        pitch_K=shared["reduced_pitch_rate_K"],
        pitch_t0star=1.0,
        lev_shed_mode="hirato_probe",
        a0_crit=shared["lesp_crit"],
        d_para=0.0,
        frames_out=frames,
        frame_skip=1,
    )
    audit = result["n3_hirato_audit"]
    first = next((row for row in audit if np.any(row["shed_eq6_paper"])), None)
    if first is None:
        return {"case": case_name, "gate": "fail-no-onset"}
    step = int(first["step"])
    frame = frames[step]
    corners = np.asarray(frame["corners"])
    corner_velocity = np.asarray(frame["corner_velocity"])
    active = np.flatnonzero(first["shed_eq6_paper"])
    chord_tangent, suction_normal = _strip_basis(corners)
    leading_edges = np.stack(
        [corners[0, :-1], corners[0, 1:]],
        axis=1,
    )[active]

    # Strip-center kinematic relative velocity.  This deliberately omits the
    # vortex-induced local edge velocity required to complete P-A.
    edge_body_velocity = 0.5 * (
        corner_velocity[0, :-1] + corner_velocity[0, 1:]
    )
    q_kinematic = np.array([speed, 0.0, 0.0]) - edge_body_velocity
    displacement_pa = first_vortex_displacement_ansari(
        q_kinematic[active],
        dt,
    )

    local_alpha = np.arctan2(
        -chord_tangent[:, 2],
        chord_tangent[:, 0],
    )
    displacement_pr_2d = first_lev_displacement_ramesh_2d(
        speed,
        np.asarray(first["A0_eq6_paper"])[active],
        local_alpha[active],
        dt,
    )
    displacement_pr = embed_chord_normal_displacement(
        displacement_pr_2d,
        chord_tangent[active],
        suction_normal[active],
    )
    displacement_historical = np.broadcast_to(
        np.array([speed * dt, 0.0, 0.0]),
        displacement_pr.shape,
    ).copy()

    tangent_norm_residual = np.abs(
        np.linalg.norm(chord_tangent[active], axis=1) - 1.0
    )
    normal_norm_residual = np.abs(
        np.linalg.norm(suction_normal[active], axis=1) - 1.0
    )
    basis_dot_residual = np.abs(
        np.einsum(
            "ij,ij->i",
            chord_tangent[active],
            suction_normal[active],
        )
    )
    return {
        "case": case_name,
        "role": "formula/geometry only; no LEV insertion, Gamma_L solve, or force",
        "grid": {"nc": nc, "ns": ns, "steps": steps},
        "step": step,
        "tstar": 3.0 * step / steps,
        "active_strips": active.tolist(),
        "active_strip_centers_y_over_halfspan": ((active + 0.5) / ns).tolist(),
        "A0_eq6": np.asarray(first["A0_eq6_paper"])[active].tolist(),
        "local_alpha_deg": np.degrees(local_alpha[active]).tolist(),
        "basis_max_residual": {
            "tangent_norm": float(np.max(tangent_norm_residual)),
            "normal_norm": float(np.max(normal_norm_residual)),
            "dot": float(np.max(basis_dot_residual)),
        },
        "historical_full_freestream_step": {
            "claim_state": "falsified",
            **_ring_metrics(leading_edges, displacement_historical, chord),
        },
        "P-A-kinematic": {
            "claim_state": "partial-ineligible",
            "missing_component": "complete local shedding-edge vortex velocity and complex-plane-to-3D identity",
            **_ring_metrics(leading_edges, displacement_pa, chord),
        },
        "P-R": {
            "claim_state": "partial",
            **_ring_metrics(leading_edges, displacement_pr, chord),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=8)
    parser.add_argument("--ns", type=int, default=24)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    if args.nc <= 0 or args.ns <= 0 or args.steps <= 0:
        parser.error("nc, ns and steps must be positive")

    spec = yaml.safe_load(SPEC_PATH.read_text())
    wp.init()
    records = []
    for case_name in ("case1", "case2"):
        case = spec["cases"][case_name]
        record = run_case(
            case_name=case_name,
            tip_twist_deg=case["tip_twist_deg"],
            nc=args.nc,
            ns=args.ns,
            steps=args.steps,
            spec=spec,
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    payload = {
        "source": str(SPEC_PATH.relative_to(HERE.parent)),
        "records": records,
        "all_formula_geometry_finite": all(
            row.get("historical_full_freestream_step", {}).get(
                "finite_nonzero", False
            )
            and row.get("P-A-kinematic", {}).get("finite_nonzero", False)
            and row.get("P-R", {}).get("finite_nonzero", False)
            for row in records
        ),
        "promotion": "none",
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0 if payload["all_formula_geometry_finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
