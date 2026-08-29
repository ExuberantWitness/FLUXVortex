"""Frozen oracle loader for the Rojratsirikul 2011 digitized figure package.

Source package (HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829):

    artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/observations/
      figure_digitization_20260829/

The five CSVs are the experiment oracles for the unified-framework
reproduction: Figure 6 (time-mean membrane zmax/c), Figure 9 (flexible and
rigid Cn), Figure 12 (rigid-wake velocity spectrum), Figures 13/15 (rigid
finite-wing shedding St and St*sin(alpha)), and Figure 14 (cross-literature
diagnostic relation only).  Every value carries ``evidence_role`` and a
digitization uncertainty; they are figure inversions, not author tables.

Contract (handoff §7 P0):
  * validate SHA-256, row counts, finiteness and key uniqueness before any
    comparison; drift raises (non-zero exit for callers);
  * unique index (figure, wing, U/Re, alpha); exact-match lookups only --
    a missing angle is NEVER silently replaced by a neighbour;
  * Figure 14 is a cross-literature diagnostic band, not this experiment's
    ground truth;
  * the legacy sparse CSV stays as a historical record; the A10 Figure 9
    anchor correction (0.50-0.52 -> 0.5569) is documented, not overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
OBSERVATIONS_DIR = (
    _REPO_ROOT
    / "artifacts/experiment/20260824_rojratsirikul2011_q16_fsi/observations/figure_digitization_20260829"
)

FIGURE06_CSV = OBSERVATIONS_DIR / "figure06_displacement_digitized.csv"
FIGURE09_CSV = OBSERVATIONS_DIR / "figure09_normal_force_digitized.csv"
FIGURE12_CSV = OBSERVATIONS_DIR / "figure12_wake_spectrum_digitized.csv"
FIGURE13_15_CSV = OBSERVATIONS_DIR / "figure13_15_rigid_wake_reference.csv"
FIGURE14_CSV = OBSERVATIONS_DIR / "figure14_2d_reference_relation.csv"

# ── Frozen identity (handoff §7 P0 / §13 self-check 2026-08-29) ───────────
FROZEN_ROW_COUNTS = {
    "figure06": 78,
    "figure09": 186,
    "figure12": 301,
    "figure13_15": 35,
    "figure14": 55,
}
FROZEN_SHA256 = {
    "figure06": "26c9ba7396638972bd79d51e5d8aaffd703b7f3ef8a6aded7edc4c85c3209611",
    "figure09": "612ec6f33ac13a8e0b293911367a279dbefa0b58eba77f4caedcf1c6364cc961",
    "figure12": "eeb08356841ededa2b8b7e60bc6b483bd60198d098c9b6de36937d4c7d031ca8",
    "figure13_15": "00df5c7d363385965458eddc39d2eed42d88cd5833004ec9ce8308947504fd3d",
    "figure14": "4fba51d05120bad0fb98ae5842f519cb29c907a450607cbe19b57edb2928d14d",
}
FIGURE12_PEAK_ST = 0.58
FIGURE12_PEAK_UNCERTAINTY = 0.02
FIGURE12_CONDITION = {"AR": 2, "alpha_deg": 15.0, "Re": 48700}
A16_U5_ZMAX_OVER_C = 0.04338
A16_U5_CN = 0.9200
A10_U5_CN = 0.5569  # corrected from the legacy sparse anchor 0.50-0.52
# Digitized St* closure tolerance: the package self-check reports
# max|St_modified - St*sin(alpha)| = 1.93e-05 from figure rounding.
ST_MODIFIED_CLOSURE_TOL = 1.0e-4


@dataclass(frozen=True, slots=True)
class Figure6Row:
    U_m_s: float
    Re: int
    alpha_deg: float
    zmax_over_c: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class Figure9Row:
    wing_type: str  # flexible_membrane | rigid_flat_plate
    U_m_s: float
    Re: int
    alpha_deg: float
    cn: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class Figure1315Row:
    Re: int
    alpha_deg: float
    st: float
    st_modified: float
    uncertainty_st: float
    uncertainty_st_modified: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"digitized oracle missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite(value: str, field: str, path: Path) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {field} in {path.name}")
    return number


def validate() -> dict[str, str]:
    """Validate identity, counts, finiteness and key uniqueness.

    Returns the observed SHA-256 map (callers can freeze it).  Raises on any
    drift, so a wrapper CLI exits non-zero exactly as the handoff requires.
    """

    paths = {
        "figure06": FIGURE06_CSV,
        "figure09": FIGURE09_CSV,
        "figure12": FIGURE12_CSV,
        "figure13_15": FIGURE13_15_CSV,
        "figure14": FIGURE14_CSV,
    }
    observed_sha: dict[str, str] = {}
    for key, path in paths.items():
        observed_sha[key] = _sha256(path)
        if observed_sha[key] != FROZEN_SHA256[key]:
            raise RuntimeError(
                f"{key} oracle drift: sha256 {observed_sha[key]} != "
                f"frozen {FROZEN_SHA256[key]}"
            )
        rows = _read_rows(path)
        if len(rows) != FROZEN_ROW_COUNTS[key]:
            raise RuntimeError(
                f"{key} oracle row count {len(rows)} != frozen {FROZEN_ROW_COUNTS[key]}"
            )
        if key == "figure06":
            seen: set[tuple[float, float]] = set()
            for row in rows:
                key_tuple = (float(row["U_m_s"]), float(row["alpha_deg"]))
                if key_tuple in seen:
                    raise ValueError(f"figure06 duplicate key {key_tuple}")
                seen.add(key_tuple)
                _finite(row["zmax_over_c"], "zmax_over_c", path)
        elif key == "figure09":
            seen = set()
            for row in rows:
                key_tuple = (
                    row["wing_type"],
                    float(row["U_m_s"]),
                    float(row["alpha_deg"]),
                )
                if key_tuple in seen:
                    raise ValueError(f"figure09 duplicate key {key_tuple}")
                seen.add(key_tuple)
                _finite(row["Cn"], "Cn", path)
        elif key == "figure12":
            for row in rows:
                _finite(row["fc_over_U"], "fc_over_U", path)
                _finite(row["PSD"], "PSD", path)
        elif key == "figure13_15":
            seen13 = set()
            for row in rows:
                key_tuple = (int(row["Re"]), float(row["alpha_deg"]))
                if row["figures"] == "12|13|15":
                    continue  # single anchor row, not part of the curve grid
                if key_tuple in seen13:
                    raise ValueError(f"figure13_15 duplicate key {key_tuple}")
                seen13.add(key_tuple)
                st = _finite(row["St_fc_over_U"], "St", path)
                st_mod = _finite(row["St_modified"], "St_modified", path)
                alpha = float(row["alpha_deg"])
                closure = abs(st_mod - st * math.sin(math.radians(alpha)))
                if closure > ST_MODIFIED_CLOSURE_TOL:
                    raise ValueError(
                        f"figure13_15 St* closure {closure:.2e} exceeds "
                        f"{ST_MODIFIED_CLOSURE_TOL:.0e} at Re={row['Re']} "
                        f"alpha={alpha}"
                    )
        elif key == "figure14":
            for row in rows:
                _finite(row["St_fit_0p17_over_sin_alpha"], "fit", path)
    return observed_sha


def freeze_sha256() -> None:
    """One-time helper: validate then print the map to paste into FROZEN_SHA256."""

    observed = validate()
    for key, digest in observed.items():
        print(f'    "{key}": "{digest}",')


def figure6_rows() -> tuple[Figure6Row, ...]:
    return tuple(
        Figure6Row(
            U_m_s=float(row["U_m_s"]),
            Re=int(row["Re"]),
            alpha_deg=float(row["alpha_deg"]),
            zmax_over_c=_finite(row["zmax_over_c"], "zmax_over_c", FIGURE06_CSV),
            uncertainty=float(row["digitization_uncertainty"]),
        )
        for row in _read_rows(FIGURE06_CSV)
    )


def figure9_rows() -> tuple[Figure9Row, ...]:
    return tuple(
        Figure9Row(
            wing_type=row["wing_type"],
            U_m_s=float(row["U_m_s"]),
            Re=int(row["Re"]),
            alpha_deg=float(row["alpha_deg"]),
            cn=_finite(row["Cn"], "Cn", FIGURE09_CSV),
            uncertainty=float(row["digitization_uncertainty"]),
        )
        for row in _read_rows(FIGURE09_CSV)
    )


def figure6_value(U_m_s: float, alpha_deg: float) -> Figure6Row:
    """Exact-match Figure 6 lookup (no neighbour substitution)."""

    for row in figure6_rows():
        if row.U_m_s == float(U_m_s) and row.alpha_deg == float(alpha_deg):
            return row
    raise KeyError(
        f"figure06 has no exact (U={U_m_s}, alpha={alpha_deg}) sample; "
        "neighbour substitution is forbidden"
    )


def figure9_value(wing_type: str, U_m_s: float, alpha_deg: float) -> Figure9Row:
    """Exact-match Figure 9 lookup (no neighbour substitution)."""

    for row in figure9_rows():
        if (
            row.wing_type == wing_type
            and row.U_m_s == float(U_m_s)
            and row.alpha_deg == float(alpha_deg)
        ):
            return row
    raise KeyError(
        f"figure09 has no exact (wing={wing_type}, U={U_m_s}, alpha={alpha_deg}) "
        "sample; neighbour substitution is forbidden"
    )


def figure12_spectrum() -> tuple[list[float], list[float], dict[str, float]]:
    """Return (fc_over_U, PSD) samples plus the condition block."""

    fc: list[float] = []
    psd: list[float] = []
    for row in _read_rows(FIGURE12_CSV):
        fc.append(_finite(row["fc_over_U"], "fc_over_U", FIGURE12_CSV))
        psd.append(_finite(row["PSD"], "PSD", FIGURE12_CSV))
    return fc, psd, dict(FIGURE12_CONDITION)


def figure1315_rows() -> tuple[Figure1315Row, ...]:
    return tuple(
        Figure1315Row(
            Re=int(row["Re"]),
            alpha_deg=float(row["alpha_deg"]),
            st=_finite(row["St_fc_over_U"], "St", FIGURE13_15_CSV),
            st_modified=_finite(row["St_modified"], "St_modified", FIGURE13_15_CSV),
            uncertainty_st=float(row["digitization_uncertainty_St"]),
            uncertainty_st_modified=float(
                row["digitization_uncertainty_St_modified"]
            ),
        )
        for row in _read_rows(FIGURE13_15_CSV)
    )


def figure14_relation() -> tuple[dict[str, float], ...]:
    """Cross-literature diagnostic band (NOT this experiment's GT)."""

    return tuple(
        {
            "alpha_deg": float(row["alpha_deg"]),
            "st_fit": _finite(
                row["St_fit_0p17_over_sin_alpha"], "fit", FIGURE14_CSV
            ),
            "st_lower": _finite(
                row["St_lower_0p15_over_sin_alpha"], "lower", FIGURE14_CSV
            ),
            "st_upper": _finite(
                row["St_upper_0p20_over_sin_alpha"], "upper", FIGURE14_CSV
            ),
        }
        for row in _read_rows(FIGURE14_CSV)
    )


__all__ = [
    "A10_U5_CN",
    "A16_U5_CN",
    "A16_U5_ZMAX_OVER_C",
    "FIGURE12_PEAK_ST",
    "FIGURE12_PEAK_UNCERTAINTY",
    "FROZEN_ROW_COUNTS",
    "FROZEN_SHA256",
    "ST_MODIFIED_CLOSURE_TOL",
    "Figure1315Row",
    "Figure6Row",
    "Figure9Row",
    "figure12_spectrum",
    "figure1315_rows",
    "figure14_relation",
    "figure6_rows",
    "figure6_value",
    "figure9_rows",
    "figure9_value",
    "freeze_sha256",
    "validate",
]
