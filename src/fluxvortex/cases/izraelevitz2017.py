"""Izraelevitz et al. (2017) Fig.14 / Scherer (1968) — rectangular AR=3
finite wing in forward water flow, prescribed heave + pitch about x/c=0.75.

12 unique motion conditions, 14 Scherer experimental markers.
Rigid rectangular planform with two free wing tips in forward water flow
at Re≈310,000 (U=3.048 m/s) — a finite-wing forward-flight case with a
nonzero freestream, not a zero-freestream flapper.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib

# Frozen paper/experiment truth (HANDOFF_IZRAELEVITZ2017 §2)
CHORD_M = 0.1016           # 4 inches
SPAN_M = 0.3048            # 12 inches (full span, tip to tip)
AREA_M2 = 0.03096768       # b*c
AR = 3.0
PIVOT_FRACTION = 0.75      # pitch axis at x/c = 0.75
FREESTREAM_M_S = 3.048     # 10 ft/s forward water flow
RHO_KG_M3 = 1000.0         # water
NU_M2_S = 1.0e-6           # water kinematic viscosity
REYNOLDS = 309676.8        # U*c/nu
HEAVE_AMPLITUDE_M = 0.06096  # h/c = 0.6 → h = 0.6 * c
FREQUENCY_HZ = 5.0         # from St=0.2, J'=6
OMEGA_RAD_S = 31.41592653589793  # 2*pi*f
PROFILE_DRAG_CD0 = 0.057   # Izraelevitz Fig.14 convention (frozen)
LESP_THRESHOLD = 0.2393    # sin(CLmax/CLa) = sin(0.90/0.065)

GT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/forward_flight_large_pitch/reproductions/"
    "unified_fluxv_upgrade_20260812/source_data/"
    "izraelevitz2017_fig14_digitized.csv"
)
GT_SHA256 = (
    "993f410c5d4857a221e57c616bf45beb5eaef5391a2deafb0b6e48e6d083b3cf"
)


def validate_gt() -> None:
    """Verify the GT CSV exists and its SHA matches the frozen value."""
    if not GT_PATH.is_file():
        raise FileNotFoundError(GT_PATH)
    observed = hashlib.sha256(GT_PATH.read_bytes()).hexdigest()
    if observed != GT_SHA256:
        raise RuntimeError(f"Izraelevitz GT CSV drift: {observed}")


@dataclass(frozen=True)
class IzraCaseConfig:
    """One of 12 unique motion conditions from Fig.14."""
    case_id: str
    theta_max_deg: float
    phase_offset_deg: float
    chord_m: float = CHORD_M
    span_m: float = SPAN_M
    area_m2: float = AREA_M2
    aspect_ratio: float = AR
    pivot_fraction_chord: float = PIVOT_FRACTION
    freestream_m_s: float = FREESTREAM_M_S
    rho_kg_m3: float = RHO_KG_M3
    nu_m2_s: float = NU_M2_S
    reynolds: float = REYNOLDS
    heave_amplitude_m: float = HEAVE_AMPLITUDE_M
    frequency_hz: float = FREQUENCY_HZ
    omega_rad_s: float = OMEGA_RAD_S
    profile_drag_coefficient: float = PROFILE_DRAG_CD0
    lesp_threshold: float = LESP_THRESHOLD
    ground_truth_path: Path = GT_PATH
    ground_truth_sha256: str = GT_SHA256

    @property
    def description(self) -> str:
        return (
            f"rectangular AR=3 finite wing, forward water flow "
            f"U={self.freestream_m_s} m/s, Re={self.reynolds:.0f}, "
            f"heave h/c=0.6, pitch theta_max={self.theta_max_deg} deg "
            f"at psi={self.phase_offset_deg} deg, pivot x/c=0.75"
        )


# 12 unique conditions (HANDOFF §2.4): theta_max ∈ {15°, 25°}
# 15° family: psi = 15, 30, 45, 60, 75, 90, 105  (7 conditions)
# 25° family: psi = 45, 60, 75, 90, 105           (5 conditions)
_IZRA_15_PHASES = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0)
_IZRA_25_PHASES = (45.0, 60.0, 75.0, 90.0, 105.0)

IZRA_CASES = {}
for _theta in (15.0, 25.0):
    _phases = _IZRA_15_PHASES if _theta == 15.0 else _IZRA_25_PHASES
    for _psi in _phases:
        _cid = f"IZRA-{int(_theta):02d}-{int(_psi):03d}"
        IZRA_CASES[_cid] = IzraCaseConfig(
            case_id=_cid,
            theta_max_deg=_theta,
            phase_offset_deg=_psi,
        )
