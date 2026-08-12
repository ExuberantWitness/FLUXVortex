"""Reference checks for the existing clean-room Python LDVM primitive.

The numerical landmarks come from the author-supplied LDVM v2.5
``motion_pr_amp45_k0.2.dat`` and ``force_pr_amp45_k0.2_le.dat`` files.  The
Fortran source and its GPL-covered output files are deliberately not vendored.
The test reconstructs the analytic Eldredge motion and the already-vendored
SD7003 mean camber line instead.

This is a *partial-parity* guard.  ``platform/ldvm_fourier.py`` has the same
LESP/Kelvin/load primitives and reproduces the first shedding event, but it is
not yet an exact time-history port.  In particular it does not apply the
Fortran source's special first-TEV zeroing operation and its public result uses
different wake-sign and pre/post-LESP conventions.  See
``SOURCE_AUDIT_RAMESH_LDVM.md`` for the measured limits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline


PLATFORM = Path(__file__).resolve().parents[1]
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from ldvm_fourier import LDVM2D  # noqa: E402


AUTHOR_OUTPUT_ROWS = 499
AUTHOR_FIRST_LEV_OUTPUT_ROW = 116
AUTHOR_FIRST_LEV_TSTAR = 1.6383540
AUTHOR_FIRST_LEV_ALPHA_DEG = 14.630000
AUTHOR_LESP_CRIT = 0.18
AUTHOR_FIRST_LEV_CL = 3.0860116
AUTHOR_FIRST_LEV_CD = 0.59517464
AUTHOR_FINAL_LEV_COUNT = 174


def _eldredge_reference_motion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the bundled 45-degree, K=0.2 ramp-hold-return motion."""

    amplitude = np.deg2rad(45.0)
    reduced_pitch_rate = 0.2
    smoothing = 11.0
    t1 = 1.0
    t2 = t1 + amplitude / (2.0 * reduced_pitch_rate)
    t3 = (
        t2
        + np.pi * amplitude / (4.0 * reduced_pitch_rate)
        - amplitude / (2.0 * reduced_pitch_rate)
    )
    t4 = t3 + amplitude / (2.0 * reduced_pitch_rate)

    def eldredge_shape(time: np.ndarray | float) -> np.ndarray:
        time = np.asarray(time)
        return (
            np.log(np.cosh(smoothing * (time - t1)))
            + np.log(np.cosh(smoothing * (time - t4)))
            - np.log(np.cosh(smoothing * (time - t2)))
            - np.log(np.cosh(smoothing * (time - t3)))
        )

    # The supplied motion contains 500 samples from t*=0 through t*=t4+1.
    time = np.linspace(0.0, t4 + 1.0, 500)
    maximum = float(eldredge_shape(0.5 * (t2 + t3)))
    alpha = amplitude * eldredge_shape(time) / maximum

    # ldvm.f95:181-187 uses first-order backward differences and copies the
    # second sample's derivative into the unused first sample.
    alpha_rate = np.empty_like(alpha)
    alpha_rate[1:] = np.diff(alpha) / np.diff(time)
    alpha_rate[0] = alpha_rate[1]
    return time, alpha, alpha_rate


def _author_sd7003_camber_slope(model: LDVM2D) -> np.ndarray:
    """Mirror ``calc_camberslope`` using the repository's SD7003 coordinates."""

    coordinates = np.loadtxt(
        ROOT / "researchpaper" / "uiuc_airfoils" / "sd7003.dat",
        skiprows=1,
    )
    surface_coordinate = np.concatenate(
        ([0.0], np.cumsum(np.abs(np.diff(coordinates[:, 0]))))
    )
    surface_y = CubicSpline(
        surface_coordinate,
        coordinates[:, 1],
        bc_type=((1, 0.0), (1, 0.0)),
    )

    x_over_c = model.xs / model.c
    upper_path = surface_y(x_over_c)
    lower_path = surface_y(1.0 + x_over_c)
    camber = 0.5 * (upper_path[::-1] + lower_path) * model.c

    slope = np.empty_like(camber)
    slope[0] = (camber[1] - camber[0]) / (model.xs[1] - model.xs[0])
    slope[1:] = np.diff(camber) / np.diff(model.xs)
    return slope


@lru_cache(maxsize=1)
def _reference_replay() -> dict[str, np.ndarray | int]:
    time, alpha, alpha_rate = _eldredge_reference_motion()
    model = LDVM2D(
        U=1.0,
        c=1.0,
        ndiv=70,
        naterm=45,
        dt=float(time[1] - time[0]),
        lesp_crit=AUTHOR_LESP_CRIT,
        pivot_xc=0.0,
        core_rc=0.02,
        max_wake=100_000,
    )
    model.dzc = _author_sd7003_camber_slope(model)

    rows: list[list[float]] = []
    kelvin_residuals: list[float] = []
    cl_identity_residuals: list[float] = []
    cd_identity_residuals: list[float] = []
    cs_identity_residuals: list[float] = []

    # The Fortran program advances motion rows 2..500 and therefore emits 499
    # force rows.  ``step`` follows the same no-output-at-initial-state contract.
    for motion_index in range(1, len(time)):
        result = model.step(
            float(alpha[motion_index]),
            float(alpha_rate[motion_index]),
            0.0,
        )
        output_row = motion_index
        rows.append(
            [
                float(output_row),
                float(time[motion_index]),
                float(np.rad2deg(alpha[motion_index])),
                float(result["A0"]),
                float(result["lesp"]),
                float(result["CNf"]),
                float(result["CSf"]),
                float(result["CLf"]),
                float(result["CDf"]),
                float(result["gamb"]),
                float(result["n_lev"]),
                float(result["n_tev"]),
            ]
        )

        kelvin_residuals.append(
            float(np.sum(model.tg) + np.sum(model.lg) + model.gam_lost - result["gamb"])
        )
        cl_identity_residuals.append(
            float(
                result["CLf"]
                - result["CNf"] * np.cos(alpha[motion_index])
                - result["CSf"] * np.sin(alpha[motion_index])
            )
        )
        cd_identity_residuals.append(
            float(
                result["CDf"]
                - result["CNf"] * np.sin(alpha[motion_index])
                + result["CSf"] * np.cos(alpha[motion_index])
            )
        )
        cs_identity_residuals.append(
            float(result["CSf"] - 2.0 * np.pi * result["A0"] ** 2)
        )

    return {
        "rows": np.asarray(rows),
        "kelvin": np.asarray(kelvin_residuals),
        "cl_identity": np.asarray(cl_identity_residuals),
        "cd_identity": np.asarray(cd_identity_residuals),
        "cs_identity": np.asarray(cs_identity_residuals),
        "final_lev_count": len(model.lg),
    }


def test_author_reference_replay_is_finite() -> None:
    replay = _reference_replay()
    rows = np.asarray(replay["rows"])

    assert rows.shape == (AUTHOR_OUTPUT_ROWS, 12)
    assert np.all(np.isfinite(rows))
    assert np.all(np.diff(rows[:, 10]) >= 0.0)
    assert np.all(np.diff(rows[:, 11]) == 1.0)
    assert rows[-1, 11] == AUTHOR_OUTPUT_ROWS


def test_author_reference_first_lesp_cap_is_reproduced() -> None:
    replay = _reference_replay()
    rows = np.asarray(replay["rows"])
    first_index = int(np.flatnonzero(rows[:, 10] > 0.0)[0])
    first = rows[first_index]

    assert int(first[0]) == AUTHOR_FIRST_LEV_OUTPUT_ROW
    assert np.isclose(first[1], AUTHOR_FIRST_LEV_TSTAR, atol=5.0e-7)
    assert np.isclose(first[2], AUTHOR_FIRST_LEV_ALPHA_DEG, atol=2.0e-6)
    assert first[4] > AUTHOR_LESP_CRIT  # pre-constraint LESP triggers shedding
    assert np.isclose(first[3], AUTHOR_LESP_CRIT, atol=5.0e-14)
    assert np.isclose(first[7], AUTHOR_FIRST_LEV_CL, atol=2.0e-3)
    assert np.isclose(first[8], AUTHOR_FIRST_LEV_CD, atol=6.0e-4)

    # Measured partial-parity limit: this clean-room replay currently creates
    # 172 LEVs versus 174 in the author source.  Keep the discrepancy visible
    # without falsely declaring exact full-history parity.
    assert abs(int(replay["final_lev_count"]) - AUTHOR_FINAL_LEV_COUNT) <= 2


def test_kelvin_and_force_definition_identities_close() -> None:
    replay = _reference_replay()

    assert np.max(np.abs(replay["kelvin"])) < 2.0e-12
    assert np.max(np.abs(replay["cl_identity"])) < 2.0e-14
    assert np.max(np.abs(replay["cd_identity"])) < 2.0e-14
    assert np.max(np.abs(replay["cs_identity"])) < 2.0e-14
